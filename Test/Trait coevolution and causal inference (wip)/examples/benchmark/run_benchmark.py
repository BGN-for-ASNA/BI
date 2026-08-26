"""Serial BF-vs-Stan benchmark for the two GDPM examples.

The earlier timings were taken with several fits sharing the box, so they are
not usable as a benchmark. This runs everything one at a time and records the
diagnostics needed to explain a difference rather than just measure one.

Three arms per example:

  stan      cmdstanr, the reference
  bf        BayesForge m.fit -- the production path. Now also reports
            sampler diagnostics via m.sampler_stats()
  numpyro   the identical model function driven through numpyro's MCMC
            directly, instrumented with num_steps / energy / diverging.
            Same sampler as bf, so bf-vs-numpyro isolates wrapper overhead
            and numpyro-vs-stan is the like-for-like sampler comparison.

Plus one diagnostic arm, primates only:

  numpyro_nc   terminal_drift reparameterised non-centered. Same posterior,
               different geometry. Stan reports E-BFMI 0.29-0.34 and max
               treedepth on 1999/2000 draws for the centered form, so this
               quantifies how much of the cost is the funnel rather than
               the engine.

Results append to results.csv after every run, so partial output is usable.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/sebastian_sosa/phylo/examples"
OUT = os.path.join(ROOT, "benchmark")
sys.path.insert(0, ROOT)


# The parameter blocks Stan saves. ESS is restricted to these so both engines
# are scored on identical quantities; numpyro otherwise also returns z_drift,
# terminal_drift and other latents Stan never wrote out.
COMPARABLE = {
    "cichlid": {"A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma"},
    "primates": {"A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma",
                 "alpha", "shape", "lambda_free", "cor_R"},
}


def ess_rhat(draws_by_chain, keep=None):
    """ESS-bulk and R-hat per scalar, from (chains, draws, ...) arrays."""
    import numpyro.diagnostics as diag
    rows = {}
    for name, a in draws_by_chain.items():
        if keep is not None and name.split("[")[0] not in keep:
            continue
        a = np.asarray(a)
        if a.ndim < 2:
            continue
        event = a.shape[2:]
        for idx in (np.ndindex(*event) if event else [()]):
            x = a[(slice(None), slice(None)) + idx]
            if np.std(x) < 1e-12:
                continue
            key = name + ("" if not idx else "[" + ",".join(str(i + 1) for i in idx) + "]")
            rows[key] = (float(diag.effective_sample_size(x)),
                         float(diag.split_gelman_rubin(x)))
    return rows


def summarise(tag, example, engine, seed, wall, by_chain, extra=None):
    er = ess_rhat(by_chain, keep=COMPARABLE[example])
    ess = np.array([v[0] for v in er.values()])
    rhat = np.array([v[1] for v in er.values()])
    worst = min(er, key=lambda k: er[k][0]) if er else ""
    row = dict(example=example, engine=engine, seed=seed, wall_s=round(wall, 1),
               n_params=len(er), min_ess=ess.min() if len(ess) else np.nan,
               median_ess=np.median(ess) if len(ess) else np.nan,
               min_ess_param=worst, max_rhat=rhat.max() if len(rhat) else np.nan,
               min_ess_per_s=(ess.min() / wall) if len(ess) else np.nan,
               median_ess_per_s=(np.median(ess) / wall) if len(ess) else np.nan)
    row.update(extra or {})
    path = os.path.join(OUT, "results.csv")
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print("  ->", {k: row[k] for k in ("wall_s", "min_ess", "median_ess",
                                       "min_ess_per_s", "max_rhat")})
    return row


def run_numpyro(example, seed, noncentered=False, warmup=500, samples=500,
                chains=4):
    """Identical model function, driven through numpyro directly."""
    import jax
    from numpyro.infer import MCMC, NUTS

    if example == "cichlid":
        sys.path.insert(0, f"{ROOT}/cichlid/bf")
        import cichlid_bf as M
        d = M.load_data()
        model = M.make_model(d["N_seg"], d["N_tips"], d["J"])
        keys = ("y", "node_seq", "parent", "ts", "tip", "tip_id", "off_rows",
                "off_cols", "level_seg", "level_valid")
    else:
        sys.path.insert(0, f"{ROOT}/primates/bf")
        import primates_bf as M
        d = M.load_data()
        model = M.make_model(d["N_seg"], d["N_tips"], d["N_obs"], d["J"], d["K"],
                             noncentered=noncentered)
        keys = ("y", "observed", "idx_longevity", "idx_maturity", "y_mean",
                "node_seq", "parent", "ts", "tip", "tip_id", "off_rows",
                "off_cols", "level_seg", "level_valid")
    obs = {k: d[k] for k in keys}

    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=chains, chain_method="parallel",
                progress_bar=False)
    t = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed),
             extra_fields=("num_steps", "energy", "diverging", "adapt_state.step_size"),
             **obs)
    # JAX dispatches asynchronously: mcmc.run returns as soon as the work is
    # enqueued, so the timer must wait on the actual arrays or it measures
    # dispatch (~10 s) instead of sampling (~1 h).
    by_chain = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(by_chain)
    wall = time.perf_counter() - t
    ex = mcmc.get_extra_fields(group_by_chain=True)
    energy = np.asarray(ex["energy"])
    # E-BFMI per chain, same definition Stan reports
    ebfmi = [float(np.sum(np.diff(e) ** 2) / (len(e) * np.var(e))) for e in energy]
    steps = np.asarray(ex["num_steps"])
    extra = dict(total_leapfrog=int(steps.sum()),
                 mean_leapfrog_per_iter=float(steps.mean()),
                 pct_at_treedepth10=float(100 * (steps >= 1023).mean()),
                 divergences=int(np.asarray(ex["diverging"]).sum()),
                 min_ebfmi=round(min(ebfmi), 4),
                 step_size=float(np.asarray(ex["adapt_state.step_size"]).ravel()[-1]))
    np.savez(os.path.join(OUT, f"{example}_numpyro{'_nc' if noncentered else ''}_s{seed}.npz"),
             **{k: np.asarray(v) for k, v in by_chain.items()})
    tag = "numpyro_nc" if noncentered else "numpyro"
    print(f"  leapfrog/iter {extra['mean_leapfrog_per_iter']:.0f} | "
          f"{extra['pct_at_treedepth10']:.1f}% at cap | E-BFMI {extra['min_ebfmi']} | "
          f"div {extra['divergences']}")
    return summarise(tag, example, tag, seed, wall, by_chain, extra)


def run_bf(example, seed, warmup=500, samples=500, chains=4):
    """The production BayesForge path. Wall clock only -- no extra_fields hook."""
    if example == "cichlid":
        sys.path.insert(0, f"{ROOT}/cichlid/bf")
        import cichlid_bf as M
        d = M.load_data()
        model = M.make_model(d["N_seg"], d["N_tips"], d["J"])
        keys = ("y", "node_seq", "parent", "ts", "tip", "tip_id", "off_rows",
                "off_cols", "level_seg", "level_valid")
    else:
        sys.path.insert(0, f"{ROOT}/primates/bf")
        import primates_bf as M
        d = M.load_data()
        model = M.make_model(d["N_seg"], d["N_tips"], d["N_obs"], d["J"], d["K"])
        keys = ("y", "observed", "idx_longevity", "idx_maturity", "y_mean",
                "node_seq", "parent", "ts", "tip", "tip_id", "off_rows",
                "off_cols", "level_seg", "level_valid")
    obs = {k: d[k] for k in keys}

    import jax
    t = time.perf_counter()
    M.m.fit(model=model, obs=obs, num_warmup=warmup, num_samples=samples,
            num_chains=chains, target_accept_prob=0.95, seed=seed,
            progress_bar=False)
    jax.block_until_ready(M.m.posteriors_by_chain_full)   # see note in run_numpyro
    wall = time.perf_counter() - t
    by_chain = {k: np.asarray(v) for k, v in M.m.posteriors_by_chain_full.items()}
    # m.fit now retains extra_fields, so the production path reports the same
    # sampler diagnostics as the direct-numpyro arm
    extra = None
    try:
        st = M.m.sampler_stats()
        extra = dict(total_leapfrog=st["leapfrog_total"],
                     mean_leapfrog_per_iter=st["leapfrog_per_iter"],
                     pct_at_treedepth10=st["pct_at_max_treedepth"],
                     divergences=st["divergences"], min_ebfmi=st["ebfmi"],
                     step_size=st["step_size"])
        print(f"  leapfrog/iter {st['leapfrog_per_iter']:.0f} | "
              f"{st['pct_at_max_treedepth']:.1f}% at cap | E-BFMI {st['ebfmi']} | "
              f"div {st['divergences']}")
    except Exception as e:
        print(f"  (sampler_stats unavailable: {e})")
    return summarise("bf", example, "bf", seed, wall, by_chain, extra)


def run_stan(example, seed, warmup=500, samples=500, chains=4):
    script = os.path.join(OUT, "_stan_run.R")
    t = time.perf_counter()
    r = subprocess.run(["Rscript", script, example, str(seed), str(warmup),
                        str(samples), str(chains)],
                       capture_output=True, text=True, cwd=OUT)
    wall = time.perf_counter() - t
    if r.returncode != 0:
        print("  STAN FAILED\n", r.stdout[-2000:], r.stderr[-2000:])
        return None
    info = json.load(open(os.path.join(OUT, f"{example}_stan_s{seed}.json")))
    by_chain = {k: np.array(v) for k, v in
                np.load(os.path.join(OUT, f"{example}_stan_s{seed}.npz")).items()}
    extra = dict(total_leapfrog=info["total_leapfrog"],
                 mean_leapfrog_per_iter=info["mean_leapfrog"],
                 pct_at_treedepth10=info["pct_treedepth"],
                 divergences=info["divergences"],
                 min_ebfmi=info["min_ebfmi"], step_size=info["step_size"])
    print(f"  leapfrog/iter {extra['mean_leapfrog_per_iter']:.0f} | "
          f"{extra['pct_at_treedepth10']:.1f}% at cap | E-BFMI {extra['min_ebfmi']} | "
          f"div {extra['divergences']}")
    return summarise("stan", example, "stan", seed, info["sampling_wall"],
                     by_chain, extra)


PLAN = [
    # most diagnostic first, so the question is answered before the long tail
    ("primates", "numpyro", 1, dict(noncentered=False)),
    ("primates", "numpyro_nc", 1, dict(noncentered=True)),
    ("primates", "stan", 1, {}),
    ("primates", "bf", 1, {}),
    ("cichlid", "numpyro", 1, {}),
    ("cichlid", "stan", 1, {}),
    ("cichlid", "bf", 1, {}),
    ("primates", "numpyro", 2, dict(noncentered=False)),
    ("primates", "stan", 2, {}),
    ("cichlid", "numpyro", 2, {}),
    ("cichlid", "stan", 2, {}),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on 'example:engine'")
    a = ap.parse_args()

    for example, engine, seed, kw in PLAN:
        tag = f"{example}:{engine}:seed{seed}"
        if a.only and a.only not in tag:
            continue
        print(f"\n=== {tag} ===", flush=True)
        try:
            if engine == "stan":
                run_stan(example, seed)
            elif engine == "bf":
                run_bf(example, seed)
            else:
                run_numpyro(example, seed, **kw)
        except Exception as e:                       # keep the queue moving
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print("\nbenchmark queue complete")
