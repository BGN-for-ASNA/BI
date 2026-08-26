"""End-to-end checks for BayesForge.run_simulations.

Run with:  python Test/Parallel/test_run_simulations.py
"""
import os
import warnings

import numpy as np
import pandas as pd

from BayesForge import bf, grid, run_simulations


def one_sim(N, seed, **_):
    """Simulate y ~ Normal(a + b*x, s), fit it back, report truth vs estimate."""
    m = bf(cores=1, rand_seed=seed, print_devices_found=False)

    a_true, b_true, s_true = 1.5, -0.8, 1.0
    x = np.asarray(m.dist.normal(0.0, 1.0, sample=True, shape=(N,)))
    y = np.asarray(m.dist.normal(a_true + b_true * x, s_true, sample=True))

    def linreg(x=None, y=None):
        a = m.dist.normal(0.0, 5.0, name="a")
        b = m.dist.normal(0.0, 5.0, name="b")
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(a + b * x, s, name="y", obs=y)

    m.fit(linreg, obs=dict(x=x, y=y), num_warmup=200, num_samples=200,
          num_chains=1, progress_bar=False, shard=False)
    est = m.summary()["mean"]
    truth = {"a": a_true, "b": b_true, "s": s_true}
    return [{"parameter": p, "simulated": truth[p], "estimations": float(est[p])}
            for p in ("a", "b", "s")]


def boom(N, seed, sim, **_):
    if sim == 1:
        raise ValueError("intentional failure")
    return {"ok": True}


def devices_seen(seed, **_):
    import jax
    m = bf(rand_seed=seed, print_devices_found=False)  # bare bf(): BF_MAX_CORES path
    return {"n_devices": jax.local_device_count(), "bf_n_devices": m.n_devices}


def cpu_probe(seed, **_):
    """Report how much of the machine this worker can actually touch."""
    import os
    import time
    import jax.numpy as jnp
    x = jnp.ones((512, 512))
    (x @ x).block_until_ready()          # force the XLA CPU thread pool up
    time.sleep(1.0)                      # hold the worker so the pool spreads
    return {"affinity": len(os.sched_getaffinity(0)),
            "cpus": tuple(sorted(os.sched_getaffinity(0))),
            "threads": len(os.listdir(f"/proc/{os.getpid()}/task"))}


def four_chains(N, seed, chain_method, **_):
    """A 4-chain fit inside a worker."""
    import jax
    m = bf(rand_seed=seed, print_devices_found=False)   # bare bf(): BF_MAX_CORES
    x = np.asarray(m.dist.normal(0.0, 1.0, sample=True, shape=(N,)))
    y = np.asarray(m.dist.normal(1.5 - 0.8 * x, 1.0, sample=True))

    def linreg(x=None, y=None):
        a = m.dist.normal(0.0, 5.0, name="a")
        b = m.dist.normal(0.0, 5.0, name="b")
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(a + b * x, s, name="y", obs=y)

    m.fit(linreg, obs=dict(x=x, y=y), num_warmup=300, num_samples=300,
          num_chains=4, chain_method=chain_method, progress_bar=False, shard=False)
    su = m.summary()
    return {"devices": jax.local_device_count(),
            "chains": m.posteriors_by_chain["a"].shape[0],
            "draws": m.posteriors_by_chain["a"].shape[1],
            "r_hat_a": float(su.loc["a", "r_hat"])}


def chains_from_arg(N, seed, num_chains, **_):
    """Takes num_chains back from the runner instead of hard-coding it."""
    import os
    import jax
    m = bf(rand_seed=seed, print_devices_found=False)
    x = np.asarray(m.dist.normal(0.0, 1.0, sample=True, shape=(N,)))
    y = np.asarray(m.dist.normal(1.5 - 0.8 * x, 1.0, sample=True))

    def linreg(x=None, y=None):
        a = m.dist.normal(0.0, 5.0, name="a")
        b = m.dist.normal(0.0, 5.0, name="b")
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(a + b * x, s, name="y", obs=y)

    m.fit(linreg, obs=dict(x=x, y=y), num_warmup=300, num_samples=300,
          num_chains=num_chains, chain_method="parallel",
          progress_bar=False, shard=False)
    return {"devices": jax.local_device_count(),
            "cores": len(os.sched_getaffinity(0)),
            "chains": m.posteriors_by_chain["a"].shape[0],
            "r_hat_a": float(m.summary().loc["a", "r_hat"])}


def check(label, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


if __name__ == "__main__":
    scen = grid(N=[60, 120], reps=2)

    print("\n--- 1/2. serial vs parallel determinism ---")
    a = run_simulations(one_sim, scen, backend="serial", progress=False)
    b = run_simulations(one_sim, scen, workers=4, progress=False)

    key = ["sim", "parameter"]
    cols = key + ["N", "rep", "seed", "simulated", "estimations"]
    aa = a[cols].sort_values(key).reset_index(drop=True)
    bb = b[cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(aa, bb)
    check("parallel results identical to serial", True)
    check("no errors", a["error"].isna().all() and b["error"].isna().all())

    print("\n--- 2. scenario columns carried through ---")
    check("scenario+meta columns present",
          {"N", "sim", "rep", "seed", "worker", "elapsed_s", "error"} <= set(b.columns))
    check("3 rows per sim", (b.groupby("sim").size() == 3).all())
    check("each (N, rep) cell appears once",
          (b[b.parameter == "a"].groupby(["N", "rep"]).size() == 1).all())
    check("recovery is sane (|err| < 0.5)",
          (b.simulated - b.estimations).abs().max() < 0.5)
    check("used >1 worker", b["worker"].nunique() > 1)

    print("\n--- 3. device budget in workers ---")
    d = run_simulations(devices_seen, 4, workers=2, cores_per_worker=2, progress=False)
    check("worker sees cores_per_worker devices", (d["n_devices"] == 2).all())
    check("bare bf() honours BF_MAX_CORES", (d["bf_n_devices"] == 2).all())

    print("\n--- 4. error isolation ---")
    e = run_simulations(boom, grid(N=[10], reps=4), workers=2,
                        on_error="record", progress=False)
    check("1 row carries an error", e["error"].notna().sum() == 1)
    check("the other 3 succeeded", (e["ok"] == True).sum() == 3)
    check("failing sim identified", e.loc[e.error.notna(), "sim"].tolist() == [1])
    try:
        run_simulations(boom, grid(N=[10], reps=4), workers=2,
                        on_error="raise", progress=False)
        check("on_error='raise' propagates", False)
    except RuntimeError as exc:
        check("on_error='raise' propagates", "intentional failure" in str(exc))

    print("\n--- 5. closure shipping (model defined inside fn) ---")
    check("closure model fitted in workers", len(b) == len(scen) * 3)

    print("\n--- 6. fn_kwargs broadcast ---")
    big = np.arange(1_000_000, dtype=float)
    k = run_simulations(lambda sim, seed, shared, offset, **_:
                        {"v": float(shared[sim]) + offset, "n": shared.size},
                        6, workers=3, fn_kwargs=dict(shared=big, offset=100.0),
                        progress=False)
    check("fn_kwargs reached every worker", (k["n"] == 1_000_000).all())
    check("fn_kwargs values correct",
          k.sort_values("sim")["v"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    print("\n--- 7. oversubscription is capped ---")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_simulations(lambda sim, seed, **_: {"s": sim}, 20, workers=20,
                        cores_per_worker=4, progress=False)
    check("capping warning raised",
          any("exceeds the" in str(x.message) for x in caught))

    print("\n--- 8. workers are pinned, not oversubscribed ---")
    if hasattr(os, "sched_setaffinity"):
        n_cpu = len(os.sched_getaffinity(0))
        w, cpw = 4, 2
        # CPUs are spread across the workers so none idle -- the block is the
        # machine divided by the worker count, floored at cores_per_worker.
        expect = max(cpw, n_cpu // w)
        p = run_simulations(cpu_probe, w, workers=w, cores_per_worker=cpw,
                            progress=False)
        print(p[["affinity", "threads"]].to_string())
        check(f"each worker pinned to {expect} CPUs", (p["affinity"] == expect).all())
        # Compare per worker process, not per row: the pool may serve several
        # tasks from one worker.
        by_worker = p.groupby("worker")["cpus"].first()
        check("distinct workers hold distinct CPU sets",
              by_worker.nunique() == len(by_worker))
        check("no CPU shared between workers",
              len({c for s in by_worker for c in s}) == expect * len(by_worker))
        check("thread count bounded, not machine-wide", (p["threads"] < 80).all())
    else:
        print("SKIP  (no sched_setaffinity on this platform)")

    print("\n--- 9. multi-chain fits inside workers ---")
    # chain_method='parallel' needs one device per chain.
    cp = run_simulations(four_chains, grid(N=[300], chain_method=["parallel"], reps=4),
                         workers=4, cores_per_worker=4, progress=False)
    check("no errors (parallel chains)", cp["error"].isna().all())
    check("worker exposes 4 devices", (cp["devices"] == 4).all())
    check("4 real chains in the posterior", (cp["chains"] == 4).all())
    check("r_hat computed across chains", cp["r_hat_a"].notna().all())

    # vectorized vmaps the chains, so 1 device is enough and every core is free
    # to run a different simulation -- the recommended setup.
    cv = run_simulations(four_chains, grid(N=[300], chain_method=["vectorized"], reps=4),
                         workers=4, cores_per_worker=1, progress=False)
    check("no errors (vectorized chains)", cv["error"].isna().all())
    check("4 chains on a single device", ((cv["chains"] == 4) & (cv["devices"] == 1)).all())
    check("same draw count either way",
          set(cp["draws"]) == set(cv["draws"]) == {300})

    print("\n--- 10. num_chains reserves a core per chain ---")
    if hasattr(os, "sched_setaffinity"):
        n_cpu = len(os.sched_getaffinity(0))
        nc = 4
        q = run_simulations(chains_from_arg, grid(N=[300], reps=2 * (n_cpu // nc)),
                            num_chains=nc, progress=False)
        check(f"workers == n_cpu // num_chains ({n_cpu // nc})",
              q["worker"].nunique() == n_cpu // nc)
        check("each worker holds num_chains cores", (q["cores"] == nc).all())
        check("each worker exposes num_chains devices", (q["devices"] == nc).all())
        check("fits drew num_chains chains", (q["chains"] == nc).all())
        check("num_chains echoed into the results", (q["num_chains"] == nc).all())
        check("no errors", q["error"].isna().all())
        check("r_hat computed", q["r_hat_a"].notna().all())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_simulations(lambda sim, seed, num_chains, **_: {"n": num_chains},
                            2, num_chains=4, cores_per_worker=1, progress=False)
        check("warns when cores_per_worker < num_chains",
              any("below num_chains" in str(x.message) for x in caught))
    else:
        print("SKIP  (no sched_setaffinity on this platform)")

    print("\n--- 11. returning a bf is rejected ---")
    r = run_simulations(lambda seed, **_: bf(cores=1, print_devices_found=False),
                        2, workers=2, on_error="record", progress=False)
    check("bf return produces a clear error",
          r["error"].notna().all()
          and "cannot cross a process boundary" in r["error"].iloc[0])

    print("\nAll checks passed.")
