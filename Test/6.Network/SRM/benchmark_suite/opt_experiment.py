"""Parametrized SRM optimization experiment with correctness + speed logging.

Env vars:
  EXP_ID      label for the log row
  MODE        ref | shard | svinit | shard_svinit
  N           nodes (default 100)
  CORES       virtual CPU devices (must divide N for shard modes; default 20)
  WARM, SAMP  warmup / sampling iters (default 600/600)
  CHAINS      default 4
  MAXTREE     "10" or "7,10"  (warmup,sampling cap)  default "10"
  TACC        target_accept_prob   default 0.8
  SVI_STEPS   SVI steps for *_svinit modes (default 2000)
  REF_FILE    json path for reference structural means (default ref_N{N}.json)

Records time, Min/Mean ESS, max R-hat, divergences, and — when a reference exists
and MODE != ref — the posterior agreement (max standardized mean diff over
structural parameters). Appends a row to opt_results.csv and prints a markdown
log line. MODE=ref additionally SAVES the structural means as the reference.
"""
import os, sys, time, json
N      = int(os.environ.get("N", "100"))
CORES  = int(os.environ.get("CORES", "20"))
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + f" --xla_force_host_platform_device_count={CORES}").strip()
os.environ.setdefault("BF_SHARD_CHECK", "0")   # skip safeguards in timing

import numpy as np
import jax, jax.numpy as jnp
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from numpyro.diagnostics import summary as np_summary
from numpyro.infer import init_to_value

EXP_ID = os.environ.get("EXP_ID", "exp")
MODE   = os.environ.get("MODE", "ref")
WARM   = int(os.environ.get("WARM", "600"))
SAMP   = int(os.environ.get("SAMP", "600"))
CHAINS = int(os.environ.get("CHAINS", "4"))
TACC   = float(os.environ.get("TACC", "0.8"))
SVI_STEPS = int(os.environ.get("SVI_STEPS", "2000"))
SEED   = 42
N_GRP_ANY, N_GRP_MERICA = 1, 3
_mt = os.environ.get("MAXTREE", "10")
MAXTREE = tuple(int(x) for x in _mt.split(",")) if "," in _mt else int(_mt)
HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.environ.get("REF_FILE", os.path.join(HERE, f"ref_N{N}.json"))
STRUCTURAL = ["sender_effects", "receiver_effects", "dyad_effects",
              "b_intercept", "b_Merica", "sr_sigma", "dr_sigma"]

m = bf(platform="cpu", cores=CORES, rand_seed=False, print_devices_found=True)
print(f"[{EXP_ID}] MODE={MODE} N={N} cores={CORES} warm={WARM} samp={SAMP} "
      f"maxtree={MAXTREE} tacc={TACC}")


def simulate():
    np.random.seed(SEED)
    mt = bf("cpu", rand_seed=False, print_devices_found=False)
    wf = np.random.normal(0, 1, size=(N, 4)); wt = np.random.normal(0, 1, size=(N, 4))
    dy = np.random.binomial(1, 0.3, size=(N, N, 3))
    for k in range(3): np.fill_diagonal(dy[:, :, k], 0)
    dy[:, :, 2] = dy[:, :, 0] * dy[:, :, 1]
    dedg = jnp.stack([mt.net.mat_to_edgl(dy[:, :, k]) for k in range(3)], axis=2)
    Any_np = np.zeros(N, dtype=int)
    Mer_np = np.random.choice([0, 1, 2], size=(N,), p=[0.25, 0.5625, 0.1875])
    Nby_a = np.array([N]); Nby_m = np.array([np.sum(Mer_np == i) for i in range(3)])
    Bi = mt.net.block_model(Any_np, 1, jnp.array(Nby_a), sample=True, name="intercept")
    Bc = mt.net.block_model(Mer_np, 3, jnp.array(Nby_m), sample=True, name="category")
    sr = mt.net.sender_receiver(jnp.array(wf), jnp.array(wt), s_mu=0.4, r_mu=-0.4, sample=True)
    dr = mt.net.dyadic_effect(dedg, d_sd=2.5, sample=True)
    net = jnp.array(mt.dist.bernoulli(logits=Bi + Bc + sr + dr, sample=True))
    Any = jnp.array(Any_np, dtype=jnp.int32); Mer = jnp.array(Mer_np, dtype=jnp.int32)
    _, na = jnp.unique(Any, return_counts=True); _, nm = jnp.unique(Mer, return_counts=True)
    return dict(network_edgl=net,
                Y_mat=NeteffectMatrix.edgelist_to_matrix_outcome(net, N),
                dyadic_preds_mat=NeteffectMatrix.edgelist_to_matrix_predictors(dedg, N),
                sender_preds=jnp.array(wf), receiver_preds=jnp.array(wt),
                Any=Any, Merica=Mer, N_per_grp_Any=na, N_per_grp_Merica=nm,
                N_dyads=int(net.shape[0]))


data = simulate()
mask = (1 - jnp.eye(N)).astype(jnp.float32)
EID, DIR = NeteffectMatrix.dyad_index_maps(N)


def model_ref(network_edgl, dyadic_preds_mat, sender_preds, receiver_preds,
              Any, Merica, N_per_grp_Any, N_per_grp_Merica, **_):
    Ba = NeteffectMatrix.block_model(Any, 1, N_per_grp_Any, name="intercept")
    Bm = NeteffectMatrix.block_model(Merica, 3, N_per_grp_Merica, name="Merica")
    SR = m.net2.sender_receiver(sender_preds, receiver_preds)
    D  = NeteffectMatrix.dyadic_effect(dyadic_preds_mat)
    m.dist.bernoulli(logits=m.net.mat_to_edgl(Ba + Bm + SR + D),
                     obs=network_edgl, name="network_edgl")


def model_shard(Y_mat, dyadic_preds_mat, sender_preds, Any_shard, Merica_shard,
                mask_mat_shard, eid_shard, dir_shard):
    Ba = NeteffectMatrix.block_model(data["Any"], 1, data["N_per_grp_Any"],
                                     name="intercept", group_row=Any_shard)
    Bm = NeteffectMatrix.block_model(data["Merica"], 3, data["N_per_grp_Merica"],
                                     name="Merica", group_row=Merica_shard)
    SR = m.net2.sender_receiver(sender_preds, data["receiver_preds"])
    D  = NeteffectMatrix.dyadic_effect(dyadic_preds_mat, eid=eid_shard, dir_idx=dir_shard)
    m.dist.bernoulli(logits=(Ba + Bm + SR + D) * mask_mat_shard, obs=Y_mat, name="Y_mat")


def dom_ref():
    return dict(network_edgl=data["network_edgl"], dyadic_preds_mat=data["dyadic_preds_mat"],
                sender_preds=data["sender_preds"], receiver_preds=data["receiver_preds"],
                Any=data["Any"], Merica=data["Merica"],
                N_per_grp_Any=data["N_per_grp_Any"], N_per_grp_Merica=data["N_per_grp_Merica"])


def dom_shard():
    return dict(Y_mat=data["Y_mat"], dyadic_preds_mat=data["dyadic_preds_mat"],
                sender_preds=data["sender_preds"], Any_shard=data["Any"],
                Merica_shard=data["Merica"], mask_mat_shard=mask,
                eid_shard=EID, dir_shard=DIR)


def structural_means(post):
    out = {}
    for k in STRUCTURAL:
        if k in post:
            v = np.asarray(post[k])
            out[k] = {"mean": v.reshape(v.shape[0], -1).mean(0).tolist(),
                      "sd":   v.reshape(v.shape[0], -1).std(0).tolist()}
    return out


def agreement(cur, ref):
    worst = 0.0
    breakdown = []
    for k, rv in ref.items():
        if k not in cur:
            continue
        cm = np.array(cur[k]["mean"]); rm = np.array(rv["mean"]); rs = np.array(rv["sd"])
        d = np.abs(cm - rm) / (rs + 1e-6)
        j = int(np.argmax(d))
        breakdown.append((k, float(d[j]), float(abs(cm.flat[j] - rm.flat[j])),
                          float(rs.flat[j]), float(cm.flat[j]), float(rm.flat[j])))
        worst = max(worst, float(np.max(d)))
    breakdown.sort(key=lambda t: -t[1])
    print("  per-param agreement (site: stdΔ, |Δ|, ref_sd, cur_mean, ref_mean):")
    for k, sd_, ad, rsd, cmn, rmn in breakdown:
        print(f"    {k:<18} std={sd_:7.2f}  |Δ|={ad:.4f}  ref_sd={rsd:.4f}  "
              f"cur={cmn:+.4f} ref={rmn:+.4f}")
    return worst


def ess_rhat(m):
    pc = m.posteriors_by_chain_full
    s = np_summary(pc)
    ess, rh = [], []
    for v in s.values():
        if "n_eff" in v: ess.extend(np.asarray(v["n_eff"]).flatten().tolist())
        if "r_hat" in v: rh.extend(np.asarray(v["r_hat"]).flatten().tolist())
    ess = np.array(ess); ess = ess[np.isfinite(ess)]
    rh = np.array(rh); rh = rh[np.isfinite(rh)]
    return (float(ess.min()) if len(ess) else 0.0,
            float(ess.mean()) if len(ess) else 0.0,
            float(rh.max()) if len(rh) else float("nan"))


# ---- choose model + run ---------------------------------------------------
# MODE=matrix → matrix formulation but NO sharding (isolates formulation vs shard)
use_matrix = MODE in ("shard", "shard_svinit", "matrix")
sharded = MODE in ("shard", "shard_svinit")
model = model_shard if use_matrix else model_ref
dom = dom_shard() if use_matrix else dom_ref()

init_strategy = None
svi_time = 0.0
if MODE in ("svinit", "shard_svinit"):
    # SVI warm-start (does NOT change the NUTS target → exact posterior).
    t0 = time.perf_counter()
    m.data_on_model = dom_ref()                 # SVI on the (replicated) model
    m.svi(model_ref, guide="diagonal", num_steps=SVI_STEPS, num_samples=200)
    svi_post = m.posteriors
    # Init only real-valued / simple-positive sites. Skip deterministics (_rf)
    # and Cholesky factors (_L): the mean of sampled cholesky factors need not be
    # a valid cholesky, which would break init_to_value's constraint transform.
    svi_means = {k: jnp.asarray(np.asarray(v).mean(0)) for k, v in svi_post.items()
                 if not k.endswith("_rf") and not k.endswith("_L")}
    init_strategy = init_to_value(values=svi_means)
    svi_time = time.perf_counter() - t0
    print(f"[{EXP_ID}] SVI warm-start done in {svi_time:.1f}s")

m.data_on_model = dom
fit_kw = dict(num_samples=SAMP, num_warmup=WARM, num_chains=CHAINS,
              shard=sharded, progress_bar=False, max_tree_depth=MAXTREE,
              target_accept_prob=TACC)
if init_strategy is not None:
    fit_kw["init_strategy"] = init_strategy

t0 = time.perf_counter()
m.fit(model, **fit_kw)
jax.block_until_ready([v for v in m.posteriors.values()])
fit_time = time.perf_counter() - t0
total_time = fit_time + svi_time

min_ess, mean_ess, max_rhat = ess_rhat(m)
try:
    div = int(np.asarray(m.sampler.get_extra_fields()["diverging"]).sum())
except Exception:
    div = -1

cur_means = structural_means(m.posteriors_full)
agree = float("nan")
if MODE == "ref":
    with open(REF_FILE, "w") as f:
        json.dump(cur_means, f)
    print(f"[{EXP_ID}] saved reference → {REF_FILE}")
elif os.path.exists(REF_FILE):
    with open(REF_FILE) as f:
        ref_means = json.load(f)
    agree = agreement(cur_means, ref_means)

s_it = total_time / (CHAINS and (WARM + SAMP)) if (WARM + SAMP) else float("nan")
row = dict(id=EXP_ID, mode=MODE, N=N, cores=CORES, warm=WARM, samp=SAMP,
           maxtree=str(MAXTREE), tacc=TACC, time_s=round(total_time, 1),
           s_per_it=round((WARM + SAMP) and total_time / (WARM + SAMP), 3),
           min_ess=round(min_ess, 1), mean_ess=round(mean_ess, 1),
           max_rhat=round(max_rhat, 3), div=div,
           agree=("" if np.isnan(agree) else round(agree, 3)))

import csv
csv_path = os.path.join(HERE, "opt_results.csv")
exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not exists: w.writeheader()
    w.writerow(row)

agree_str = "n/a" if row["agree"] == "" else f"{row['agree']} ({'MATCH' if row['agree']<0.25 else 'DIVERGE'})"
print("\n=== RESULT ===")
print(f"| {EXP_ID} | {MODE} | {N} | {WARM}+{SAMP} | {row['time_s']} | "
      f"{row['s_per_it']} | {row['min_ess']} | {row['mean_ess']} | {row['max_rhat']} | "
      f"{div} | {agree_str} |")
