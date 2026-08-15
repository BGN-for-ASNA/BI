"""Small-N diagnostic for the matrix-SRM sharded path.

Runs no-shard vs shard at small N with few iterations to (a) surface what the
shard-check reports, (b) confirm correctness, (c) measure the per-iteration
speed ratio — without paying the full 2.67 h benchmark.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix

N            = int(os.environ.get("DIAG_N", "40"))
CORES        = int(os.environ.get("DIAG_CORES", "4"))
WARM         = int(os.environ.get("DIAG_WARM", "60"))
SAMP         = int(os.environ.get("DIAG_SAMP", "60"))
CHAINS       = int(os.environ.get("DIAG_CHAINS", "4"))
SEED         = 42
N_GRP_ANY, N_GRP_MERICA = 1, 3
assert N % CORES == 0, f"N={N} not divisible by cores={CORES}"

m = bf(platform="cpu", cores=CORES, rand_seed=False, print_devices_found=True)
print(f"n_devices={m.n_devices}  N={N}  chains={CHAINS}  warm={WARM} samp={SAMP}\n")


def simulate_data():
    np.random.seed(SEED)
    mt = bf("cpu", rand_seed=False, print_devices_found=False)
    wide_focal  = np.random.normal(0, 1, size=(N, 4))
    wide_target = np.random.normal(0, 1, size=(N, 4))
    dyadic = np.random.binomial(1, 0.3, size=(N, N, 3))
    for k in range(3):
        np.fill_diagonal(dyadic[:, :, k], 0)
    dyadic[:, :, 2] = dyadic[:, :, 0] * dyadic[:, :, 1]
    dyadic_edgl = jnp.stack([mt.net.mat_to_edgl(dyadic[:, :, k]) for k in range(3)], axis=2)
    Any_np    = np.zeros(N, dtype=int)
    Merica_np = np.random.choice([0, 1, 2], size=(N,), p=[0.25, 0.5625, 0.1875])
    Nby_any    = np.array([N])
    Nby_merica = np.array([np.sum(Merica_np == i) for i in range(N_GRP_MERICA)])
    B_int = mt.net.block_model(Any_np, N_GRP_ANY, jnp.array(Nby_any), sample=True, name="intercept")
    B_cat = mt.net.block_model(Merica_np, N_GRP_MERICA, jnp.array(Nby_merica), sample=True, name="category")
    sr = mt.net.sender_receiver(jnp.array(wide_focal), jnp.array(wide_target), s_mu=0.4, r_mu=-0.4, sample=True)
    dr = mt.net.dyadic_effect(dyadic_edgl, d_sd=2.5, sample=True)
    net_edgl = jnp.array(mt.dist.bernoulli(logits=B_int + B_cat + sr + dr, sample=True))
    Any    = jnp.array(Any_np, dtype=jnp.int32)
    Merica = jnp.array(Merica_np, dtype=jnp.int32)
    _, Nany = jnp.unique(Any, return_counts=True)
    _, Nmer = jnp.unique(Merica, return_counts=True)
    return dict(
        network_edgl=net_edgl,
        Y_mat=NeteffectMatrix.edgelist_to_matrix_outcome(net_edgl, N),
        dyadic_preds_mat=NeteffectMatrix.edgelist_to_matrix_predictors(dyadic_edgl, N),
        sender_preds=jnp.array(wide_focal), receiver_preds=jnp.array(wide_target),
        Any=Any, Merica=Merica, N_per_grp_Any=Nany, N_per_grp_Merica=Nmer,
        N_dyads=int(net_edgl.shape[0]))


data = simulate_data()
mask_mat = (1 - jnp.eye(N)).astype(jnp.float32)
EID, DIR = NeteffectMatrix.dyad_index_maps(N)   # (N,N) row-shardable gather maps


def model_no_shard(network_edgl, dyadic_preds_mat, sender_preds, receiver_preds,
                   Any, Merica, N_per_grp_Any, N_per_grp_Merica, **_):
    B_any    = NeteffectMatrix.block_model(Any, N_GRP_ANY, N_per_grp_Any, name="intercept")
    B_Merica = NeteffectMatrix.block_model(Merica, N_GRP_MERICA, N_per_grp_Merica, name="Merica")
    SR  = m.net2.sender_receiver(sender_preds, receiver_preds)
    D   = NeteffectMatrix.dyadic_effect(dyadic_preds_mat)
    m.dist.bernoulli(logits=m.net.mat_to_edgl(B_any + B_Merica + SR + D),
                     obs=network_edgl, name="network_edgl")


def model_shard(Y_mat, dyadic_preds_mat, sender_preds, Any_shard, Merica_shard, mask_mat_shard):
    B_any    = NeteffectMatrix.block_model(data["Any"], N_GRP_ANY, data["N_per_grp_Any"],
                                           name="intercept", group_row=Any_shard)
    B_Merica = NeteffectMatrix.block_model(data["Merica"], N_GRP_MERICA, data["N_per_grp_Merica"],
                                           name="Merica", group_row=Merica_shard)
    SR  = m.net2.sender_receiver(sender_preds, data["receiver_preds"])
    D   = NeteffectMatrix.dyadic_effect(dyadic_preds_mat)
    logits = (B_any + B_Merica + SR + D) * mask_mat_shard
    m.dist.bernoulli(logits=logits, obs=Y_mat, name="Y_mat")


def model_shard_gather(Y_mat, dyadic_preds_mat, sender_preds, Any_shard,
                       Merica_shard, mask_mat_shard, eid_shard, dir_shard):
    B_any    = NeteffectMatrix.block_model(data["Any"], N_GRP_ANY, data["N_per_grp_Any"],
                                           name="intercept", group_row=Any_shard)
    B_Merica = NeteffectMatrix.block_model(data["Merica"], N_GRP_MERICA, data["N_per_grp_Merica"],
                                           name="Merica", group_row=Merica_shard)
    SR  = m.net2.sender_receiver(sender_preds, data["receiver_preds"])
    D   = NeteffectMatrix.dyadic_effect(dyadic_preds_mat, eid=eid_shard, dir_idx=dir_shard)
    logits = (B_any + B_Merica + SR + D) * mask_mat_shard
    m.dist.bernoulli(logits=logits, obs=Y_mat, name="Y_mat")


def run(label, model, dom, shard):
    m.data_on_model = dom
    t0 = time.perf_counter()
    m.fit(model, num_samples=SAMP, num_warmup=WARM, num_chains=CHAINS,
          shard=shard, progress_bar=False)
    jax.block_until_ready([v for v in m.posteriors.values()])
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.1f}s")
    return dt


if __name__ == "__main__":
    print(">>> no-shard"); sys.stdout.flush()
    t_ns = run("no-shard", model_no_shard, dict(
        network_edgl=data["network_edgl"], dyadic_preds_mat=data["dyadic_preds_mat"],
        sender_preds=data["sender_preds"], receiver_preds=data["receiver_preds"],
        Any=data["Any"], Merica=data["Merica"],
        N_per_grp_Any=data["N_per_grp_Any"], N_per_grp_Merica=data["N_per_grp_Merica"]),
        shard=False)

    print("\n>>> shard (scatter)"); sys.stdout.flush()
    t_s = run("shard-scatter", model_shard, dict(
        Y_mat=data["Y_mat"], dyadic_preds_mat=data["dyadic_preds_mat"],
        sender_preds=data["sender_preds"], Any_shard=data["Any"],
        Merica_shard=data["Merica"], mask_mat_shard=mask_mat), shard=True)

    print("\n>>> shard (gather)"); sys.stdout.flush()
    t_g = run("shard-gather", model_shard_gather, dict(
        Y_mat=data["Y_mat"], dyadic_preds_mat=data["dyadic_preds_mat"],
        sender_preds=data["sender_preds"], Any_shard=data["Any"],
        Merica_shard=data["Merica"], mask_mat_shard=mask_mat,
        eid_shard=EID, dir_shard=DIR), shard=True)

    print(f"\nspeedup scatter (no-shard / shard): {t_ns / t_s:.2f}×")
    print(f"speedup gather  (no-shard / shard): {t_ns / t_g:.2f}×")
    print(f"gather vs scatter: {t_s / t_g:.2f}×")
