"""
BF CPU benchmark — matrix-form SRM, N=400 nodes.

Tests both sharding modes:
  - BF Matrix (no shard, chain×4, chain_method='parallel')
  - BF Matrix (shard,    chain×4, cores=20, chain_method='vectorized')

N=400 → N_dyads=79,800.
cores=20 chosen because 400 % 20 == 0 (exact divisibility required for axis-0 sharding).
The no-shard run also uses cores=20 so a single bf instance serves both configs;
chain_method='parallel' only needs num_chains (4) devices, extras are idle.

Output: benchmark_bf_matrix.csv (same columns as benchmark_results.csv)
"""

import numpy as np
import time
import os
import sys
import pandas as pd
import jax
import jax.numpy as jnp
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from numpyro.diagnostics import summary as numpyro_summary

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N             = 400
CORES         = 16   # 400 % 4 == 0 (4 devices/chain); 100 rows/device
NUM_CHAINS    = 4
ITER_SAMPLING = 1000
ITER_WARMUP   = 1000
SEED          = 42
N_GRP_ANY     = 1
N_GRP_MERICA  = 3

m = bf(platform='cpu', cores=CORES, rand_seed=False, print_devices_found=True, shard=True)
print(f"\nn_devices={m.n_devices}  chains={NUM_CHAINS}\n")

# ---------------------------------------------------------------------------
# Data simulation — same seed / logic as benchmark_bf_sharding.py
# ---------------------------------------------------------------------------
def simulate_data():
    np.random.seed(SEED)
    N_focal_vars = 4
    N_target_vars = 4
    N_dyad_vars = 3

    m_tmp = bf('cpu', rand_seed=False, print_devices_found=False)

    wide_focal_np  = np.random.normal(0, 1, size=(N, N_focal_vars))
    wide_target_np = np.random.normal(0, 1, size=(N, N_target_vars))

    dyadic_mat_np = np.random.binomial(1, 0.3, size=(N, N, N_dyad_vars))
    for k in range(N_dyad_vars):
        np.fill_diagonal(dyadic_mat_np[:, :, k], 0)
    dyadic_mat_np[:, :, 2] = dyadic_mat_np[:, :, 0] * dyadic_mat_np[:, :, 1]

    dyadic_edgl = jnp.stack(
        [m_tmp.net.mat_to_edgl(dyadic_mat_np[:, :, k]) for k in range(N_dyad_vars)],
        axis=2,
    )

    Any_np    = np.zeros(N, dtype=int)
    Merica_np = np.random.choice([0, 1, 2], size=(N,), p=[0.25, 0.5625, 0.1875])
    N_by_grp_Any    = np.array([N])
    N_by_grp_Merica = np.array([np.sum(Merica_np == i) for i in range(N_GRP_MERICA)])

    B_int = m_tmp.net.block_model(Any_np,    N_GRP_ANY,    jnp.array(N_by_grp_Any),    sample=True, name='intercept')
    B_cat = m_tmp.net.block_model(Merica_np, N_GRP_MERICA, jnp.array(N_by_grp_Merica), sample=True, name='category')
    sr = m_tmp.net.sender_receiver(
        jnp.array(wide_focal_np), jnp.array(wide_target_np), s_mu=0.4, r_mu=-0.4, sample=True)
    dr = m_tmp.net.dyadic_effect(dyadic_edgl, d_sd=2.5, sample=True)
    network_edgl = jnp.array(m_tmp.dist.bernoulli(logits=B_int + B_cat + sr + dr, sample=True))

    Any    = jnp.array(Any_np,    dtype=jnp.int32)
    Merica = jnp.array(Merica_np, dtype=jnp.int32)
    _, N_per_grp_Any    = jnp.unique(Any,    return_counts=True)
    _, N_per_grp_Merica = jnp.unique(Merica, return_counts=True)

    Y_mat            = NeteffectMatrix.edgelist_to_matrix_outcome(network_edgl, N)
    dyadic_preds_mat = NeteffectMatrix.edgelist_to_matrix_predictors(dyadic_edgl, N)

    return dict(
        network_edgl     = network_edgl,
        Y_mat            = Y_mat,
        dyadic_preds_mat = dyadic_preds_mat,
        sender_preds     = jnp.array(wide_focal_np),
        receiver_preds   = jnp.array(wide_target_np),
        Any              = Any,
        Merica           = Merica,
        N_per_grp_Any    = N_per_grp_Any,
        N_per_grp_Merica = N_per_grp_Merica,
        N_dyads          = int(network_edgl.shape[0]),
    )

print(f"Simulating data for N={N}, seed={SEED} ...")
data     = simulate_data()
mask_mat = (1 - jnp.eye(N)).astype(jnp.float32)
N_dyads  = data['N_dyads']
print(f"  N_dyads={N_dyads},  N_obs={N_dyads*2}\n")

# ---------------------------------------------------------------------------
# Model definitions — close over outer m and data globals
# ---------------------------------------------------------------------------
def model_srm_matrix_no_shard(
    network_edgl, dyadic_preds_mat, sender_preds, receiver_preds,
    Any, Merica, N_per_grp_Any, N_per_grp_Merica, **_
):
    B_any    = NeteffectMatrix.block_model(Any,    N_GRP_ANY,    N_per_grp_Any,    name='intercept')
    B_Merica = NeteffectMatrix.block_model(Merica, N_GRP_MERICA, N_per_grp_Merica, name='Merica')
    SR_mat   = m.net2.sender_receiver(sender_preds, receiver_preds)
    D_mat    = NeteffectMatrix.dyadic_effect(dyadic_preds_mat)
    logits_edgl = m.net.mat_to_edgl(B_any + B_Merica + SR_mat + D_mat)
    m.dist.bernoulli(logits=logits_edgl, obs=network_edgl, name='network_edgl')


def model_srm_matrix_sharded(Y_mat, dyadic_preds_mat, sender_preds,
                             Any_shard, Merica_shard, mask_mat_shard):
    """Sharded row-by-row: shard-local arrays arrive as (N/k, *).

    data_on_model contains Y_mat, dyadic_preds_mat, sender_preds,
    Any_shard, Merica_shard, mask_mat_shard — all sharded along axis 0.
    receiver_preds, full Any/Merica, N_per_grp_* from closure (replicated).

    block_model samples b (G,G) replicated, then gathers:
        b[Any_shard[:, None], Any_all[None, :]]  →  (N/k, N)
    so the output matches the sharded Y_mat rows without shape errors.
    """
    B_any    = NeteffectMatrix.block_model(
        data['Any'], N_GRP_ANY, data['N_per_grp_Any'],
        name='intercept', group_row=Any_shard)           # (N/k, N)
    B_Merica = NeteffectMatrix.block_model(
        data['Merica'], N_GRP_MERICA, data['N_per_grp_Merica'],
        name='Merica', group_row=Merica_shard)            # (N/k, N)
    SR_mat   = m.net2.sender_receiver(sender_preds, data['receiver_preds'])  # (N/k, N)
    D_mat    = NeteffectMatrix.dyadic_effect(dyadic_preds_mat)                # (N/k, N)
    logits   = (B_any + B_Merica + SR_mat + D_mat) * mask_mat_shard          # (N/k, N)
    m.dist.bernoulli(logits=logits, obs=Y_mat, name='Y_mat')

# ---------------------------------------------------------------------------
# ESS helper
# ---------------------------------------------------------------------------
def compute_ess(posteriors):
    try:
        pc   = {k: np.expand_dims(v, 0) for k, v in posteriors.items()}
        s    = numpyro_summary(pc)
        effs = []
        for v in s.values():
            if 'n_eff' in v:
                effs.extend(np.array(v['n_eff']).flatten().tolist())
        effs = np.array(effs)
        effs = effs[np.isfinite(effs)]
        return (float(np.min(effs)), float(np.mean(effs))) if len(effs) else (0.0, 0.0)
    except Exception:
        return 0.0, 0.0

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_no_shard():
    m.data_on_model = dict(
        network_edgl     = data['network_edgl'],
        dyadic_preds_mat = data['dyadic_preds_mat'],
        sender_preds     = data['sender_preds'],
        receiver_preds   = data['receiver_preds'],
        Any              = data['Any'],
        Merica           = data['Merica'],
        N_per_grp_Any    = data['N_per_grp_Any'],
        N_per_grp_Merica = data['N_per_grp_Merica'],
    )
    t0 = time.perf_counter()
    m.fit(model_srm_matrix_no_shard,
          num_samples=ITER_SAMPLING, num_warmup=ITER_WARMUP,
          num_chains=NUM_CHAINS, shard=False, progress_bar=True)
    jax.block_until_ready([v for v in m.posteriors.values()])
    return time.perf_counter() - t0


def run_sharded():
    m.data_on_model = dict(
        Y_mat            = data['Y_mat'],
        dyadic_preds_mat = data['dyadic_preds_mat'],
        sender_preds     = data['sender_preds'],
        Any_shard        = data['Any'],        # sharded to (N/k,) per device
        Merica_shard     = data['Merica'],     # sharded to (N/k,) per device
        mask_mat_shard   = mask_mat,           # sharded to (N/k, N) per device
    )
    t0 = time.perf_counter()
    m.fit(model_srm_matrix_sharded,
          num_samples=ITER_SAMPLING, num_warmup=ITER_WARMUP,
          num_chains=NUM_CHAINS, shard=True, progress_bar=True)
    jax.block_until_ready([v for v in m.posteriors.values()])
    return time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'benchmark_bf_matrix_c16.csv')

    hdr = f"{'Model':<40} | {'Time (s)':<10} | {'Min ESS':<10} | {'Mean ESS':<10}"
    sep = '-' * len(hdr)
    print(sep); print(hdr); print(sep); sys.stdout.flush()

    # Load previous no-shard result if it exists
    results = []
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        no_shard_prev = prev[prev['Model'].str.contains('no shard')]
        for _, row in no_shard_prev.iterrows():
            r = row.to_dict()
            r['Sharded'] = False  # fix column
            results.append(r)
            print(f"{r['Model']:<40} | {r['Time_s']:<10.2f} | {r['Min_ESS']:<10.2f} | {r['Mean_ESS']:<10.2f}  [loaded from previous run]")
        sys.stdout.flush()

    # Run only sharded
    label = f'BF Matrix (shard, chain×4, cores={CORES})'
    print(f"\n>>> {label}")
    sys.stdout.flush()
    elapsed = run_sharded()
    min_ess, mean_ess = compute_ess(m.posteriors)

    results.append({
        'Model':         label,
        'N_nodes':       N,
        'N_dyads':       N_dyads,
        'Time_s':        elapsed,
        'Min_ESS':       min_ess,
        'Mean_ESS':      mean_ess,
        'Iter_Warmup':   ITER_WARMUP,
        'Iter_Sampling': ITER_SAMPLING,
        'Num_chains':    NUM_CHAINS,
        'Cores':         CORES,
        'Sharded':       True,
    })

    print(f"{label:<40} | {elapsed:<10.2f} | {min_ess:<10.2f} | {mean_ess:<10.2f}")
    sys.stdout.flush()

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print(sep)
    no_shard_rows = df[df['Sharded'] == False]
    shard_rows    = df[df['Sharded'] == True]
    if len(no_shard_rows) and len(shard_rows):
        ratio = no_shard_rows.iloc[0]['Time_s'] / shard_rows.iloc[0]['Time_s']
        print(f"\nSpeedup (no-shard / shard): {ratio:.2f}×")
    print(f"Results saved to {csv_path}")
