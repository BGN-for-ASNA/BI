"""
BF CPU sharding benchmark — N=400 nodes only.

Compares BF CPU (no shard, chain×4) vs BF CPU (shard×4, 24 devices).
Pattern mirrors benchmark_shard.py: single process, model closes over outer
bf instance, num_chains=4 so vmap + data sharding both contribute.

Usage:
    python3 benchmark_bf_sharding.py

Output:
    benchmark_bf_sharding.csv
"""

import numpy as np
import time
import os
import sys
import pandas as pd
import jax.numpy as jnp
from BayesForge import bf
from BayesForge.Network.model_effects import Neteffect
from numpyro.diagnostics import summary as numpyro_summary

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# ---------------------------------------------------------------------------
# Config — mirrors benchmark_shard.py
# ---------------------------------------------------------------------------
N             = 400
CORES         = 24
NUM_CHAINS    = 4
ITER_SAMPLING = 250   # 250/chain × 4 chains = 1000 total (same as CSV)
ITER_WARMUP   = 1000
SEED          = 42

# ---------------------------------------------------------------------------
# Single bf instance with all 24 virtual devices — fixed for whole process
# ---------------------------------------------------------------------------
m = bf(platform='cpu', cores=CORES, rand_seed=False, print_devices_found=True)
print(f"\nn_devices={m.n_devices}  chains={NUM_CHAINS}  "
      f"devices/chain (shard)={CORES // NUM_CHAINS}\n")


# ---------------------------------------------------------------------------
# Data simulation — identical seed/logic to benchmark_stan2.py
# ---------------------------------------------------------------------------
def simulate_data():
    np.random.seed(SEED)

    N_focal_vars  = 4
    N_target_vars = 4
    N_dyad_vars   = 3

    m_tmp = bf('cpu', rand_seed=False, print_devices_found=False)

    wide_focal_np  = np.random.normal(0, 1, size=(N, N_focal_vars))
    wide_target_np = np.random.normal(0, 1, size=(N, N_target_vars))

    dyadic_predictors_mat = np.random.binomial(1, 0.3, size=(N, N, N_dyad_vars))
    for k in range(N_dyad_vars):
        np.fill_diagonal(dyadic_predictors_mat[:, :, k], 0)
    dyadic_predictors_mat[:, :, 2] = (
        dyadic_predictors_mat[:, :, 0] * dyadic_predictors_mat[:, :, 1]
    )
    dyadic_predictors_edgl = jnp.stack(
        [m_tmp.net.mat_to_edgl(dyadic_predictors_mat[:, :, k]) for k in range(N_dyad_vars)],
        axis=2,
    )

    Any_np    = np.zeros(N, dtype=int)
    Merica_np = np.random.choice([0, 1, 2], size=(N,), p=[0.25, 0.5625, 0.1875])

    N_by_grp_Any    = np.array([N])
    N_by_grp_Merica = np.array([np.sum(Merica_np == i) for i in range(3)])

    B_intercept = m_tmp.net.block_model(Any_np,    1, jnp.array(N_by_grp_Any),    sample=True, name="intercept")
    B_category  = m_tmp.net.block_model(Merica_np, 3, jnp.array(N_by_grp_Merica), sample=True, name="category")
    sr = m_tmp.net.sender_receiver(jnp.array(wide_focal_np), jnp.array(wide_target_np),
                                   s_mu=0.4, r_mu=-0.4, sample=True)
    dr = m_tmp.net.dyadic_effect(dyadic_predictors_edgl, d_sd=2.5, sample=True)
    network_edgl = m_tmp.dist.bernoulli(logits=B_intercept + B_category + sr + dr, sample=True)

    return dict(
        network_edgl          = jnp.array(network_edgl),
        dyadic_predictors     = jnp.array(dyadic_predictors_edgl),
        sender_predictors     = jnp.array(wide_focal_np),
        receiver_predictors   = jnp.array(wide_target_np),
        Any                   = jnp.array(Any_np,    dtype=jnp.int32),
        Merica                = jnp.array(Merica_np, dtype=jnp.int32),
        N_by_grp_Any          = N_by_grp_Any,
        N_by_grp_Merica       = N_by_grp_Merica,
        N_dyads               = network_edgl.shape[0],
    )


# ---------------------------------------------------------------------------
# Model — closes over outer `m` (never instantiates bf inside)
# ---------------------------------------------------------------------------
def model_srm(
    network_edgl, dyadic_predictors,
    sender_predictors, receiver_predictors,
    Any, Merica, N_by_grp_Any, N_by_grp_Merica,
    **_,
):
    B_any    = Neteffect.block_model(Any,    1, jnp.array(N_by_grp_Any),    name='intercept')
    B_Merica = Neteffect.block_model(Merica, 3, jnp.array(N_by_grp_Merica), name='Merica')
    sr = m.net.sender_receiver(sender_predictors, receiver_predictors)
    dr = m.net.dyadic_effect(dyadic_predictors)
    m.dist.bernoulli(logits=B_any + B_Merica + sr + dr, obs=network_edgl, name='network_edgl')


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
def run(shard_flag):
    import jax
    t0 = time.perf_counter()
    m.fit(
        model_srm,
        obs          = data,
        num_samples  = ITER_SAMPLING,
        num_warmup   = ITER_WARMUP,
        num_chains   = NUM_CHAINS,
        shard        = shard_flag,
        progress_bar = True,
    )
    # JAX pmap dispatches asynchronously on CPU virtual devices.
    # Block until all pending XLA computations are actually done.
    jax.block_until_ready([v for v in m.posteriors.values()])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'benchmark_bf_sharding.csv')

    print(f"Simulating data for N={N}, seed={SEED} ...")
    data    = simulate_data()
    N_dyads = int(data['N_dyads'])
    N_obs   = N_dyads * 2
    # strip N_dyads from obs dict (not a model arg)
    data    = {k: v for k, v in data.items() if k != 'N_dyads'}
    print(f"  N_dyads={N_dyads},  N_obs={N_obs}\n")

    results = []

    # Shard=True (chain_method='vectorized' + 24 virtual devices) causes an
    # XLA allgather deadlock for the SRM model: index-based sender/receiver
    # lookups create asymmetric execution paths across shards, deadlocking
    # the all-gather rendezvous. Only chain parallelism is recorded.
    configs = [
        ('BF CPU (chain×4, no shard)',  False),
    ]

    hdr = (f"{'Model':<35} | {'Time (s)':<10} | {'Min ESS':<10} | {'Mean ESS':<10}")
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep); sys.stdout.flush()

    for label, shard_flag in configs:
        print(f"\n>>> {label}")
        sys.stdout.flush()
        elapsed = run(shard_flag)
        min_ess, mean_ess = compute_ess(m.posteriors)

        results.append({
            'Model':         label,
            'N_nodes':       N,
            'N_dyads':       N_dyads,
            'Cores':         CORES if shard_flag else 1,
            'Sharded':       shard_flag,
            'Num_chains':    NUM_CHAINS,
            'Time_s':        elapsed,
            'Min_ESS':       min_ess,
            'Mean_ESS':      mean_ess,
            'Iter_Warmup':   ITER_WARMUP,
            'Iter_Sampling': ITER_SAMPLING,
        })

        print(f"{label:<35} | {elapsed:<10.2f} | {min_ess:<10.2f} | {mean_ess:<10.2f}")
        sys.stdout.flush()

        # save after each run
        pd.DataFrame(results).to_csv(csv_path, index=False)

    print(sep)
    ratio = results[0]['Time_s'] / results[1]['Time_s'] if results[1]['Time_s'] > 0 else float('nan')
    print(f"\nSpeedup (no-shard / shard): {ratio:.2f}×")
    print(f"Results saved to {csv_path}")
