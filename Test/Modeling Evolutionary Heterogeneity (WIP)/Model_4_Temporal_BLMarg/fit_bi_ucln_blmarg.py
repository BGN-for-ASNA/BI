"""
Model 4 (v3): Temporal Heterogeneity (UCLN) — Substitution Aligned
====================================================================
Algorithmic Improvements (v3):
  1. Q-NORMALIZATION: Mean rate 1.0.
  2. SUBST-METRICS: SubstLength = sum(bl_base * branch_rate).
  3. CALIBRATED BEAST COMP: Reporting substitution tree.
"""

import numpyro
numpyro.set_host_device_count(4)
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from BI import bi

sys.path.append('..')
from tree_data import get_tree_data

m = bi(platform='cpu')

# --- Data ---
leaf_likelihoods = jnp.load("../primate_data.npy")
N_taxa, L, _ = leaf_likelihoods.shape

left_children, right_children, bl_init = get_tree_data()
N_internal = len(left_children)
N_nodes = N_taxa + N_internal

# --- HKY Rate Matrix (Q-Normalized) ---
_IS_TRANSITION = jnp.array([
    [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 0, 0, 0], [0, 1, 0, 0],
], dtype=jnp.float32)

def get_hky_Q(kappa, pi):
    Q_off = (1.0 + (kappa - 1.0) * _IS_TRANSITION) * (1.0 - jnp.eye(4)) * pi[None, :]
    Q = Q_off - jnp.diag(Q_off.sum(axis=1))
    mean_rate = -jnp.sum(pi * jnp.diag(Q))
    return Q / mean_rate

def discrete_gamma_rates(alpha, K=4):
    probs   = jnp.linspace(0, 1, K + 1)[1:-1]
    z       = jax.scipy.stats.norm.ppf(probs)
    a9      = 9.0 * alpha
    Q_int   = jnp.maximum(1.0 - 1.0 / a9 + z / jnp.sqrt(a9), 0.0) ** 3
    cdf_int = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_int)
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_int, jnp.array([1.0])])
    return jnp.maximum(K * jnp.diff(cdf_all), 1e-6)

# --- Model ---
m.data_on_model = {
    "left":      left_children,
    "right":     right_children,
    "bl_init":   bl_init,
    "leaf_liks": leaf_likelihoods,
}

def model(left, right, bl_init, leaf_liks):
    pi = m.dist.dirichlet(jnp.array([3.0, 2.0, 1.0, 4.0]), name="pi")

    kappa = m.dist.half_normal(10.0, name="kappa")
    alpha = m.dist.half_normal(5.0, name="alpha")
    Q     = get_hky_Q(kappa, pi)

    mu_c    = m.dist.normal(0.0, 1.0, name="mu_c")
    sigma_c = m.dist.half_normal(1.0, name="sigma_c")
    z_c     = m.dist.normal(jnp.zeros(N_nodes), 1.0, name="z_c")
    branch_rates = jnp.exp(mu_c + sigma_c * z_c)

    # Simplified BL scale
    bl_scale = m.dist.log_normal(0.0, 0.5, shape=(N_nodes,), name="bl_scale")
    bl_base = bl_init * bl_scale

    # Substitution-unit effective length
    effective_bl = bl_base * branch_rates

    K     = 4
    rates = discrete_gamma_rates(alpha, K)

    def calc_rate_lik(r):
        Q_t     = jnp.einsum('xy,n->nxy', Q, effective_bl * r)
        P_mat   = jax.vmap(jax.scipy.linalg.expm)(Q_t)
        L_nodes = jnp.zeros((N_nodes, L, 4))
        L_nodes = L_nodes.at[:N_taxa].set(leaf_liks)

        def node_update(i, L_val):
            idx = i - N_taxa
            l_c, r_c = left[idx], right[idx]
            L_l = jnp.einsum('xy,sy->sx', P_mat[l_c], L_val[l_c])
            L_r = jnp.einsum('xy,sy->sx', P_mat[r_c], L_val[r_c])
            return L_val.at[i].set(L_l * L_r)

        final_L = jax.lax.fori_loop(N_taxa, N_nodes, node_update, L_nodes)
        return jnp.dot(final_L[N_nodes - 1], pi)

    site_liks_by_rate = jax.vmap(calc_rate_lik)(rates)
    log_lik = jnp.sum(jnp.log(jnp.maximum(jnp.mean(site_liks_by_rate, axis=0), 1e-30)))
    numpyro.factor("phylo_lik", log_lik)


print("Starting BI fit (Model 4 v3 — UCLN Alignment Full Data)...")
m.fit(model, num_samples=300, num_warmup=500, num_chains=4)

post = m.posteriors
if post is not None:
    bl_base_samples = np.array(bl_init)[None, :] * np.array(post['bl_scale'])
    z_samples        = np.array(post['z_c'])
    mu_samples       = np.array(post['mu_c']).flatten()
    sg_samples       = np.array(post['sigma_c']).flatten()
    rates            = np.exp(mu_samples[:, None] + sg_samples[:, None] * z_samples)

    effective_bl = bl_base_samples * rates
    
    total_subst = np.sum(effective_bl, axis=1)

    def get_height(bl):
        dist = {22: 0.0}
        for i in range(11 - 1, -1, -1):
            p = 12 + i
            l_idx = int(left_children[i])
            r_idx = int(right_children[i])
            dist[l_idx] = dist[p] + float(bl[l_idx])
            dist[r_idx] = dist[p] + float(bl[r_idx])
        return max([dist[t] for t in range(12)])

    heights = np.array([get_height(s) for s in effective_bl])

    df = pd.DataFrame({
        'kappa':       np.array(post['kappa']).flatten(),
        'alpha':       np.array(post['alpha']).flatten(),
        'mu_c':        mu_samples,
        'sigma_c':     sg_samples,
        'SubstHeight': heights,
        'SubstLength': total_subst,
        # Compatibility
        'TreeHeight':  heights,
        'TotalLength': total_subst
    })
    df.to_csv("bi_ucln_blmarg_post.csv", index=False)
    print("Posteriors saved to bi_ucln_blmarg_post.csv")
    for col in ['kappa', 'alpha', 'mu_c', 'sigma_c', 'SubstHeight', 'SubstLength']:
        print(f"  {col}: {df[col].mean():.3f} (±{df[col].std():.3f})")
