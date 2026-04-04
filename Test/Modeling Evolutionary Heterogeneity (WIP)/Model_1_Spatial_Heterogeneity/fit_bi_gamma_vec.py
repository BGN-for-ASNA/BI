import os
import sys
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pandas as pd
from BI import bi

# Add parent directory to path for tree_data.py
sys.path.append('..')
from tree_data import get_tree_data

m = bi(platform='cpu')

# Load Real Data
leaf_likelihoods = jnp.load("../primate_data.npy")[:, :100, :]
N_taxa, L, _ = leaf_likelihoods.shape

# Load Tree
left_children, right_children, branch_lengths = get_tree_data()
N_internal = len(left_children)
N_nodes = N_taxa + N_internal

# Vectorized HKY rate matrix
_IS_TRANSITION = jnp.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=jnp.float32)

def get_hky_Q(kappa, pi):
    Q_off = (1.0 + (kappa - 1.0) * _IS_TRANSITION) * (1.0 - jnp.eye(4)) * pi[None, :]
    return Q_off - jnp.diag(Q_off.sum(axis=1))

def discrete_gamma_rates(alpha, K=4):
    probs   = jnp.linspace(0, 1, K + 1)[1:-1]
    z       = jax.scipy.stats.norm.ppf(probs)
    a9      = 9.0 * alpha
    Q_int   = jnp.maximum(1.0 - 1.0 / a9 + z / jnp.sqrt(a9), 0.0) ** 3
    cdf_int = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_int)
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_int, jnp.array([1.0])])
    return jnp.maximum(K * jnp.diff(cdf_all), 1e-6)

m.data_on_model = {
    "left": left_children,
    "right": right_children,
    "bl": branch_lengths,
    "leaf_liks": leaf_likelihoods
}

def model(left, right, bl, leaf_liks):
    kappa = m.dist.half_normal(10.0, name="kappa")
    alpha = m.dist.half_normal(5.0, name="alpha")
    pi = jnp.array([0.3, 0.2, 0.1, 0.4])
    Q = get_hky_Q(kappa, pi)
    K = 4
    rates = discrete_gamma_rates(alpha, K)
    
    def calc_rate_lik(r):
        Q_t = jnp.einsum('xy,n->nxy', Q, bl * r)
        P_matrices = jax.vmap(jax.scipy.linalg.expm)(Q_t)
        L_nodes = jnp.zeros((N_nodes, L, 4))
        L_nodes = L_nodes.at[:N_taxa].set(leaf_liks)
        def node_update(i, L_val):
            idx = i - N_taxa
            l, r_c = left[idx], right[idx]
            L_l = jnp.einsum('xy,sy->sx', P_matrices[l], L_val[l])
            L_r = jnp.einsum('xy,sy->sx', P_matrices[r_c], L_val[r_c])
            return L_val.at[i].set(L_l * L_r)
        final_L = jax.lax.fori_loop(N_taxa, N_nodes, node_update, L_nodes)
        return jnp.dot(final_L[N_nodes - 1], pi)

    site_liks_by_rate = jax.vmap(calc_rate_lik)(rates)
    mean_site_liks = jnp.mean(site_liks_by_rate, axis=0)
    log_likelihood = jnp.sum(jnp.log(jnp.maximum(mean_site_liks, 1e-30)))
    numpyro.factor("phylo_lik", log_likelihood)

print("Starting BI fit (Vectorized Spatial Heterogeneity) ...")
m.fit(model, num_samples=200, num_warmup=100) # Fast run for comparison

post = m.posteriors
if post is not None:
    df = pd.DataFrame({
        'kappa': np.array(post['kappa']).flatten(),
        'alpha': np.array(post['alpha']).flatten()
    })
    df.to_csv("bi_gamma_vec_post.csv", index=False)
    print("Posteriors saved to bi_gamma_vec_post.csv")
