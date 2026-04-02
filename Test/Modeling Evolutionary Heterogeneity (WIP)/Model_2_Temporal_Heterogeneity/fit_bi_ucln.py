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
N_init = len(left_children)
N_nodes = N_taxa + N_init

def get_hky_Q(kappa, pi):
    Q = jnp.zeros((4,4))
    Q = Q.at[0,2].set(kappa * pi[2])
    Q = Q.at[2,0].set(kappa * pi[0])
    Q = Q.at[1,3].set(kappa * pi[3])
    Q = Q.at[3,1].set(kappa * pi[1])
    Q = Q.at[0,1].set(pi[1])
    Q = Q.at[0,3].set(pi[3])
    Q = Q.at[1,0].set(pi[0])
    Q = Q.at[1,2].set(pi[2])
    Q = Q.at[2,1].set(pi[1])
    Q = Q.at[2,3].set(pi[3])
    Q = Q.at[3,0].set(pi[0])
    Q = Q.at[3,2].set(pi[2])
    diag = -jnp.sum(Q, axis=1)
    for i in range(4):
        Q = Q.at[i,i].set(diag[i])
    return Q

def discrete_gamma_rates(alpha, K=4):
    """
    Wilson-Hilferty transformation (Yang 1994) for high-fidelity Discrete Gamma 
    category-means as used in BEAST/MrBayes.
    """
    probs = jnp.linspace(0, 1, K + 1)[1:-1]
    z = jax.scipy.special.erfinv(2 * probs - 1) * jnp.sqrt(2)
    term1 = 1.0 - 1.0 / (9.0 * alpha)
    term2 = z / jnp.sqrt(9.0 * alpha)
    Q_internal = jnp.power(jnp.maximum(term1 + term2, 0.0), 3)
    cdf_internal = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_internal)
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_internal, jnp.array([1.0])])
    rates = K * (cdf_all[1:] - cdf_all[:-1])
    return jnp.maximum(rates, 1e-6)

m.data_on_model = {
    "left": left_children,
    "right": right_children,
    "bl": branch_lengths,
    "leaf_liks": leaf_likelihoods
}

def model(left, right, bl, leaf_liks):
    # Evolutionary parameters
    kappa = m.dist.half_normal(10.0, name="kappa")
    alpha = m.dist.half_normal(5.0, name="alpha")
    pi = jnp.array([0.3, 0.2, 0.1, 0.4])
    Q = get_hky_Q(kappa, pi)
    
    # Temporal Heterogeneity: UCLN relaxed clock
    mu_c = m.dist.normal(0.0, 1.0, name="mu_c")
    sigma_c = m.dist.half_normal(1.0, name="sigma_c")
    z_c = m.dist.normal(jnp.zeros(N_nodes), 1.0, name="z_c")
    branch_rates = jnp.exp(mu_c + sigma_c * z_c) 
    
    # Spatial Heterogeneity: +Gamma(4) discretization (Yang 1994)
    K = 4
    rates = discrete_gamma_rates(alpha, K)
    
    # Total branch distance = duration * rate
    # d_j = bl * branch_rates
    
    def calc_rate_lik(r):
        """Likelihood calculation for a given Gamma rate category."""
        # Adjusted distances = duration * branch_rate * gamma_site_rate
        Q_t = jnp.einsum('xy,n->nxy', Q, bl * branch_rates * r)
        P_matrices = jax.vmap(jax.scipy.linalg.expm)(Q_t)
        
        L_nodes = jnp.zeros((N_nodes, L, 4))
        L_nodes = L_nodes.at[:N_taxa].set(leaf_liks)
        
        def node_update(i, L_val):
            idx = i - N_taxa
            l = left[idx]
            r_c = right[idx]
            L_l = jnp.einsum('xy,sy->sx', P_matrices[l], L_val[l])
            L_r = jnp.einsum('xy,sy->sx', P_matrices[r_c], L_val[r_c])
            return L_val.at[i].set(L_l * L_r)
        
        final_L = jax.lax.fori_loop(N_taxa, N_nodes, node_update, L_nodes)
        root_L = final_L[N_nodes - 1]
        site_liks = jnp.dot(root_L, pi)
        return site_liks

    # Vectorize across Gamma site-rates
    site_liks_by_rate = jax.vmap(calc_rate_lik)(rates)
    mean_site_liks = jnp.mean(site_liks_by_rate, axis=0)
    
    log_likelihood = jnp.sum(jnp.log(jnp.maximum(mean_site_liks, 1e-30)))
    numpyro.factor("phylo_lik", log_likelihood)

print("Starting BI fit (UCLN + Gamma) on Real Primate Data...")
m.fit(model, num_samples=200, num_warmup=100)

post = m.posteriors
if post is not None:
    df = pd.DataFrame({
        'kappa': np.array(post['kappa']).flatten(),
        'alpha': np.array(post['alpha']).flatten(),
        'mu_c': np.array(post['mu_c']).flatten(),
        'sigma_c': np.array(post['sigma_c']).flatten()
    })
    df.to_csv("bi_ucln_post.csv", index=False)
    print("Posteriors saved to bi_ucln_post.csv")
