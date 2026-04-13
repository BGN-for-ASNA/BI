import os
import numpyro
numpyro.set_host_device_count(4)
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from BI import bi

# Add parent directory to path for tree_data.py
sys.path.append('..')
from tree_data import get_tree_data

m = bi(platform='cpu')

# Load Real Data
# leaf_likelihoods shape: (N_taxa, L, 4)
leaf_likelihoods = jnp.load("../primate_data.npy")
N_taxa, L, _ = leaf_likelihoods.shape

# Load Tree
left_children, right_children, branch_lengths = get_tree_data()
N_internal = len(left_children)
N_nodes = N_taxa + N_internal

# HKY transition matrix generator
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
    # Normalize Q so that mean rate is 1.0
    mean_rate = -jnp.dot(pi, diag)
    return Q / mean_rate

def discrete_gamma_rates(alpha, K=4):
    # Probabilities for internal quantiles (e.g., 0.25, 0.5, 0.75 for K=4)
    probs = jnp.linspace(0, 1, K + 1)[1:-1]
    
    # Approximate internal Gamma quantiles using Wilson-Hilferty
    z = jax.scipy.special.erfinv(2 * probs - 1) * jnp.sqrt(2)
    term1 = 1.0 - 1.0 / (9.0 * alpha)
    term2 = z / jnp.sqrt(9.0 * alpha)
    Q_internal = jnp.power(jnp.maximum(term1 + term2, 0.0), 3)
    
    # Normalized incomplete gamma P(a, x) = gamma(a, x)/Gamma(a)
    # Mean of category k defined by quantiles [Q_{k-1}, Q_k] is:
    # R_k = K * [P(alpha+1, alpha*Q_k) - P(alpha+1, alpha*Q_{k-1})]
    cdf_internal = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_internal)
    
    # Combine with boundary values: P(shape, 0) = 0 and P(shape, inf) = 1
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_internal, jnp.array([1.0])])
    rates = K * (cdf_all[1:] - cdf_all[:-1])
    
    # Ensure numerical stability
    return jnp.maximum(rates, 1e-6)

m.data_on_model = {
    "left": left_children,
    "right": right_children,
    "bl": branch_lengths,
    "leaf_liks": leaf_likelihoods
}

def model(left, right, bl, leaf_liks):
    kappa = m.dist.half_normal(10.0, name="kappa")
    alpha = m.dist.half_normal(5.0, name="alpha")
    
    # State frequencies - empirical from data or informative prior
    pi = jnp.array([0.3, 0.2, 0.1, 0.4]) # rough estimates for mtDNA
    
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
            l = left[idx]
            r_c = right[idx]
            L_l = jnp.einsum('xy,sy->sx', P_matrices[l], L_val[l])
            L_r = jnp.einsum('xy,sy->sx', P_matrices[r_c], L_val[r_c])
            return L_val.at[i].set(L_l * L_r)
        
        final_L = jax.lax.fori_loop(N_taxa, N_nodes, node_update, L_nodes)
        root_L = final_L[N_nodes - 1]
        site_liks = jnp.dot(root_L, pi)
        return site_liks

    site_liks_by_rate = jax.vmap(calc_rate_lik)(rates)
    mean_site_liks = jnp.mean(site_liks_by_rate, axis=0)
    
    log_likelihood = jnp.sum(jnp.log(jnp.maximum(mean_site_liks, 1e-30)))
    numpyro.factor("phylo_lik", log_likelihood)

print("Starting BI fit (Spatial Heterogeneity + Gamma) on Real Primate Data...")
m.fit(model, num_samples=1, num_warmup=1, num_chains=1)

post = m.posteriors
if post is not None:
    df = pd.DataFrame({
        'kappa': np.array(post['kappa']).flatten(),
        'alpha': np.array(post['alpha']).flatten()
    })
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "bi_gamma_post.csv")
    df.to_csv(output_path, index=False)
    print(f"Posteriors saved to {output_path}")
