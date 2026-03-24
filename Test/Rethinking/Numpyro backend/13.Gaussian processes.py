from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

model_name = "13.Gaussian processes"

print(f'Running BI for {model_name}')

# 1. Data Simulation (Gaussian Process - Islands)
N_islands = 10
Dm = jnp.array([
    [0, 0.48, 0.72, 3.59, 4.34, 4.38, 4.8, 5.25, 5.37, 5.86],
    [0.48, 0, 0.32, 3.2, 3.96, 4, 4.41, 4.88, 5.01, 5.49],
    [0.72, 0.32, 0, 2.9, 3.66, 3.7, 4.12, 4.6, 4.73, 5.23],
    [3.59, 3.2, 2.9, 0, 1, 1.13, 1.5, 1.95, 2.22, 2.71],
    [4.34, 3.96, 3.66, 1, 0, 0.22, 0.61, 1, 1.45, 1.93],
    [4.38, 4, 3.7, 1.13, 0.22, 0, 0.4, 0.9, 1.25, 1.74],
    [4.8, 4.41, 4.12, 1.5, 0.61, 0.4, 0, 0.51, 1.01, 1.49],
    [5.25, 4.88, 4.6, 1.95, 1, 0.9, 0.51, 0, 0.61, 1.1],
    [5.37, 5.01, 4.73, 2.22, 1.45, 1.25, 1.01, 0.61, 0, 0.49],
    [5.86, 5.49, 5.23, 2.71, 1.93, 1.74, 1.49, 1.1, 0.49, 0]
])

# Known parameters
f_true = 3.0
a_true = 1.0
b_true = 1.0
g_true = 0.1
P_vals = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

m_sim = bi(platform='cpu')
# Simulate k using GP kernel
K_sim = a_true**2 * jnp.exp(-b_true**2 * Dm**2) + jnp.diag(jnp.repeat(g_true**2 + 1e-2, N_islands))
k_sim = m_sim.dist.multivariate_normal(jnp.zeros(N_islands), K_sim, sample=True)

lambda_sim = jnp.exp(f_true + k_sim) * P_vals
D_sim = m_sim.dist.poisson(lambda_sim, sample=True)

d = pd.DataFrame(dict(island=jnp.arange(N_islands), P=P_vals, D=D_sim))

# 2. BI Model
m = bi(platform='cpu')
m.data_on_model = dict(
    P=P_vals.astype(jnp.float32),
    D=D_sim.astype(jnp.int32),
    Dm=Dm.astype(jnp.float32),
    N=N_islands
)

def model_gp(P, D, Dm, N):
    a = m.dist.exponential(1.0, name='a')
    b = m.dist.exponential(1.0, name='b')
    g = m.dist.exponential(1.0, name='g')
    f = m.dist.normal(3.0, 1.0, name='f')
    K = a**2 * jnp.exp(-b**2 * Dm**2) + jnp.diag(jnp.repeat(g**2 + 1e-2, N))
    k = m.dist.multivariate_normal(jnp.zeros(N), K, name='k')
    lambda_ = jnp.exp(f + k) * P
    m.dist.poisson(lambda_, obs=D)

print("Fitting BI model...")
m.fit(model_gp, num_samples=1000, num_warmup=1000)
bi_summary = m.summary()
print("BI Summary:")
print(bi_summary)

# 3. Stan Model
stan_code = """
data{
    int N;
    array[N] int D;
    vector[N] P;
    matrix[N,N] Dm;
}
parameters{
    real f;
    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> g;
    vector[N] k;
}
model{
    matrix[N,N] K;
    a ~ exponential( 1 );
    b ~ exponential( 1 );
    g ~ exponential( 1 );
    f ~ normal( 3 , 1 );
    for ( i in 1:(N-1) )
        for ( j in (i+1):N ) {
            K[i,j] = a^2 * exp( -b^2 * Dm[i,j]^2 );
            K[j,i] = K[i,j];
        }
    for ( i in 1:N ) K[i,i] = a^2 + g^2 + 1e-2;
    k ~ multi_normal( rep_vector(0,N) , K );
    for ( i in 1:N ) {
        D[i] ~ poisson( exp(f + k[i]) * P[i] );
    }
}
"""
stan_data = {
    'N': N_islands,
    'D': D_sim.tolist(),
    'P': P_vals.tolist(),
    'Dm': Dm.tolist()
}
stan_df = build_stan_model(stan_code, data=stan_data, chains=4)

# 4. Comparison
param_map = {'a': 'a', 'b': 'b', 'g': 'g', 'f': 'f'}
bi_samples = m.posteriors
bi_df = pd.DataFrame({k: bi_samples[k] for k in param_map.keys()})
plot_comparaison(bi_df, stan_df, param_map, model_name=model_name)

# 5. Parameter Recovery Analysis
def estimate_rec(Dm, P_vals, f_true, a_true, b_true, g_true):
    N = len(P_vals)
    K_sim = a_true**2 * jnp.exp(-b_true**2 * Dm**2) + jnp.diag(jnp.repeat(g_true**2 + 1e-2, N))
    k_sim = np.random.multivariate_normal(np.zeros(N), K_sim)
    lambda_sim = np.exp(f_true + k_sim) * P_vals
    D_sim = np.random.poisson(lambda_sim)
    
    m_rec = bi(print_devices_found=False)
    m_rec.data_on_model = {
        'P': jnp.array(P_vals, dtype=jnp.float32),
        'D': jnp.array(D_sim, dtype=jnp.int32),
        'Dm': jnp.array(Dm, dtype=jnp.float32),
        'N': N
    }
    def model_rec(P, D, Dm, N):
        a = m_rec.dist.exponential(1.0, name='a')
        b = m_rec.dist.exponential(1.0, name='b')
        g = m_rec.dist.exponential(1.0, name='g')
        f = m_rec.dist.normal(3.0, 1.0, name='f')
        K = a**2 * jnp.exp(-b**2 * Dm**2) + jnp.diag(jnp.repeat(g**2 + 1e-2, N))
        k = m_rec.dist.multivariate_normal(jnp.zeros(N), K, name='k')
        lambda_ = jnp.exp(f + k) * P
        m_rec.dist.poisson(lambda_, obs=D)
        
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    sum_df = m_rec.summary()
    return sum_df.iloc[:, 0]

def param_recovery(Dm, P_vals, f_sims, a_sims, b_sims, g_sims, nsim):
    results = []
    # Use more data for recovery (duplicate islands to 30)
    Dm_ext = np.tile(Dm, (3, 3))
    P_ext = np.tile(P_vals, 3)
    
    for i in range(nsim):
        est = estimate_rec(Dm_ext, P_ext, f_sims[i], a_sims[i], b_sims[i], g_sims[i])
        for param in ['f', 'a', 'b']:
            true_val = f_sims[i] if param == 'f' else (a_sims[i] if param == 'a' else b_sims[i])
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(true_val),
                'estimations': float(est[param])
            })
    
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("\nRunning Parameter Recovery (100 simulations)...")
nsim = int(os.environ.get("BI_NSIM", 100))
f_sims = np.random.normal(3.0, 0.5, size=(nsim,))
a_sims = np.random.exponential(1.0, size=(nsim,))
b_sims = np.random.exponential(1.0, size=(nsim,))
g_sims = np.random.exponential(0.5, size=(nsim,))

res = param_recovery(Dm, P_vals, f_sims, a_sims, b_sims, g_sims, nsim = nsim)
