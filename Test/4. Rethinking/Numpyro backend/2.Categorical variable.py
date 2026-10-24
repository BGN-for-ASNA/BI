from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "2.Categorical variable"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# import data ------------------------------------------------
data_path = m.load.milk(only_path = True)
m.data(data_path, sep=';') 
m.index(["clade"])
m.scale(['kcal_per_g'])

def model_BF(kcal_per_g, index_clade):
    a = m.dist.normal(0, 0.5, shape=(4,))
    s = m.dist.exponential( 1)    
    mu = a[index_clade]
    m.dist.normal(mu, s, obs=kcal_per_g)

m.data_to_model(['kcal_per_g', "index_clade"])
m.fit(model_BF) 
m.summary()

print('Running Stan')
stan_code = """
data {
    int<lower=1> N;
    int<lower=1> K;
    vector[N] kcal;
    array[N] int clade;
}
parameters {
    vector[K] a;
    real<lower=0> s;
}
model {
    s ~ exponential(1);
    a ~ normal(0, 0.5);
    kcal ~ normal(a[clade], s);
}
"""
data = {
    'N': len(m.df),
    'K': 4,
    'kcal': m.df.kcal_per_g.values,
    'clade': (m.df.index_clade.values + 1).astype(int)
}

stan_df = build_stan_model(stan_code, data = data, chains=4)

print('Posterior distributions comparison')
param_map = {
    'a[0]': 'a[1]',
    'a[1]': 'a[2]',
    'a[2]': 'a[3]',
    'a[3]': 'a[4]',
    's': 's'
}
plot_comparaison(m, stan_df, param_map=param_map, model_name=model_name)

print('Running Parameters recovery')
def model_rec(kcal_per_g, index_clade):
    a = m.dist.normal(0, 0.5, shape=(4,))
    s = m.dist.exponential( 1)    
    mu = a[index_clade]
    m.dist.normal(mu, s, obs=kcal_per_g)

def simulate_data(index_clade, a, s):    
    kcal_per_g = m.dist.normal(a[index_clade], s, sample = True)
    return kcal_per_g

def estimate(index_clade, a, s):
    kcal_per_g = simulate_data(index_clade, a, s)
    m_rec = bf(print_devices_found=False)
    m_rec.df = pd.DataFrame({"kcal_per_g": kcal_per_g, "index_clade": index_clade})
    m_rec.data_to_model(['kcal_per_g', 'index_clade'])
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False) 
    summary = m_rec.summary()
    return summary.iloc[:,0]

def param_recovery(index_clade, a_sim, s_sim, nsim):
    results = []
    for i in range(nsim):
        estimations = estimate(index_clade, a_sim[i], s_sim[i])
        # a has 4 components
        for j in range(4):
            results.append({
                'sim': i,
                'parameter': f'a[{j}]',
                'simulated': float(a_sim[i,j]),
                'estimations': float(estimations[f'a[{j}]'])
            })
        results.append({
            'sim': i,
            'parameter': 's',
            'simulated': float(s_sim[i]),
            'estimations': float(estimations['s'])
        })

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)    
    return df_res

N = 100
nsim = int(os.environ.get("BF_NSIM", 10)) 
a_sim = np.random.normal(0, 0.5, size=(nsim, 4))
s_sim = np.random.exponential(1, size=(nsim,))
index_clade = np.random.choice([0,1,2,3], size=N)

res = param_recovery(index_clade, a_sim, s_sim, nsim = nsim)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
