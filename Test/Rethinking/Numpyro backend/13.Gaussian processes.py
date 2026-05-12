#%%
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
# Setup device------------------------------------------------
m = bi(platform='cpu')

# Import Data & Data Manipulation ------------------------------------------------
# Import
data_path = m.load.kline2(only_path=True)
m.data(data_path, sep=';') 

from importlib.resources import files
data_path2 = files('BI.Resources') / 'islandsDistMatrix.csv'
islandsDistMatrix = pd.read_csv(data_path2, index_col=0)

m.data_to_model(['total_tools', 'population'])
m.data_on_model["society"] = jnp.arange(0,10)# index observations
m.data_on_model["Dmat"] = islandsDistMatrix.values # Distance matrix

def model(Dmat, population, society, total_tools):
    a = m.dist.exponential(1, name = 'a')
    b = m.dist.exponential(1, name = 'b')
    g = m.dist.exponential(1, name = 'g')

    # non-centered Gaussian Process prior
    etasq = m.dist.exponential(2, name = 'etasq')
    rhosq = m.dist.exponential(0.5, name = 'rhosq')
    SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
    SIGMA = SIGMA.at[jnp.diag_indices(Dmat.shape[0])].add(0.01)
    k = m.dist.multivariate_normal(0, SIGMA, name = 'k')

    lambda_ = a * population**b / g * jnp.exp(k[society])

    m.dist.poisson(lambda_, obs=total_tools)
    
#%%
# Run sampler ------------------------------------------------
print("Fitting BI model...")
m.fit(model, num_samples=1000, num_warmup=1000)
print("BI Summary:")
m.summary()

#%%
# 3. Stan Model 

stan_code = """ 
functions{
  matrix cov_GPL2(matrix x, real sq_alpha, real sq_rho, real delta) {
    int N = dims(x)[1];
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sq_alpha + delta;
      for (j in (i + 1):N) {
        K[i, j] = sq_alpha * exp(-sq_rho * square(x[i,j]) );
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sq_alpha + delta;
    return K;
  }
}

data{
  array[10] int T;
  array[10] int society;
  array[10] int P;
  matrix[10,10] Dmat;
}

parameters{
 real<lower=0> a;
 real<lower=0> b;
 real<lower=0> etasq;
 real<lower=0> g; 
 real<lower=0> rhosq;
 vector[10] k;
}

model{
  vector[10] lambda;
  matrix[10,10] SIGMA;
  rhosq ~ exponential( 0.5 );
  etasq ~ exponential( 2 );
  a ~ exponential( 1 );
  b ~ exponential( 1 );
  g ~ exponential( 1 );

  SIGMA = cov_GPL2(Dmat, etasq, rhosq, 0.01);
  k ~ multi_normal( rep_vector(0,10) , SIGMA );
  for ( i in 1:10 ) {
    lambda[i] = (a * P[i]^b/g) * exp(k[society[i]]);
  }
  T ~ poisson( lambda );
}
"""
data = {
    'T' : m.df["total_tools"].values.astype(int),
    'P' : m.df["population"].values.astype(int),
    'society' : np.array(m.data_on_model['society']+1).astype(int),
    'Dmat' : np.array(islandsDistMatrix)
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data= data, chains=4)

#%%
# 4. Comparison
param_map = {'a': 'a', 'b': 'b', 'g': 'g', 'etasq': 'etasq', 'rhosq': 'rhosq'}
bi_df = prepare_bi_data(m)
plot_comparaison(bi_df, df_stan, param_map, model_name=model_name)

#%%
# 5. Parameter Recovery Analysis
def estimate_rec(Dm_rec, P_rec, a_true, b_true, g_true, etasq_true, rhosq_true, society):
    K_sim = etasq_true * jnp.exp(-rhosq_true * jnp.square(Dm_rec))
    K_sim = K_sim.at[jnp.diag_indices(Dm_rec.shape[0])].add(0.01)
    k_sim =  m.dist.multivariate_normal(0, K_sim, sample = True)
    
    # lambda_sim: a * P^b / g * exp(k)
    lambda_sim = a_true * P_rec**b_true / g_true * jnp.exp(k_sim[society])
    total_tools_sim = m.dist.poisson(lambda_sim, sample = True)
    
    m.data_on_model = {
        'population': jnp.array(P_rec, dtype=jnp.float32),
        'total_tools': total_tools_sim,
        'Dmat': jnp.array(Dm_rec, dtype=jnp.float32),
        'society': society
    }
    
    def model_rec(Dmat, population, society, total_tools):
        a = m.dist.exponential(1, name = 'a')
        b = m.dist.exponential(1, name = 'b')
        g = m.dist.exponential(1, name = 'g')

        # non-centered Gaussian Process prior
        etasq = m.dist.exponential(2, name = 'etasq')
        rhosq = m.dist.exponential(0.5, name = 'rhosq')
        SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
        SIGMA = SIGMA.at[jnp.diag_indices(Dmat.shape[0])].add(0.01)
        k = m.dist.multivariate_normal(0, SIGMA, name =  'k')

        lambda_ = a * population**b / g * jnp.exp(k[society])
        m.dist.poisson(lambda_, obs=total_tools)
        
    m.fit(model_rec, num_samples=1000, num_warmup=1000, progress_bar=False)
    sum_df = m.summary()
    return sum_df.iloc[:, 0]

def param_recovery(Dm, P_vals, a_sims, b_sims, g_sims, etasq_sims, rhosq_sims, society, nsim):
    results = []
    # Keep original 10 islands
    Dm_rec = Dm
    P_rec = P_vals
    
    for i in range(nsim):
        est = estimate_rec(Dm_rec, P_rec, a_sims[i], b_sims[i], g_sims[i], etasq_sims[i], rhosq_sims[i], society)
        for param in ['a', 'b', 'g', 'etasq', 'rhosq']:
            if param == 'a': true_val = a_sims[i]
            elif param == 'b': true_val = b_sims[i]
            elif param == 'g': true_val = g_sims[i]
            elif param == 'etasq': true_val = etasq_sims[i]
            else: true_val = rhosq_sims[i]
            
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(true_val),
                'estimations': float(est[param])
            })
    
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("\nRunning Parameter Recovery...")
nsim = int(os.environ.get("BI_NSIM", 5))
a_sims = m.dist.exponential(1.0, sample=True, shape=(nsim,))
b_sims = m.dist.exponential(1.0, sample=True, shape=(nsim,))
g_sims = m.dist.exponential(1.0, sample=True, shape=(nsim,))
etasq_sims = m.dist.exponential(2.0, sample=True, shape=(nsim,))
rhosq_sims = m.dist.exponential(0.5, sample=True, shape=(nsim,))

Dm = m.data_on_model['Dmat']
P_vals = m.data_on_model['population']
society = m.data_on_model["society"]
res = param_recovery(Dm, P_vals, a_sims, b_sims, g_sims, etasq_sims, rhosq_sims, society, nsim = nsim)



# %%
