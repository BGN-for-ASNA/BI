from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from importlib.resources import files

model_name = "13.Gaussian processes"

print(f'Running BI for {model_name}')
# Setup device------------------------------------------------
m = bi(platform='cpu', backend='tfp')

# Import Data & Data Manipulation ------------------------------------------------
# Import
data_path = m.load.kline2(only_path=True)
m.data(data_path, sep=';') 

data_path2 = files('BI.Resources') / 'islandsDistMatrix.csv'
islandsDistMatrix = pd.read_csv(data_path2, index_col=0)

m.data_to_model(['total_tools', 'population'])
m.data_on_model["society"] = jnp.arange(0,10)# index observations
m.data_on_model["Dmat"] = islandsDistMatrix.values.astype(jnp.float32) # Distance matrix

def model(Dmat, population, society, total_tools):
    a = yield m.dist.exponential(1, name = 'a')
    b = yield m.dist.exponential(1, name = 'b')
    g = yield m.dist.exponential(1, name = 'g')

    # non-centered Gaussian Process prior
    etasq = yield m.dist.exponential(2, name = 'etasq')
    rhosq = yield m.dist.exponential(0.5, name = 'rhosq')
    SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
    SIGMA = SIGMA.at[jnp.diag_indices(Dmat.shape[0])].add(0.01)
    k = yield m.dist.multivariate_normal(jnp.zeros(Dmat.shape[0]), SIGMA, name = 'k')

    lambda_ = a * population**b / g * jnp.exp(k[society])

    yield m.dist.poisson(lambda_, obs=total_tools)
    
# Run sampler ------------------------------------------------
print("Fitting BI model...")
m.fit(model, num_samples=1000, num_warmup=1000)
print("BI Summary:")
m.summary()

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

# 4. Comparison
param_map = {'a': 'a', 'b': 'b', 'g': 'g', 'etasq': 'etasq', 'rhosq': 'rhosq'}
plot_comparaison(m, df_stan, param_map, model_name=model_name)

# 5. Parameter Recovery Analysis
def estimate_rec(Dm_rec, P_rec, a_true, b_true, g_true, etasq_true, rhosq_true, society):
    K_sim = etasq_true * jnp.exp(-rhosq_true * jnp.square(Dm_rec))
    K_sim = K_sim.at[jnp.diag_indices(Dm_rec.shape[0])].add(0.01)
    K_sim = K_sim.astype(jnp.float32)
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    k_sim = m_rec.dist.multivariate_normal(jnp.zeros(Dm_rec.shape[0], dtype=jnp.float32), K_sim, sample = True)
    
    # lambda_sim: a * P^b / g * exp(k)
    lambda_sim = a_true * P_rec**b_true / g_true * jnp.exp(k_sim[society])
    total_tools_sim = m_rec.dist.poisson(lambda_sim.astype(jnp.float32), sample = True)
    
    m_rec.data_on_model = {
        'population': jnp.array(P_rec, dtype=jnp.float32),
        'total_tools': jnp.array(total_tools_sim, dtype=jnp.int32),
        'Dmat': jnp.array(Dm_rec, dtype=jnp.float32),
        'society': jnp.array(society, dtype=jnp.int32)
    }
    
    def model_rec(Dmat, population, society, total_tools):
        a = yield m_rec.dist.exponential(1, name = 'a')
        b = yield m_rec.dist.exponential(1, name = 'b')
        g = yield m_rec.dist.exponential(1, name = 'g')

        # non-centered Gaussian Process prior
        etasq = yield m_rec.dist.exponential(2, name = 'etasq')
        rhosq = yield m_rec.dist.exponential(0.5, name = 'rhosq')
        SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
        SIGMA = SIGMA.at[jnp.diag_indices(Dmat.shape[0])].add(0.01)
        k = yield m_rec.dist.multivariate_normal(jnp.zeros(Dmat.shape[0]), SIGMA, name = 'k')

        lambda_ = a * population**b / g * jnp.exp(k[society])
        yield m_rec.dist.poisson(lambda_, obs=total_tools)
        
    m_rec.fit(model_rec, num_samples=1000, num_warmup=1000, progress_bar=False)
    sum_df = m_rec.summary()
    return sum_df.iloc[:, 0]

def param_recovery(nsim):
    results = []
    N_islands = 30 # Increased island count for better recovery signal
    
    for i in range(nsim):
        # 1. Simulate Island Geography (random coordinates in 2D space)
        pos = np.random.uniform(0, 10, size=(N_islands, 2))
        Dm_rec = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=-1))
        
        # 2. Simulate Populations (log-scale similar to real data)
        P_rec = np.exp(np.random.normal(10, 1.5, size=N_islands))
        society_rec = np.arange(N_islands)
        
        # 3. Sample true parameters from priors for this simulation
        # Using a temporary bi instance to sample from priors
        m_tmp = bi(platform='cpu')
        a_true = float(m_tmp.dist.exponential(1.0, sample=True))
        b_true = float(m_tmp.dist.exponential(1.0, sample=True))
        g_true = float(m_tmp.dist.exponential(1.0, sample=True))
        etasq_true = float(m_tmp.dist.exponential(2.0, sample=True))
        rhosq_true = float(m_tmp.dist.exponential(0.5, sample=True))

        # 4. Estimate
        est = estimate_rec(Dm_rec, P_rec, a_true, b_true, g_true, etasq_true, rhosq_true, society_rec)
        
        for param, true_val in zip(['a', 'b', 'g', 'etasq', 'rhosq'], 
                                 [a_true, b_true, g_true, etasq_true, rhosq_true]):
            # TFP summary might have [0] suffix
            est_key = param if param in est.index else f"{param}[0]"
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': true_val,
                'estimations': float(est[est_key])
            })
    
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("\nRunning Parameter Recovery...")
nsim = int(os.environ.get("BI_NSIM", 5))
res = param_recovery(nsim = nsim)
