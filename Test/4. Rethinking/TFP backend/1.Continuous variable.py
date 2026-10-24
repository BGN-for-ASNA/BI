from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "1.Continuous variable"

print(f'Running BF for {model_name}')
m = bf(platform='cpu', backend='tfp')

# import data ------------------------------------------------
data_path = m.load.howell1(only_path = True)
m.data(data_path, sep=';') 
m.df = m.df[m.df.age > 18]
m.scale(['weight'])

# define model ------------------------------------------------
def model_BF(weight, height):
    a = yield m.dist.normal(178, 20, name='a')
    b = yield m.dist.log_normal(0, 1, name='b')  
    s = yield m.dist.uniform(0, 50, name='s')   
    y = yield m.dist.normal(a+b*weight, s, shape = (1,), obs = height)


# Run sampler ------------------------------------------------
m.fit(model_BF, num_samples=1000, num_warmup=1000) 
m.summary()

print('Running Stan')
stan_code = """
data{
  int<lower=0> N;
  vector[N] height;
  vector[N] weight;
}
parameters{
  real a;
  real<lower=0> b;
  real<lower=0,upper=50> s;
}
model{
  vector[N] mu;
  s ~ uniform( 0 , 50 );
  b ~ lognormal( 0 , 1 );
  a ~ normal( 178 , 20 );
  for ( i in 1:N ) {
    mu[i] = a + b* weight[i] ;
  }
  height ~ normal( mu , s );  
}
"""
data = {
  'N': len(m.df),
  'height': m.df.height.values,
  'weight': m.df.weight.values,
}

stan_df = build_stan_model(stan_code, data = data, chains=4)

print('Posterior distributions comparison')
param_map = {'a[0]': 'a', 'b[0]': 'b', 's[0]': 's'}
plot_comparaison(m, stan_df, param_map=param_map, model_name=model_name)

print('Running Parameters recovery')
def model_rec(weight, height):    
    a = yield m.dist.normal( 0, 1, name='a')   
    b = yield m.dist.normal(0, 1, name='b')
    s = yield m.dist.exponential( 1, name='s')
    yield m.dist.normal(a + b * weight , s, obs=height)

def simulate_height(weight, a, b, s):    
    weight_scaled = (weight - weight.mean())/weight.std()
    height = m.dist.normal( a + b * weight_scaled , s, sample = True)
    return weight_scaled, height

def estimate(weight, a, b, s):
    weight_scaled, height = simulate_height(weight, a, b, s)
    m_rec = bf(print_devices_found=False, backend='tfp')
    m_rec.df = pd.DataFrame({"weight": weight_scaled, "height": height})
    m_rec.data_to_model(['weight', 'height'])
    m_rec.fit(model_rec, num_samples=500, num_warmup=500, progress_bar=False) 
    sum_df = m_rec.summary()
    return sum_df.iloc[:,0]

def param_recovery(weight_data, a_sims, b_sims, s_sims, nsim):
    results = []
    for i in range(nsim):
        estimations = estimate(weight_data[i,:], a_sims[i,:], b_sims[i,:], s_sims[i,:])
        for param in ['a', 'b', 's']:
            true_val = a_sims[i,0] if param == 'a' else (b_sims[i,0] if param == 'b' else s_sims[i,0])
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(true_val),
                'estimations': float(estimations[param])
            })
    
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

N = 200
nsim = int(os.getenv('BF_NSIM', 10))
a_sims = np.random.normal(0, 1, size=(nsim, 1))
b_sims = np.random.normal(0, 1, size=(nsim, 1))
s_sims = np.random.exponential(1, size=(nsim, 1))
weight_data = np.random.normal(80, 30, size=(nsim, N))

res = param_recovery(weight_data, a_sims, b_sims, s_sims, nsim = nsim)

# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(weight, height):
    a = m_ref.dist.normal(178, 20)
    b = m_ref.dist.log_normal(0, 1)
    s = m_ref.dist.uniform(0, 50)
    m_ref.dist.normal(a + b * weight, s, obs=height)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'weight': m.data_on_model['weight'], 'height': m.data_on_model['height']})
