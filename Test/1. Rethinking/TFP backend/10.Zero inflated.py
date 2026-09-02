from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import expit

model_name = "10.Zero inflated"

print(f'Running BF for {model_name}')
m = bf(platform='cpu', backend='tfp')

# 1. Data Simulation -----------------------------------------
# Simulate production of manuscripts
prob_drink = 0.2  # 20% of days
rate_work = 1     # average 1 manuscript per day
N = 365

np.random.seed(42)
drink = np.random.binomial(1, prob_drink, N)
y = (1 - drink) * np.random.poisson(rate_work, N)
df = pd.DataFrame({'y': y})

m.df = df
m.data_on_model = {'y': jnp.array(y, dtype=jnp.float32)} # Cast to float for TFP

def model_BF(y):
    ap = yield m.dist.normal(-1.5, 1.0, name='ap')
    # Broadcast to match 'y' shape for Independent wrapper in tfp_dists.py
    p = jnp.full_like(y, expit(ap))
    al = yield m.dist.normal(1.0, 0.5, name='al')
    lambda_ = jnp.full_like(y, jnp.exp(al))
    yield m.dist.zero_inflated_poisson(p, lambda_, obs=y)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    array[N] int y;
}
parameters {
    real al;
    real ap;
}
model {
    real p;
    real lambda;
    al ~ normal(1, 0.5);
    ap ~ normal(-1.5, 1);
    
    lambda = exp(al);
    p = inv_logit(ap);
    
    for (n in 1:N) {
        if (y[n] == 0)
            target += log_mix(p, 0, poisson_lpmf(0 | lambda));
        else
            target += log1m(p) + poisson_lpmf(y[n] | lambda);
    }
}
"""
data_stan = {
    'N': N,
    'y': y.astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {'ap[0]': 'ap', 'al[0]': 'al'}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(y_sim):
    m_rec = bf(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {'y': jnp.array(y_sim, dtype=jnp.float32)}
    def model_rec(y):
        ap = yield m_rec.dist.normal(-1.5, 1.0, name='ap')
        p = jnp.full_like(y, expit(ap))
        al = yield m_rec.dist.normal(1.0, 0.5, name='al')
        lambda_ = jnp.full_like(y, jnp.exp(al))
        yield m_rec.dist.zero_inflated_poisson(p, lambda_, obs=y)
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(y_true, ap_sims, al_sims, nsim):
    results = []
    N_rec = len(y_true) * 50 # 50x signal boost
    for i in range(nsim):
        p_true = float(expit(ap_sims[i]))
        lambda_true = float(np.exp(al_sims[i]))
        drink_sim = np.random.binomial(1, p_true, N_rec)
        y_sim = (1 - drink_sim) * np.random.poisson(lambda_true, N_rec)
        
        est = estimate(y_sim)
        results.append({'sim': i, 'parameter': 'ap', 'simulated': float(ap_sims[i]), 'estimations': float(est['ap'])})
        results.append({'sim': i, 'parameter': 'al', 'simulated': float(al_sims[i]), 'estimations': float(est['al'])})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BF_NSIM', 10))
ap_sims = np.random.normal(-1.5, 1.0, nsim_test)
al_sims = np.random.normal(1.0, 0.5, nsim_test)

recovery_results = param_recovery(y, ap_sims, al_sims, nsim=nsim_test)


# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(y):
    ap = m_ref.dist.normal(-1.5, 1.0)
    p = jnp.full_like(y, expit(ap))
    al = m_ref.dist.normal(1.0, 0.5)
    lambda_ = jnp.full_like(y, jnp.exp(al))
    m_ref.dist.zero_inflated_poisson(p, lambda_, obs=y)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'y': m.data_on_model['y']})
