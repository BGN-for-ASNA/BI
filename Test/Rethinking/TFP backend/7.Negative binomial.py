from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "7.Negative binomial"

print(f'Running BI for {model_name}')
m = bi(platform='cpu', backend='tfp')

# 1. Data Simulation ----------------------------------------
np.random.seed(1)
num_days = 300
y_days = np.random.poisson(1.5, num_days)
num_weeks = 40
y_weeks = np.random.poisson(0.5 * 7, num_weeks)

y_all = np.concatenate([y_days, y_weeks])
exposure = np.concatenate([np.repeat(1, num_days), np.repeat(7, num_weeks)])
monastery = np.concatenate([np.repeat(0, num_days), np.repeat(1, num_weeks)])
df = pd.DataFrame({'y': y_all, 'days': exposure, 'monastery': monastery})
df['log_days'] = np.log(df['days'])

m.df = df
m.data_to_model(['y', 'log_days', 'monastery'])

def model_bi(y, log_days, monastery):
    a = yield m.dist.normal(0, 1, name='a')
    b = yield m.dist.normal(0, 1, name='b')
    lambda_ = jnp.exp(log_days + a + b * monastery)
    yield m.dist.poisson(lambda_, obs=y)

print("Fitting BI model...")
m.fit(model_bi, num_samples=1000, num_warmup=1000)
print("BI Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    array[N] int y;
    vector[N] log_days;
    vector[N] monastery;
}
parameters {
    real a;
    real b;
}
model {
    vector[N] lambda;
    a ~ normal(0, 1);
    b ~ normal(0, 1);
    lambda = exp(log_days + a + b * monastery);
    y ~ poisson(lambda);
}
"""
data_stan = {
    'N': len(df),
    'y': df['y'].values.astype(int).tolist(),
    'log_days': df['log_days'].values.astype(float).tolist(),
    'monastery': df['monastery'].values.astype(float).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
bi_df = prepare_bi_data(m)
param_map = {'a[0]': 'a', 'b[0]': 'b'}
plot_comparaison(bi_df, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(log_days, monastery, a_true, b_true):
    lambda_sim = np.exp(log_days + a_true + b_true * monastery)
    y_sim = np.random.poisson(lambda_sim)
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'log_days': jnp.array(log_days),
        'monastery': jnp.array(monastery),
        'y': jnp.array(y_sim)
    }
    def model_rec(log_days, monastery, y):
        a = yield m_rec.dist.normal(0, 1, name='a')
        b = yield m_rec.dist.normal(0, 1, name='b')
        lambda_ = jnp.exp(log_days + a + b * monastery)
        yield m_rec.dist.poisson(lambda_, obs=y)
        
    m_rec.fit(model_rec, num_samples=500, num_warmup=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(log_days, monastery, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(log_days, monastery, a_sims[i], b_sims[i])
        results.append({'sim': i, 'parameter': 'a', 'simulated': float(a_sims[i]), 'estimations': float(est['a'])})
        results.append({'sim': i, 'parameter': 'b', 'simulated': float(b_sims[i]), 'estimations': float(est['b'])})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BI_NSIM', 10))
a_sims = np.random.normal(0, 1, nsim_test)
b_sims = np.random.normal(0, 1, nsim_test)

recovery_results = param_recovery(df.log_days.values, df.monastery.values, a_sims, b_sims, nsim=nsim_test)
