from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "11.Varying intercepts"

print(f'Running BI for {model_name}')
m = bi(platform='cpu', backend='tfp')

# 1. Data Loading -------------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BI", "Resources")) + os.sep
df = pd.read_csv(data_path + 'reedfrogs.csv', sep=';')
df["tank"] = np.arange(len(df))

m.df = df
m.data_on_model = {
    'tank': jnp.array(df.tank.values, dtype=jnp.int32),
    'surv': jnp.array(df.surv.values, dtype=jnp.float32), # Cast to float for TFP
    'density': jnp.array(df.density.values, dtype=jnp.float32) # Cast to float for TFP
}

def model_bi(tank, surv, density):
    sigma = yield m.dist.exponential(1.0, shape=(1,), name='sigma')
    a_bar = yield m.dist.normal(0.0, 1.5, shape=(1,), name='a_bar')
    # Non-Centered Parametrization
    z = yield m.dist.normal(0, 1, shape=(48,), name='z')
    alpha = a_bar + z * sigma
    p = alpha[tank]
    yield m.dist.binomial(total_count=density, logits=p, obs=surv)

print("Fitting BI model...")
m.fit(model_bi, num_samples=1000, num_warmup=1000)
print("BI Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N_obs;
    array[N_obs] int S;
    array[N_obs] int D;
    array[N_obs] int tank;
}
parameters {
    real a_bar;
    vector[48] a;
    real<lower=0> sigma;
}
model {
    vector[N_obs] p;
    sigma ~ exponential(1);
    a_bar ~ normal(0, 1.5);
    a ~ normal(a_bar, sigma);
    for (i in 1:N_obs) {
        p[i] = a[tank[i]];
    }
    S ~ binomial_logit(D, p);
}
"""
data_stan = {
    'N_obs': len(df),
    'S': df['surv'].values.astype(int).tolist(),
    'D': df['density'].values.astype(int).tolist(),
    'tank': (df['tank'].values + 1).astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a_bar[0]': 'a_bar',
    'sigma[0]': 'sigma'
}
bi_df = prepare_bi_data(m)
plot_comparaison(bi_df, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(tank, density, a_bar_true, sigma_true, alpha_true):
    p_true = alpha_true[tank]
    surv_sim = np.random.binomial(density.astype(int), jax.nn.sigmoid(p_true)).astype(float)
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'tank': jnp.array(tank, dtype=jnp.int32),
        'surv': jnp.array(surv_sim, dtype=jnp.float32),
        'density': jnp.array(density, dtype=jnp.float32)
    }
    
    def model_rec(tank, surv, density):
        sigma = yield m_rec.dist.exponential(1.0, name='sigma')
        a_bar = yield m_rec.dist.normal(0.0, 1.5, name='a_bar')
        z = yield m_rec.dist.normal(0, 1, shape=(48,), name='z')
        alpha = a_bar + z * sigma
        p = alpha[tank]
        yield m_rec.dist.binomial(total_count=density, logits=p, obs=surv)
        
    m_rec.fit(model_rec, num_samples=500, num_warmup=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(tank, density, a_bar_sims, sigma_sims, alpha_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(tank, density, a_bar_sims[i], sigma_sims[i], alpha_sims[i])
        for param in ['a_bar', 'sigma']:
            true_val = a_bar_sims[i, 0] if param == 'a_bar' else sigma_sims[i, 0]
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(true_val),
                'estimations': float(est[param])
            })
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BI_NSIM', 10))
a_bar_sims = np.random.normal(0, 1.5, (nsim_test, 1))
sigma_sims = np.random.exponential(1.0, (nsim_test, 1))
alpha_sims = np.random.normal(a_bar_sims, sigma_sims, (nsim_test, 48))

# Tile data for stronger signal (50x)
tank_ext = np.tile(df.tank.values, 50)
dens_ext = np.tile(df.density.values, 50).astype(float)
recovery_results = param_recovery(tank_ext, dens_ext, a_bar_sims, sigma_sims, alpha_sims, nsim=nsim_test)
