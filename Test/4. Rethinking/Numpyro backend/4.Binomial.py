from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "4.Binomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Preparation ----------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'chimpanzees.csv', sep=';')
m.df = df
m.data_on_model = {'pulled_left': jnp.array(df.pulled_left.values)}

def model_BF(pulled_left):
    a = m.dist.normal(0, 10)
    m.dist.binomial(total_count=1, logits=a, obs=pulled_left)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=0> N;
    array[N] int pulled_left;
}
parameters {
    real a;
}
model {
    a ~ normal(0, 10);
    pulled_left ~ binomial_logit(1, a);
}
"""
data_stan = {
    'N': len(df),
    'pulled_left': df.pulled_left.values.astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
plot_comparaison(m, df_stan, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def sim_pulled_left(a):
    return m.dist.binomial(total_count=1, logits=a, sample=True, shape=(len(df),))

def estimate(a):
    pulled_left_sim = sim_pulled_left(a)
    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {"pulled_left": pulled_left_sim}
    def model_rec(pulled_left):
        a = m_rec.dist.normal(0, 10)
        m_rec.dist.binomial(total_count=1, logits=a, obs=pulled_left)
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(a_true, nsim):
    results = []
    for i in range(nsim):
        est = estimate(a_true[i, 0])
        results.append({
            'sim': i,
            'parameter': 'a',
            'simulated': float(a_true[i, 0]),
            'estimations': float(est['a'])
        })
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_sim = np.random.normal(0, 1, size=(nsim_test, 1))
recovery_results = param_recovery(a_sim, nsim=nsim_test)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
