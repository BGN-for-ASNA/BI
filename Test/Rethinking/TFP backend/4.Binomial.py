from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "4.Binomial"

print(f'Running BI for {model_name}')
m = bi(platform='cpu', backend='tfp')

# 1. Data Preparation ----------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BI", "Resources")) + os.sep
df = pd.read_csv(data_path + 'chimpanzees.csv', sep=';')
m.df = df
m.data_on_model = {'pulled_left': jnp.array(df.pulled_left.values)}

def model_bi(pulled_left):
    alpha = yield m.dist.normal(0, 10, shape=(1,))
    yield m.dist.binomial(total_count=1, logits=alpha, obs=pulled_left)

print("Fitting BI model...")
m.fit(model_bi, num_samples=1000, num_warmup=1000)
print("BI Summary:")
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
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {"pulled_left": pulled_left_sim}
    def model_rec(pulled_left):
        alpha = yield m_rec.dist.normal(0, 10, shape=(1,))
        yield m_rec.dist.binomial(total_count=1, logits=alpha, obs=pulled_left)
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(a_true, nsim):
    results = []
    for i in range(nsim):
        est = estimate(a_true[i])
        results.append({
            'sim': i,
            'parameter': 'a',
            'simulated': a_true[i,0],
            'estimations': est['a']
        })
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BI_NSIM', 100))
a_sim = np.random.normal(0, 1, size=(nsim_test, 1))
recovery_results = param_recovery(a_sim, nsim=nsim_test)
