from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "8.Multinomial"

print(f'Running BI for {model_name}')
m = bi(platform='cpu', backend='tfp')

# 1. Data Simulation ----------------------------------------
N = 500
income = np.array([1, 2, 5])
score = 0.5 * income
p_true = jax.nn.softmax(score)

np.random.seed(1)
career = np.random.choice([0, 1, 2], size=N, p=np.array(p_true))
df = pd.DataFrame({'career': career})
unique_income = np.array([1, 2, 5])

m.df = df
m.data_on_model = {
    'career': jnp.array(df.career.values),
    'unique_income': jnp.array(unique_income).astype(jnp.int32)
}

def model_bi(career, unique_income):
    a = yield m.dist.normal(0, 1, shape=(2,))
    b = yield m.dist.half_normal(0.5, shape=(1,))
    # Career 3 is pivot (0)
    s1 = a[0] + b * unique_income[0]
    s2 = a[1] + b * unique_income[1]
    s3 = jnp.zeros(1)
    
    # We want to concatenate them. Since s1, s2 are scalars from a[i] + b*x, but concatenated we need them to be a vector.
    # a[0] is a scalar, b is shape (1,), unique_income[0] is scalar. So s1 is shape (1,)
    p = jax.nn.softmax(jnp.concatenate([s1, s2, s3]))
    yield m.dist.categorical(probs=p, obs=career)

print("Fitting BI model...")
m.fit(model_bi, num_samples=1000, num_warmup=1000)
print("BI Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    int K;
    array[N] int career;
    vector[K] career_income;
}
parameters {
    vector[K-1] a;
    real<lower=0> b;
}
model {
    vector[K] s;
    a ~ normal(0, 1);
    b ~ normal(0, 0.5);
    s[1] = a[1] + b * career_income[1];
    s[2] = a[2] + b * career_income[2];
    s[3] = 0;
    career ~ categorical_logit(s);
}
"""
data_stan = {
    'N': N,
    'K': 3,
    'career': (df.career.values + 1).astype(int).tolist(),
    'career_income': unique_income.astype(float).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a_1': 'a[1]',
    'a_2': 'a[2]',
    'b': 'b'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(unique_income, a_true, b_true):
    # Ensure scalars for numpy array construction
    s1 = float(a_true[0] + b_true * unique_income[0])
    s2 = float(a_true[1] + b_true * unique_income[1])
    s3 = 0.0
    p = jax.nn.softmax(np.array([s1, s2, s3]))
    career_sim = np.random.choice([0,1,2], size=500, p=p)
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'career': jnp.array(career_sim),
        'unique_income': jnp.array(unique_income).astype(jnp.int32)
    }
    def model_rec(career, unique_income):
        a = yield m_rec.dist.normal(0, 1, shape=(2,))
        b = yield m_rec.dist.half_normal(0.5, shape=(1,))
        s1 = a[0] + b * unique_income[0]
        s2 = a[1] + b * unique_income[1]
        s3 = jnp.zeros(1)
        p = jax.nn.softmax(jnp.concatenate([s1, s2, s3]))
        yield m_rec.dist.categorical(probs=p, obs=career)
        
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(unique_income, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(unique_income, a_sims[i], b_sims[i])
        results.append({'sim': i, 'parameter': 'a[0]', 'simulated': a_sims[i,0], 'estimations': est['a[0]']})
        results.append({'sim': i, 'parameter': 'a[1]', 'simulated': a_sims[i,1], 'estimations': est['a[1]']})
        results.append({'sim': i, 'parameter': 'b', 'simulated': b_sims[i,0], 'estimations': est['b']})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = 100 
a_sims = np.random.normal(0, 1, size=(nsim_test, 2))
b_sims = np.abs(np.random.normal(0, 0.5, size=(nsim_test, 1)))

recovery_results = param_recovery(unique_income, a_sims, b_sims, nsim=nsim_test)
