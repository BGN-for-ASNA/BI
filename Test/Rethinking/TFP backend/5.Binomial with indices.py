from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "5.Binomial with indices"

print(f'Running BF for {model_name}')
m = bf(platform='cpu', backend='tfp')

# 1. Data Preparation ----------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'chimpanzees.csv', sep=';')
df['prosoc_left'] = df.prosoc_left.values
df['condition'] = df.condition.values
df['actor'] = df.actor.values - 1 # 0-indexed

m.df = df
m.data_to_model(['prosoc_left', 'condition', 'actor', 'pulled_left'])

def model_BF(actor, prosoc_left, condition, pulled_left):
    a = yield m.dist.normal(0, 10, shape=(7,), name='a')
    bp = yield m.dist.normal(0, 10, name='bp')
    bpc = yield m.dist.normal(0, 10, name='bpc')
    logits = a[actor] + (bp + bpc * condition) * prosoc_left
    yield m.dist.binomial(total_count=1, logits=logits, obs=pulled_left)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=0> N;
    int<lower=0> N_actors;
    array[N] int pulled_left;
    array[N] int prosoc_left;
    array[N] int condition;
    array[N] int actor;
}
parameters {
    vector[N_actors] a;
    real bp;
    real bpc;
}
model {
    vector[N] p;
    a ~ normal(0, 10);
    bp ~ normal(0, 10);
    bpc ~ normal(0, 10);
    for (i in 1:N) {
        p[i] = a[actor[i]] + (bp + bpc * condition[i]) * prosoc_left[i];
    }
    pulled_left ~ binomial_logit(1, p);
}
"""
data_stan = {
    'N': len(df),
    'N_actors': 7,
    'pulled_left': df.pulled_left.values.astype(int).tolist(),
    'prosoc_left': df.prosoc_left.values.astype(int).tolist(),
    'condition': df.condition.values.astype(int).tolist(),
    'actor': (df.actor.values + 1).astype(int).tolist() # 1-indexed for Stan
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a[1]', 'a[1]': 'a[2]', 'a[2]': 'a[3]', 'a[3]': 'a[4]',
    'a[4]': 'a[5]', 'a[5]': 'a[6]', 'a[6]': 'a[7]',
    'bp[0]': 'bp', 'bpc[0]': 'bpc'
}
BF_df = prepare_bi_data(m)
plot_comparaison(BF_df, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(actor, prosoc_left, condition, a_true, bp_true, bpc_true):
    logits = a_true[actor] + (bp_true + bpc_true * condition) * prosoc_left
    pulled_left_sim = np.random.binomial(1, jax.nn.sigmoid(logits))
    
    m_rec = bf(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'actor': jnp.array(actor),
        'prosoc_left': jnp.array(prosoc_left),
        'condition': jnp.array(condition),
        'pulled_left': jnp.array(pulled_left_sim)
    }
    def model_rec(actor, prosoc_left, condition, pulled_left):
        a = yield m_rec.dist.normal(0, 10, shape=(7,), name='a')
        bp = yield m_rec.dist.normal(0, 10, name='bp')
        bpc = yield m_rec.dist.normal(0, 10, name='bpc')
        logits = a[actor] + (bp + bpc * condition) * prosoc_left
        yield m_rec.dist.binomial(total_count=1, logits=logits, obs=pulled_left)
        
    m_rec.fit(model_rec, num_samples=500, num_warmup=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(actor, prosoc_left, condition, a_sims, bp_sims, bpc_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(actor, prosoc_left, condition, a_sims[i], bp_sims[i], bpc_sims[i])
        for j in range(7):
            results.append({'sim': i, 'parameter': f'a[{j}]', 'simulated': float(a_sims[i,j]), 'estimations': float(est[f'a[{j}]'])})
        results.append({'sim': i, 'parameter': 'bp', 'simulated': float(bp_sims[i,0]), 'estimations': float(est['bp'])})
        results.append({'sim': i, 'parameter': 'bpc', 'simulated': float(bpc_sims[i,0]), 'estimations': float(est['bpc'])})
        
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BF_NSIM', 10))
a_sims = np.random.normal(0, 1, size=(nsim_test, 7))
bp_sims = np.random.normal(0, 1, size=(nsim_test, 1))
bpc_sims = np.random.normal(0, 1, size=(nsim_test, 1))

recovery_results = param_recovery(df.actor.values, df.prosoc_left.values, df.condition.values, a_sims, bp_sims, bpc_sims, nsim=nsim_test)


# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(actor, prosoc_left, condition, pulled_left):
    a = m_ref.dist.normal(0, 10, shape=(7,))
    bp = m_ref.dist.normal(0, 10)
    bpc = m_ref.dist.normal(0, 10)
    logits = a[actor] + (bp + bpc * condition) * prosoc_left
    m_ref.dist.binomial(total_count=1, logits=logits, obs=pulled_left)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'actor': m.data_on_model['actor'], 'prosoc_left': m.data_on_model['prosoc_left'], 'condition': m.data_on_model['condition'], 'pulled_left': m.data_on_model['pulled_left']})
