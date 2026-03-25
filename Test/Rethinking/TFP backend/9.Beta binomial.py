from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "9.Beta binomial"

print(f'Running BI for {model_name}')
m = bi(platform='cpu', backend='tfp')

# 1. Data Loading -------------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BI", "Resources")) + os.sep
df = pd.read_csv(data_path + 'UCBadmit.csv', sep=';')
df["gid"] = (df["applicant.gender"] != "male").astype(int)

m.df = df
m.data_on_model = {
    'gid': jnp.array(df.gid.values),
    'applications': jnp.array(df.applications.values),
    'admit': jnp.array(df.admit.values)
}

def model_bi(gid, applications, admit):
    phi = yield m.dist.exponential(1.0, shape=(1,))
    alpha = yield m.dist.normal(0.0, 1.5, shape=(2,))
    theta = phi + 2.0
    pbar = jax.nn.sigmoid(alpha[gid])
    concentration1 = pbar * theta
    concentration0 = (1.0 - pbar) * theta
    yield m.dist.beta_binomial(total_count=applications, concentration1=concentration1, concentration0=concentration0, obs=admit)

print("Fitting BI model...")
m.fit(model_bi, num_samples=1000, num_warmup=1000)
print("BI Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    array[N] int admit;
    array[N] int applications;
    array[N] int gid;
}
parameters {
    vector[2] a;
    real<lower=0> phi;
}
transformed parameters {
    real theta;
    theta = phi + 2;
}
model {
    vector[N] pbar;
    phi ~ exponential(1);
    a ~ normal(0, 1.5);
    for (i in 1:N) {
        pbar[i] = inv_logit(a[gid[i]]);
    }
    admit ~ beta_binomial(applications, pbar * theta, (1 - pbar) * theta);
}
"""
data_stan = {
    'N': len(df),
    'admit': df.admit.values.astype(int).tolist(),
    'applications': df.applications.values.astype(int).tolist(),
    'gid': (df.gid.values + 1).astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
plot_comparaison(m, df_stan, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(gid, applications, a_true, phi_true):
    theta = phi_true + 2.0
    pbar = jax.nn.sigmoid(a_true[gid])
    admit_sim = np.random.binomial(applications, np.random.beta(pbar * theta, (1.0 - pbar) * theta))
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'gid': jnp.array(gid),
        'applications': jnp.array(applications),
        'admit': jnp.array(admit_sim)
    }
    def model_rec(gid, applications, admit):
        phi = yield m_rec.dist.exponential(1.0, shape=(1,))
        alpha = yield m_rec.dist.normal(0.0, 1.5, shape=(2,))
        theta = phi + 2.0
        pbar = jax.nn.sigmoid(alpha[gid])
        yield m_rec.dist.beta_binomial(total_count=applications, concentration1=pbar*theta, concentration0=(1-pbar)*theta, obs=admit)
        
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(gid, applications, a_sims, phi_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(gid, applications, a_sims[i], phi_sims[i])
        results.append({'sim': i, 'parameter': 'alpha[0]', 'simulated': a_sims[i,0], 'estimations': est['alpha[0]']})
        results.append({'sim': i, 'parameter': 'alpha[1]', 'simulated': a_sims[i,1], 'estimations': est['alpha[1]']})
        results.append({'sim': i, 'parameter': 'phi', 'simulated': phi_sims[i,0], 'estimations': est['phi']})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BI_NSIM', 100))
a_sims = np.random.normal(0, 1.5, size=(nsim_test, 2))
phi_sims = np.random.exponential(1.0, size=(nsim_test, 1))

recovery_results = param_recovery(df.gid.values, df.applications.values, a_sims, phi_sims, nsim=nsim_test)
