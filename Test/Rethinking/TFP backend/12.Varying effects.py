from Utils import *
from BI import bi
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax

model_name = "12.Varying effects"

print(f'Running BI for {model_name}')
# 1. Data Simulation (Varying Effects - Cafes)
a = 3.5  
b = -1.0  
sigma_a = 1.0  
sigma_b = 0.5  
rho = -0.7  
Mu = jnp.array([a, b])
sigmas = jnp.array([sigma_a, sigma_b])
Rho_sim = jnp.array([[1, rho], [rho, 1]])
Sigma = jnp.diag(sigmas) @ Rho_sim @ jnp.diag(sigmas)

N_cafes = 20
m_sim = bi(platform='cpu')
vary_effects = m_sim.dist.multivariate_normal(Mu, Sigma, shape=(N_cafes,), sample=True)
a_cafe_sim = vary_effects[:, 0]
b_cafe_sim = vary_effects[:, 1]

N_visits = 30 
afternoon = jnp.tile(jnp.arange(2), N_visits * N_cafes // 2)
cafe_id = jnp.repeat(jnp.arange(N_cafes), N_visits)
mu_sim = a_cafe_sim[cafe_id] + b_cafe_sim[cafe_id] * afternoon
sigma_sim = 0.5
wait = m_sim.dist.normal(mu_sim, sigma_sim, sample=True)
d = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))

# 2. BI Model (Non-Centered Parametrization)
m = bi(platform='cpu', backend='tfp')
m.data_on_model = dict(
    cafe=jnp.array(d.cafe.values, dtype=jnp.int32),
    wait=jnp.array(d.wait.values, dtype=jnp.float32),
    N_cafes=N_cafes,
    afternoon=jnp.array(d.afternoon.values, dtype=jnp.float32)
)

def model(cafe, wait, N_cafes, afternoon):
    a = yield m.dist.normal(5.0, 2.0, name='a')
    b = yield m.dist.normal(-1.0, 0.5, name='b')
    sigma = yield m.dist.exponential(1.0, name='sigma')
    sigma_cafe = yield m.dist.exponential(1.0, shape=(2,), name='sigma_cafe')
    
    # Manual 2D correlation via tanh transformation to ensure stability
    rho_unconstrained = yield m.dist.normal(0, 1, name='rho_unconstrained')
    rho = jnp.tanh(rho_unconstrained)
    
    # Construct Cholesky factor of correlation matrix
    L_Rho = jnp.array([[1.0, 0.0], [rho, jnp.sqrt(1.0 - rho**2)]])
    
    # Non-centered Varying Effects
    z = yield m.dist.normal(0, 1, shape=(N_cafes, 2), name='z')
    vary_effects = jnp.stack([a, b]) + (jnp.diag(sigma_cafe) @ L_Rho @ z.T).T
    
    a_cafe, b_cafe = vary_effects[:, 0], vary_effects[:, 1]
    mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
    yield m.dist.normal(mu, sigma, obs=wait)

print("Fitting BI model...")
m.fit(model, num_samples=1000, num_warmup=1000)
bi_summary = m.summary()
print("BI Summary Index:")
print(bi_summary.index.tolist())
print("BI Summary:")
print(bi_summary)

# 3. Stan Model (Non-Centered Parametrization)
stan_code = """
data{
    int len;
    int N_cafes;
    vector[len] wait;
    vector[len] afternoon;
    array[len] int cafe;
}
parameters{
    cholesky_factor_corr[2] L_Rho;
    real a;
    real b;
    real<lower=0> sigma;
    vector<lower=0>[2] sigma_cafe;
    matrix[2, N_cafes] z;
}
transformed parameters {
    matrix[N_cafes, 2] vary_effects;
    vector[N_cafes] a_cafe;
    vector[N_cafes] b_cafe;
    vary_effects = (rep_matrix([a, b]', N_cafes) + diag_pre_multiply(sigma_cafe, L_Rho) * z)';
    a_cafe = vary_effects[, 1];
    b_cafe = vary_effects[, 2];
}
model{
    to_vector(z) ~ std_normal();
    L_Rho ~ lkj_corr_cholesky( 2 );
    sigma ~ exponential( 1 );
    sigma_cafe ~ exponential( 1 );
    b ~ normal( -1 , 0.5 );
    a ~ normal( 5 , 2 );
    
    vector[len] mu;
    for ( i in 1:len ) {
        mu[i] = a_cafe[cafe[i]] + b_cafe[cafe[i]] * afternoon[i];
    }
    wait ~ normal( mu , sigma );
}
generated quantities {
    matrix[2, 2] Rho;
    Rho = multiply_lower_tri_self_transpose(L_Rho);
}
"""
stan_data = {
    'wait': d['wait'].values.astype(float),
    'afternoon': d['afternoon'].values.astype(float),
    'cafe': d['cafe'].values.astype(int) + 1,
    'N_cafes': N_cafes,
    'len': len(d)
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=stan_data, chains=4)

# 4. Comparison
param_map = {
    'a': 'a',
    'b': 'b',
    'sigma': 'sigma',
    'sigma_cafe_0': 'sigma_cafe[1]',
    'sigma_cafe_1': 'sigma_cafe[2]',
    'rho': 'Rho[2,1]'
}

bi_df = prepare_bi_data(m)
# Map rho from rho_unconstrained
bi_df['rho'] = np.tanh(bi_df['rho_unconstrained'])

bi_df = bi_df.rename(columns={
    'sigma_cafe[0]': 'sigma_cafe_0',
    'sigma_cafe[1]': 'sigma_cafe_1'
})

plot_comparaison(bi_df, df_stan, param_map, model_name=model_name)

# 5. Parameter Recovery Analysis
def estimate_rec(a_true, b_true, sigma_cafe_true, rho_true, sigma_true):
    N_cafes_rec = 50
    N_visits_rec = 40
    
    Mu = jnp.array([a_true, b_true])
    Sig = jnp.diag(sigmas)
    Cov = Sig @ Rho_sim @ Sig
    Cov = Cov + jnp.eye(2) * 1e-6
    
    vary_effects = np.random.multivariate_normal(Mu, Cov, size=N_cafes_rec)
    a_cafe, b_cafe = vary_effects[:, 0], vary_effects[:, 1]
    
    cafe_id = np.repeat(np.arange(N_cafes_rec), N_visits_rec)
    afternoon = np.tile(np.arange(2), N_visits_rec * N_cafes_rec // 2)
    
    mu = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
    wait_sim = np.random.normal(mu, sigma_true).astype(float)
    
    m_rec = bi(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'cafe': jnp.array(cafe_id, dtype=jnp.int32),
        'wait': jnp.array(wait_sim, dtype=jnp.float32),
        'N_cafes': N_cafes_rec,
        'afternoon': jnp.array(afternoon, dtype=jnp.float32)
    }
    def model_rec(cafe, wait, N_cafes, afternoon):
        a = yield m_rec.dist.normal(5.0, 2.0, name='a')
        b = yield m_rec.dist.normal(-1.0, 0.5, name='b')
        sigma = yield m_rec.dist.exponential(1.0, name='sigma')
        sigma_cafe = yield m_rec.dist.exponential(1.0, shape=(2,), name='sigma_cafe')
        rho_unconstrained = yield m_rec.dist.normal(0, 1, name='rho_unconstrained')
        rho = jnp.tanh(rho_unconstrained)
        L_Rho = jnp.array([[1.0, 0.0], [rho, jnp.sqrt(1.0 - rho**2)]])
        z = yield m_rec.dist.normal(0, 1, shape=(N_cafes, 2), name='z')
        vary_effects = jnp.stack([a, b]) + (jnp.diag(sigma_cafe) @ L_Rho @ z.T).T
        a_cafe, b_cafe = vary_effects[:, 0], vary_effects[:, 1]
        mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
        yield m_rec.dist.normal(mu, sigma, obs=wait)
        
    m_rec.fit(model_rec, num_samples=1000, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

print("\nRunning Parameter Recovery...")
nsim = int(os.getenv('BI_NSIM', 20))
results = []
for i in range(nsim):
    a_sim = np.random.normal(3.5, 0.5)
    b_sim = np.random.normal(-1.0, 0.5)
    sigma_cafe_sim = np.random.exponential(1.0, size=2)
    rho_sim = np.random.uniform(-0.9, 0.9)
    sigma_sim = 0.5
    
    est = estimate_rec(a_sim, b_sim, sigma_cafe_sim, rho_sim, sigma_sim)
    results.append({'sim': i, 'parameter': 'a', 'simulated': float(a_sim), 'estimations': float(est['a'])})
    results.append({'sim': i, 'parameter': 'b', 'simulated': float(b_sim), 'estimations': float(est['b'])})
            
df_res = pd.DataFrame(results)
plot_recovery(df_res, model_name=model_name)
