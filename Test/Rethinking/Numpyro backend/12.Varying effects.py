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
a = 3.5  # average morning wait time
b = -1.0  # average difference afternoon wait time
sigma_a = 1.0  # std dev in intercepts
sigma_b = 0.5  # std dev in slopes
rho = -0.7  # correlation between intercepts and slopes
Mu = jnp.array([a, b])
sigmas = jnp.array([sigma_a, sigma_b])
Rho_sim = jnp.array([[1, rho], [rho, 1]])
Sigma = jnp.diag(sigmas) @ Rho_sim @ jnp.diag(sigmas)

N_cafes = 20
m_sim = bi(platform='cpu')
vary_effects = m_sim.dist.multivariate_normal(Mu, Sigma, shape=(N_cafes,), sample=True)
a_cafe_sim = vary_effects[:, 0]
b_cafe_sim = vary_effects[:, 1]

N_visits = 10
afternoon = jnp.tile(jnp.arange(2), N_visits * N_cafes // 2)
cafe_id = jnp.repeat(jnp.arange(N_cafes), N_visits)
mu_sim = a_cafe_sim[cafe_id] + b_cafe_sim[cafe_id] * afternoon
sigma_sim = 0.5
wait = m_sim.dist.normal(mu_sim, sigma_sim, sample=True)
d = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))

# 2. BI Model
m = bi(platform='cpu')
m.data_on_model = dict(
    cafe=jnp.array(d.cafe.values, dtype=jnp.int32),
    wait=jnp.array(d.wait.values, dtype=jnp.float32),
    N_cafes=N_cafes,
    afternoon=jnp.array(d.afternoon.values, dtype=jnp.float32)
)

def model(cafe, wait, N_cafes, afternoon):
    alpha = m.dist.normal(5.0, 2.0, name='a')
    beta = m.dist.normal(-1.0, 0.5, name='b')
    sigma = m.dist.exponential(1.0, name='sigma')
    sigma_cafe = m.dist.exponential(1.0, shape=(2,), name='sigma_cafe')
    Rho = m.dist.lkj(2, 2.0, name='Rho')
    
    cov = jnp.outer(sigma_cafe, sigma_cafe) * Rho
    a_b_cafe = m.dist.multivariate_normal(jnp.stack([alpha, beta]), cov, shape=(N_cafes,), name='a_b_cafe')
    
    a_cafe, b_cafe = a_b_cafe[:, 0], a_b_cafe[:, 1]
    mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
    m.dist.normal(mu, sigma, obs=wait)

m.fit(model, num_samples=1000, num_warmup=1000)
bi_summary = m.summary()
print("BI Summary:")
print(bi_summary)

# 3. Stan Model
stan_code = """
data{
    int len;
    int N_cafes;
    vector[len] wait;
    array[len] int afternoon;
    array[len] int cafe;
}
parameters{
    corr_matrix[2] Rho;
    real a;
    vector[N_cafes] a_cafe;
    real b;
    vector[N_cafes] b_cafe;      
    real<lower=0> sigma;
    vector<lower=0>[2] sigma_cafe;   
}
model{
    vector[len] mu;
    Rho ~ lkj_corr( 2 );
    sigma ~ exponential( 1 );
    sigma_cafe ~ exponential( 1 );
    b ~ normal( -1 , 0.5 );    
    a ~ normal( 5 , 2 );
    {
        array[N_cafes] vector[2] YY;
        vector[2] MU;
        MU = [ a , b ]';
        for ( j in 1:N_cafes ) YY[j] = [ a_cafe[j] , b_cafe[j] ]';
        YY ~ multi_normal( MU , quad_form_diag(Rho , sigma_cafe) );
    }
    for ( i in 1:len ) {
        mu[i] = a_cafe[cafe[i]] + b_cafe[cafe[i]] * afternoon[i];        
    }
    wait ~ normal( mu , sigma );
}
"""
stan_data = {
    'wait': d['wait'].values.astype(float),
    'afternoon': d['afternoon'].values.astype(int),
    'cafe': d['cafe'].values.astype(int) + 1,
    'N_cafes': N_cafes,
    'len': len(d)
}
stan_df = build_stan_model(stan_code, data=stan_data, chains=4)

# 4. Comparison
param_map = {
    'a': 'a',
    'b': 'b',
    'sigma': 'sigma',
    'sigma_cafe_0': 'sigma_cafe[1]',
    'sigma_cafe_1': 'sigma_cafe[2]',
    'Rho_0_1': 'Rho[2,1]'
}

bi_samples = m.posteriors
bi_df = pd.DataFrame({
    'a': bi_samples['a'],
    'b': bi_samples['b'],
    'sigma': bi_samples['sigma'],
    'sigma_cafe_0': bi_samples['sigma_cafe'][:, 0],
    'sigma_cafe_1': bi_samples['sigma_cafe'][:, 1],
    'Rho_0_1': bi_samples['Rho'][:, 0, 1]
})

plot_comparaison(bi_df, stan_df, param_map, model_name=model_name)

# 5. Parameter Recovery Analysis
def estimate_rec(afternoon, cafe_id, a_true, b_true, sigma_cafe_true, rho_true, sigma_true):
    Mu = jnp.array([a_true, b_true])
    sigmas = jnp.array([sigma_cafe_true[0], sigma_cafe_true[1]])
    Rho_sim = jnp.array([[1, rho_true], [rho_true, 1]])
    Cov = jnp.diag(sigmas) @ Rho_sim @ jnp.diag(sigmas)
    
    vary_effects = np.random.multivariate_normal(Mu, Cov, size=20)
    a_cafe, b_cafe = vary_effects[:, 0], vary_effects[:, 1]
    mu = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
    wait_sim = np.random.normal(mu, sigma_true)
    
    m_rec = bi(print_devices_found=False)
    m_rec.data_on_model = {
        'cafe': jnp.array(cafe_id, dtype=jnp.int32),
        'wait': jnp.array(wait_sim, dtype=jnp.float32),
        'N_cafes': 20,
        'afternoon': jnp.array(afternoon, dtype=jnp.float32)
    }
    def model_rec(cafe, wait, N_cafes, afternoon):
        alpha = m_rec.dist.normal(5.0, 2.0, name='a')
        beta = m_rec.dist.normal(-1.0, 0.5, name='b')
        sigma = m_rec.dist.exponential(1.0, name='sigma')
        sigma_cafe = m_rec.dist.exponential(1.0, shape=(2,), name='sigma_cafe')
        Rho = m_rec.dist.lkj(2, 2.0, name='Rho')
        cov = jnp.outer(sigma_cafe, sigma_cafe) * Rho
        a_b_cafe = m_rec.dist.multivariate_normal(jnp.stack([alpha, beta]), cov, shape=(N_cafes,), name='a_b_cafe')
        a_cafe, b_cafe = a_b_cafe[:, 0], a_b_cafe[:, 1]
        mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
        m_rec.dist.normal(mu, sigma, obs=wait)
        
    m_rec.fit(model_rec, num_samples=500, progress_bar=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

print("\nRunning Parameter Recovery (100 simulations)...")
nsim = int(os.environ.get("BI_NSIM", 100))
results = []
for i in range(nsim):
    a_sim = np.random.normal(3.5, 0.5)
    b_sim = np.random.normal(-1.0, 0.2)
    sigma_cafe_sim = np.random.exponential(1.0, size=2)
    rho_sim = np.random.uniform(-0.9, 0.9)
    sigma_sim = 0.5
    
    est = estimate_rec(d.afternoon.values, d.cafe.values, a_sim, b_sim, sigma_cafe_sim, rho_sim, sigma_sim)
    results.append({'sim': i, 'parameter': 'a', 'simulated': a_sim, 'estimations': est['a']})
    results.append({'sim': i, 'parameter': 'b', 'simulated': b_sim, 'estimations': est['b']})
    # Rho[0,1]
    results.append({'sim': i, 'parameter': 'rho', 'simulated': rho_sim, 'estimations': est['Rho[0, 1]']})

df_res = pd.DataFrame(results)
plot_recovery(df_res, model_name=model_name)
