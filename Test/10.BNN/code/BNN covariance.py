#%%
import matplotlib.pyplot as plt
from jax import random, vmap
from BayesForge import bf
import jax.numpy as jnp
import numpy as np
import jax
m = bf(platform='cpu')

#%%
import numpyro.distributions as dd
import pandas as pd

# setup platform------------------------------------------------

a = 3.5  # average morning wait time
b = -1  # average difference afternoon wait time
sigma_a = 1  # std dev in intercepts
sigma_b = 0.5  # std dev in slopes
rho = -0.7  # correlation between intercepts and slopes
Mu = jnp.array([a, b])
cov_ab = sigma_a * sigma_b * rho
Sigma = jnp.array([[sigma_a**2, cov_ab], [cov_ab, sigma_b**2]])
jnp.array([1, 2, 3, 4]).reshape(2, 2).T
sigmas = jnp.array([sigma_a, sigma_b])  # standard deviations
Rho = jnp.array([[1, rho], [rho, 1]])  # correlation matrix
print("original Rho")
print(Rho)
# now matrix multiply to get covariance matrix
Sigma = jnp.diag(sigmas) @ Rho @ jnp.diag(sigmas)

N_cafes = 20
seed = random.PRNGKey(5)  # used to replicate example
vary_effects = m.dist.multivariate_normal(Mu, Sigma, shape=(N_cafes,), sample = True)
a_cafe = vary_effects[:, 0]
b_cafe = vary_effects[:, 1]

seed = random.PRNGKey(22)
N_visits = 10
afternoon = jnp.tile(jnp.arange(2), N_visits * N_cafes // 2)
cafe_id = jnp.repeat(jnp.arange(N_cafes), N_visits)
mu = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
sigma = 0.5  # std dev within cafes
wait = m.dist.normal(mu, sigma, sample = True)
d = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))
#d.to_csv('../resources/data/Sim data multivariatenormal.csv', index=False)

# %%
# import data ------------------------------------------------
m = bf()
m.df = d

m.data_on_model = dict(
    cafe = jnp.array(m.df.cafe.values, dtype=jnp.int32),
    wait = jnp.array(m.df.wait.values, dtype=jnp.float32),
    N_cafes = len(m.df.cafe.unique()),
    afternoon = jnp.array(m.df.afternoon.values, dtype=jnp.float32)
)

def model(cafe, wait, N_cafes, afternoon):
    a = m.dist.normal(5, 2,  name = 'a')
    b = m.dist.normal(-1, 0.5, name = 'b')
    sigma = m.dist.exponential( 1,  name = 'sigma')

    sigma_cafe = m.dist.exponential(1, shape=(2,),  name = 'sigma_cafe')    
    Rho = m.dist.lkj(2, 2, name = 'Rho')
    cov = jnp.outer(sigma_cafe, sigma_cafe) * Rho
    a_cafe_b_cafe = m.dist.multivariate_normal(jnp.stack([a, b]), cov, shape = [N_cafes], name = 'a_b_cafe')    

    a_cafe, b_cafe = a_cafe_b_cafe[:, 0], a_cafe_b_cafe[:, 1]
    mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
    m.dist.normal(mu, sigma, obs=wait)

# Run sampler ------------------------------------------------
m.fit(model) 

#%%
# import data ------------------------------------------------
m2 = bf()
m2.df = d

m2.data_on_model = dict(
    cafe = jnp.array(m.df.cafe.values, dtype=jnp.int32),
    wait = jnp.array(m.df.wait.values, dtype=jnp.float32),
    N_cafes = len(m.df.cafe.unique()),
    afternoon = jnp.array(m.df.afternoon.values, dtype=jnp.float32)
)

def model(cafe, wait, N_cafes, afternoon):
    # Global parameters for intercept and slope
    a = m2.dist.normal(5, 2, name='a')
    b = m2.dist.normal(-1, 0.5, name='b')
    sigma = m2.dist.exponential(1, name='sigma')
    
    a_cafe_b_cafe = m.bnn.cov(10,N_cafes,a,b)
    
    # Split the network output into cafe-specific intercept and slope
    a_cafe, b_cafe = a_cafe_b_cafe[:, 0], a_cafe_b_cafe[:, 1]
    mu = a_cafe[cafe] + b_cafe[cafe] * afternoon
    
    m2.dist.normal(mu, sigma, obs=wait)

# Run sampler ------------------------------------------------
m2.fit(model)

# %%
plt.scatter(m.posteriors['a_b_cafe'],m2.posteriors['rf'])
plt.xlabel("Standard Multi-level Model Posteriors")
plt.ylabel("BNN Model Posteriors")
plt.title("Posterior Estimates Comparison: Standard vs BNN")
# save plot
plt.savefig('results/BNN_cov.png')

## Recovering RHo of BNN
print("BNN rho : \n", m.bnn.get_rho(m2.posteriors['rf']))

print("original Rho: \n", Rho)