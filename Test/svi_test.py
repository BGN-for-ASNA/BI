#%%
from BI.Main.main import bi
import os

m = bi(platform='cpu', rand_seed = 0)

# Simulated data ---------------------------------------------
x = m.dist.normal(0, 1, shape = (100,), sample = True)
a = m.dist.normal(0, 1, sample = True)
b = m.dist.log_normal(0, 1, sample = True)
s = m.dist.exponential(1, sample = True)
y = m.dist.normal(a + b * x , s, sample = True)
m.data_on_model = dict(x = x, y = y)

# Model ------------------------------------------------
def model(x, y):    
    alpha = m.dist.normal( 0, 1, name = 'a')
    beta = m.dist.log_normal( 0, 1, name = 'b')   
    sigma = m.dist.exponential( 1, name = 's')
    m.dist.normal(alpha + beta * x , sigma, obs=y)

################################################################
# MCMC ------------------------------------------------
################################################################
m.fit(model, num_samples=1000) 
m.summary()

#%%
################################################################
# SVI Build in ------------------------------------------------
################################################################
m.svi(model, num_samples=1000, guide='multivariate') 

m.summary()

#%%
################################################################
# SVI  ------------------------------------------------
################################################################

from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
import jax.random as random
rng_key = random.PRNGKey(0)

loss = Trace_ELBO(num_particles=10)
optimizer = Adam(step_size=0.05) 
guide = AutoMultivariateNormal(model)
svi = SVI(model=model, guide=guide, optim=optimizer, loss=loss)

svi_result = svi.run(rng_key, num_steps=10000, x=x, y=y)


# Extract Posterior Samples
# Take the learned parameters from the result
params = svi_result.params

# Sample from the posterior distribution using the guide and the learned params
posterior_samples = guide.sample_posterior(
    rng_key, 
    params, 
    sample_shape=(1000,)  # Request 1000 samples
)
print(posterior_samples['a'].mean()) # -0.76
print(posterior_samples['b'].mean()) # 0.66 
print(posterior_samples['s'].mean()) # 0.82

