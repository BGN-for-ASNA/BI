# %%
# https://num.pyro.ai/en/0.13.1/tutorials/discrete_imputation.html
import numbers
import numpy as np
import funsor.ops as ops

# Patch Funsor's clamp for scalar numbers
ops.clamp.register((numbers.Number, np.number))(lambda x, min=None, max=None: np.clip(x, min, max))

from BayesForge import bf
import jax.numpy as jnp
import numpyro
import numpyro
from jax import numpy as jnp, random, ops
from jax.scipy.special import expit
from numpyro import distributions as dist, sample
from numpyro.infer.mcmc import MCMC
from numpyro.infer.hmc import NUTS
from math import inf
from graphviz import Digraph

simkeys = random.split(random.PRNGKey(0), 10)
nsim = 5000
mcmc_key = random.PRNGKey(1)

m = bf()

# %%
# Using numpyro reproductibility
b_A = 0.25
b_B = 0.25
s_Y = 0.25
Z = random.normal(simkeys[0], (nsim,))
A = random.bernoulli(simkeys[1], expit(Z))
B = random.bernoulli(simkeys[2], expit(Z))
Y = A * b_A + B * b_B + s_Y * random.normal(simkeys[3], (nsim,))

# %%
# Missing at random, conditional on observed data (MAR)
A_obs = random.bernoulli(simkeys[4], expit(3 * (Y - Y.mean())))
Aobs = jnp.where(A_obs, A, -1)
A_obsidx = jnp.where(A_obs)

# generate complete case arrays
Acc = Aobs[A_obsidx]
Bcc = B[A_obsidx]
Ycc = Y[A_obsidx]
# %%
# Model
def model(A, B, Y):
    ntotal = A.shape[0]
    A_isobs = A >= 0
    mu_A = m.dist.normal(0, 2.5)
    b_B_A = m.dist.normal(0, 2.5)
    b_A = m.dist.normal(0, 2.5)
    b_B = m.dist.normal(0, 2.5)
    s_Y = m.dist.exponential(1.0)
    
    with m.dist.plate("obs", ntotal):
        eta_A = mu_A + b_B_A * B
        dist_A = m.dist.bernoulli(logits=eta_A, create_obj=True)
        # 1. Likelihood when A is observed
        A_valid = jnp.maximum(A, 0)
        ll_obs = dist_A.log_prob(A_valid) + m.dist.normal(b_A * A_valid + b_B * B, s_Y, create_obj=True).log_prob(Y)
        # 2. Marginal likelihood when A is missing (sum over A=0 and A=1)
        ll_0 = dist_A.log_prob(0) + m.dist.normal(b_B * B, s_Y, create_obj=True).log_prob(Y)
        ll_1 = dist_A.log_prob(1) + m.dist.normal(b_A + b_B * B, s_Y, create_obj=True).log_prob(Y)
        ll_mis = jnp.logaddexp(ll_0, ll_1)
        # Apply log-likelihood based on whether A is observed or missing
        numpyro.factor("log_lik", jnp.where(A_isobs, ll_obs, ll_mis))
m.data_on_model = {'A': Aobs, 'B': B, 'Y': Y}
m.fit(model)
m.summary()

# %%
# Model without plate (scalar log-likelihood via jnp.sum)
def model_no_plate(A, B, Y):
    A_isobs = A >= 0
    mu_A = m.dist.normal(0, 2.5)
    b_B_A = m.dist.normal(0, 2.5)
    b_A = m.dist.normal(0, 2.5)
    b_B = m.dist.normal(0, 2.5)
    s_Y = m.dist.exponential(1.0)
    
    eta_A = mu_A + b_B_A * B
    dist_A = m.dist.bernoulli(logits=eta_A, create_obj=True)
    
    # 1. Likelihood when A is observed
    A_valid = jnp.maximum(A, 0) # Avoid NaN by clamping -1 to 0.  
    ll_obs = dist_A.log_prob(A_valid) + m.dist.normal(b_A * A_valid + b_B * B, s_Y, create_obj=True).log_prob(Y)
    
    # 2. Marginal likelihood when A is missing (sum over A=0 and A=1)
    ll_0 = dist_A.log_prob(0) + m.dist.normal(b_B * B, s_Y, create_obj=True).log_prob(Y) # Complete case log-likelihood when A=0
    ll_1 = dist_A.log_prob(1) + m.dist.normal(b_A + b_B * B, s_Y, create_obj=True).log_prob(Y) # Complete case log-likelihood when A=1
    ll_mis = jnp.logaddexp(ll_0, ll_1) # Complete case marginal log-likelihood when A is missing
    
    # Sum over all observations into a single scalar factor (no plate needed)
    m.dist.factor("log_lik", jnp.sum(jnp.where(A_isobs, ll_obs, ll_mis)))

m.data_on_model = {'A': Aobs, 'B': B, 'Y': Y}
m.fit(model_no_plate)
m.summary()

# %%
# Missing Not At Random (MNAR): Missingness conditional on covariate A itself
A_obs_mnar = random.bernoulli(simkeys[5], 0.9 - 0.8 * A)
Aobs_mnar = jnp.where(A_obs_mnar, A, -1)

# %%
# Model with sampling/observation mechanism (MNAR with marginalization without plate)
def model_mnar_no_plate(A, B, Y):
    A_isobs = A >= 0
    mu_A = m.dist.normal(0, 2.5)
    b_B_A = m.dist.normal(0, 2.5)
    b_A = m.dist.normal(0, 2.5)
    b_B = m.dist.normal(0, 2.5)
    s_Y = m.dist.exponential(1.0)
    
    # Observation probabilities conditional on A
    p_Aobs0 = m.dist.beta(1.0, 1.0)  # P(observed | A=0)
    p_Aobs1 = m.dist.beta(1.0, 1.0)  # P(observed | A=1)
    
    eta_A = mu_A + b_B_A * B
    dist_A = m.dist.bernoulli(logits=eta_A, create_obj=True)
    
    # 1. Likelihood when A is observed (observation indicator = 1)
    A_valid = jnp.maximum(A, 0)
    prob_obs = jnp.where(A_valid == 1, p_Aobs1, p_Aobs0)
    ll_obs = (
        dist_A.log_prob(A_valid)
        + m.dist.normal(b_A * A_valid + b_B * B, s_Y, create_obj=True).log_prob(Y)
        + jnp.log(prob_obs)
    )
    
    # 2. Marginal likelihood when A is missing (observation indicator = 0, marginalize over A in {0, 1})
    ll_0 = (
        dist_A.log_prob(0)
        + m.dist.normal(b_B * B, s_Y, create_obj=True).log_prob(Y)
        + jnp.log(1.0 - p_Aobs0)
    )
    ll_1 = (
        dist_A.log_prob(1)
        + m.dist.normal(b_A + b_B * B, s_Y, create_obj=True).log_prob(Y)
        + jnp.log(1.0 - p_Aobs1)
    )
    ll_mis = jnp.logaddexp(ll_0, ll_1)
    
    # Total scalar factor across all data (no plate needed)
    m.dist.factor("log_lik", jnp.sum(jnp.where(A_isobs, ll_obs, ll_mis)))

m.data_on_model = {'A': Aobs_mnar, 'B': B, 'Y': Y}
m.fit(model_mnar_no_plate)
m.summary()

#%%
