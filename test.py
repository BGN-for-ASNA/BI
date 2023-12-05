#%%
import flet as ft
from gui.gui import main
from bayesian.distribution import *
ft.app(target=main)






#%%
import tensorflow as tf
import numpy as np
num_chains = 4
init_state = list(named_model.sample(num_chains))[:-1]
observed_data=(d2.age.values)
target_log_prob_fn = lambda *x: named_model.log_prob(x + named_model.sample(x = observed_data))

#%%
inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn = target_log_prob_fn, # function that returns its (possibly unnormalized) log-density under the target distribution
    step_size=0.1,
    num_leapfrog_steps=3
)
#%%
num_results = int(10e3) #number of hmc iterations
n_burnin = int(5e3)     #number of burn-in steps
step_size = 0.01
num_leapfrog_steps = 10
tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=n_burnin,
    kernel=inner_kernel,
    current_state= init_state)

#%%
kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=inner_kernel,
    target_accept_prob = 0.8,
    num_adaptation_steps = 500
)


# %%
named_model.log_prob(named_model.sample(x = observed_data))

# %%
target_log_prob_fn(np.array([2.,]))



# %%
