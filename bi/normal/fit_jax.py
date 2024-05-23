import tensorflow_probability as tfp
import tensorflow as tf
import arviz as az
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import JointDistributionNamedAutoBatched as JDNAB
tfd = tfp.distributions
import random as r
from functools import partial


class fit():
    def __init__(self, obs_names, observed_data_jax) :
        self.obs_names = obs_names
        self.observed_data_jax = observed_data_jax
        self.keys = self.tensor._flat_resolve_names()

    @partial(jit, static_argnums=(0,))
    def target_log_prob(self, *params):
        param_dict = {}
        for i, key in enumerate(self.keys):
            if key != self.obs_names:
                param_dict[key] = params[i]
        param_dict[self.obs_names] = self.observed_data_jax
        return self.tensor.log_prob(param_dict)

    
    @partial(jit, static_argnums=(0,))
    def run_chain(self, key):        
        init_key, sample_key = random.split(random.PRNGKey(0))
        init_params = self.tensor.sample(seed=jnp.array(init_key, dtype=jnp.uint32))
        init_params.pop(self.obs_names)
        init_params = list(init_params.values())   
        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn = self.target_log_prob, step_size = 1e-3)
        return tfp.mcmc.sample_chain(2000,
                                      current_state=init_params,
                                      parallel_iterations = 4,
                                      kernel=kernel,
                                      trace_fn=lambda _, results: results.target_log_prob,
                                      num_burnin_steps=500,
                                      seed=key)
    

    def parallele_chains(self, n_chain = int(4)):
        # Jax parallel
        rng_keys = jax.random.split(random.PRNGKey(0), n_chain)
        return jax.pmap(self.run_chain)(rng_keys)

    
    def tfp_trace_to_arviz(self, posterior, 
                           sample_stats,
                           var_names=None, 
                           sample_stats_name=['log_likelihood','tree_size','diverging','energy','mean_tree_accept']):
        sample_stats = {k:jnp.transpose(v) for k, v in zip(sample_stats_name, sample_stats)}
        trace = {}

        for name, samp in zip(var_names, posterior):
            if len(samp.shape) == 2:
                transposed_shape = [1, 0]

            elif len(samp.shape) == 3:
                transposed_shape = [1, 0, 2]

            else:
                transposed_shape = [1, 0, 2, 3]

            trace[name] = tf.transpose(samp, transposed_shape)

        trace = az.from_dict(posterior=trace, sample_stats=sample_stats)
        self.trace = trace

