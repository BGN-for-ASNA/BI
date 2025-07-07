from jax import jit
import tensorflow_probability 
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re
from BI.Samplers.Model_handler import model_handler 

class mcmc(model_handler):
    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)

    def convert_to_tensor(self, model, vars):
        self.tensor = JointDistributionCoroutine(lambda: model(**vars))
        return self.tensor

    def init_Model(self, model,  init = None, bijectors = None, seed=0):
        init_key, key = jax.random.split(jax.random.PRNGKey(int(seed)))
        init_key = jnp.array(init_key)
        self.init_key = init_key
        self.get_model_args(model)
        self.get_model_var(model)
        self.get_model_distributions(model)
        self.tensor = self.convert_to_tensor(model, self.vars) 

        if init is None:
            init_params = self.tensor.sample(seed = init_key)
            self.init_params = list(init_params)[:-1]
        else:
            self.init_params = init

        if bijectors is None:
             _, self.bijectors = self.initialise(self.model_info, self.init_params, self.obs_name)
        else:
            self.bijectors = bijectors
        #names = self.model_info.keys()

    @staticmethod
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio
        )

    def NUTS(self, model, obs, n_chains = 1, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, seed=0, name=None):        

        @jit
        def run_chain(key):
            inner_kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                step_size= 1e-3
            )

            kernel = tensorflow_probability.substrates.jax.mcmc.TransformedTransitionKernel(
                    inner_kernel=inner_kernel,
                    bijector=self.bijectors
            )

            hmc  = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel,
                target_accept_prob=.8,
                num_adaptation_steps=int(0.8*500),
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                      inner_results=pkr.inner_results._replace(step_size=new_step_size)
                  ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

            return tfp.mcmc.sample_chain(num_results = num_results,
                                         num_steps_between_results = num_steps_between_results,
                                         current_state= self.init_params,
                                         kernel=hmc,
                                         trace_fn=self.trace_fn,
                                         num_burnin_steps=num_burnin_steps,
                                         parallel_iterations = parallel_iterations,
                                         seed=key)

        Ndevices = jax.local_device_count(backend=None)
        if(n_chains > Ndevices):
            runs = jnp.ceil(n_chains/Ndevices)
            result = []
            for run in range(int(runs)):
                rng_keys = jax.random.split( jax.random.PRNGKey(0), Ndevices)
                result.append(jax.pmap(run_chain)(rng_keys))
            return result
        else:

            rng_keys = jax.random.split(jax.random.PRNGKey(0), n_chains)
            result =  jax.pmap(run_chain)(rng_keys)

            self.posterior, self.sample_stats = result[0], result[1]
            

    def run(self, model,  obs, n_chains = 1, init = None, bijectors = None, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, seed=0, name=None):
        self.obs_name = obs
        obs = self.data_on_model[obs]
        

        self.init_Model(model, init = init, bijectors = bijectors, seed=seed)

        if target_log_prob_fn == None:
            def target_log_prob(*params):
                return self.tensor.log_prob(params + (obs,))
        else:
            def target_log_prob(*params):
                return self.target_log_prob_fn(params + (obs,))

        self.sampler = self.NUTS(model = self.tensor, obs = obs,  n_chains = n_chains,  target_log_prob_fn = target_log_prob,
        num_results = num_results, num_burnin_steps=num_burnin_steps, num_steps_between_results=num_steps_between_results,
        parallel_iterations = parallel_iterations, seed=seed,name=name)

        return self.sampler

