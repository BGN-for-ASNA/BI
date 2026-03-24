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

    def init_Model(self, model,  init = None, bijectors = None, seed=0, num_obs=1):
        init_key, key = jax.random.split(jax.random.PRNGKey(int(seed)))
        init_key = jnp.array(init_key)
        self.init_key = init_key
        self.get_model_args(model)
        self.get_model_var(model)
        self.get_model_distributions(model)
        self.tensor = self.convert_to_tensor(model, self.vars) 

        if init is None:
            init_params = self.tensor.sample(seed = init_key)
            self.init_params = list(init_params)[:-num_obs]
        else:
            self.init_params = init

        if bijectors is None:
             _, self.bijectors = self.initialise(self.model_info, self.init_params, self.obs_names)
        else:
            self.bijectors = bijectors

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

        if isinstance(seed, (int, float)):
            master_key = jax.random.PRNGKey(int(seed))
        else:
            master_key = seed

        Ndevices = jax.local_device_count(backend=None)
        if(n_chains > Ndevices):
            runs = int(jnp.ceil(n_chains/Ndevices))
            run_keys = jax.random.split(master_key, runs)
            all_samples = []
            all_stats = []
            for run in range(runs):
                rng_keys = jax.random.split(run_keys[run], Ndevices)
                res, stats = jax.pmap(run_chain)(rng_keys)
                all_samples.append(res)
                all_stats.append(stats)
            
            # Stack across runs to get (n_chains, num_results, ...)
            num_params = len(all_samples[0])
            self.posterior = []
            for p in range(num_params):
                self.posterior.append(jnp.concatenate([run_res[p] for run_res in all_samples], axis=0))
            
            self.sample_stats = []
            num_stats = len(all_stats[0])
            for s in range(num_stats):
                self.sample_stats.append(jnp.concatenate([run_stats[s] for run_stats in all_stats], axis=0))

        else:
            rng_keys = jax.random.split(master_key, n_chains)
            self.posterior, self.sample_stats =  jax.pmap(run_chain)(rng_keys)

    def get_samples(self, group_by_chain=True):
        """
        Get posterior samples from the fitted model.
        Args:
            group_by_chain (bool): If True, returns samples with chain dimension. TFP backend currently always returns with chain dimension if num_chains > 1.
        Returns:
            dict: Dictionary of parameter names and their posterior samples.
        """
        var_names = list(self.model_info.keys())
        trace = {}
        for name, samp in zip(var_names, self.posterior):
            trace[name] = samp
        return trace

    def run(self, seed=0, model = None,  obs = None, n_chains = 1, init = None, bijectors = None, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, name=None, **kwargs):
        
        if model is None:
            model = self.model
        
        if kwargs:
            if self.data_on_model is None:
                self.data_on_model = {}
            self.data_on_model.update(kwargs)

        if obs is None:
            obs = self.find_obs_in_model(model)
        
        if isinstance(obs, str):
            obs = [obs]
        
        self.obs_names = obs
        
        # Verify all obs exist in data_on_model
        for name in obs:
            if name not in self.data_on_model:
                raise KeyError(f"Observed variable '{name}' not found in data_on_model. "
                               f"Available keys: {list(self.data_on_model.keys())}")
        
        obs_tuple = tuple(self.data_on_model[name] for name in obs)
        
        self.init_Model(model, init = init, bijectors = bijectors, seed=0, num_obs=len(obs)) 

        if target_log_prob_fn == None:
            def target_log_prob(*params):
                return self.tensor.log_prob(params + obs_tuple)
        else:
            def target_log_prob(*params):
                return self.target_log_prob_fn(params + obs_tuple)

        # Handle seed - ensure it's a PRNGKey
        if isinstance(seed, (int, float)):
            key = jax.random.PRNGKey(int(seed))
        else:
            key = seed

        self.sampler = self.NUTS(model = self.tensor, obs = obs_tuple,  n_chains = n_chains,  target_log_prob_fn = target_log_prob,
        num_results = num_results, num_burnin_steps=num_burnin_steps, num_steps_between_results=num_steps_between_results,
        parallel_iterations = parallel_iterations, seed=key, name=name)

        return self.sampler

