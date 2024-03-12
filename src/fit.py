# %%
import tensorflow_probability as tfp
import tensorflow as tf
import functools
import arviz as az
import numpy as np

@tf.function(autograph=False)
def trace_fn(_, pkr): 
    return (
        pkr.inner_results.inner_results.accepted_results.target_log_prob,
        pkr.inner_results.inner_results.accepted_results.num_leapfrog_steps,
        #pkr.inner_results.inner_results.accepted_results.has_divergence,
        #pkr.inner_results.inner_results.accepted_results.energy,
        pkr.inner_results.inner_results.log_accept_ratio
    )

@tf.function(autograph=False)
def target_log_prob_fn(model, observed_data, *args):    
    param_dict = {name: value for name, value in zip(model._flat_resolve_names(), args)}
    param_dict= {**param_dict, **observed_data}
    return model.log_prob(**param_dict) 
    #return model.log_prob(model.sample(**param_dict))   

@tf.function(autograph=True)
def sampleH(model,
            observed_data,
            params,
            init,
            bijectors,
            parallel_iterations,
            num_results = 2000, 
            num_burnin_steps=500,
            step_size = 0.065,
            num_leapfrog_steps = 5,
            num_adaptation_steps = 400,
            num_chains = 4):
    
    unnormalized_posterior_log_prob = functools.partial(target_log_prob_fn, model, observed_data)

    if init is None:
        # For multiple likelihoods, initial_state need to remove the correct outputs
        init = model.sample(num_chains)
        for k in observed_data.keys():
            init.pop(k)

        init = list(init.values())

    if bijectors is None:
        bijectors = [tfp.bijectors.Identity() for _ in init]

    #print(params)
    #print(model)
    #print(num_chains)
    #print(model.sample(num_chains))
    #initial_state = [model.sample(num_chains)[param].numpy() for param in params]    
    #bijectors = [tfp.Identity() for _ in params]

    results, sample_stats =  tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=init,
    kernel=tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                step_size= step_size,
                num_leapfrog_steps= num_leapfrog_steps),
            bijector=bijectors),
         num_adaptation_steps= num_adaptation_steps),
    trace_fn = trace_fn,
    parallel_iterations = parallel_iterations)  
    return results, sample_stats

def tfp_trace_to_arviz(
    posterior, sample_stats,
    var_names=None, 
    sample_stats_name=['log_likelihood','tree_size','mean_tree_accept']):

    if var_names is None:
        var_names = ["var " + str(x) for x in range(len(sample_stats))]
        
    sample_stats = {k:v.numpy().T for k, v in zip(sample_stats_name, sample_stats)}
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
    return trace

@tf.function(autograph=False)
def run_modelH(model, 
                observed_data,
                params,
                init = None,
                bijectors = None,
                parallel_iterations = 1,
                num_results = 2000, 
                num_burnin_steps=500,
                step_size = 0.065,
                num_leapfrog_steps = 5,
                num_adaptation_steps = 400,
                num_chains = 4):
    tf.config.experimental.enable_tensor_float_32_execution(False)   
    res  =  sampleH(model = model, observed_data = observed_data,
                    params = params,
                    init = init,
                    bijectors = bijectors,
                    parallel_iterations = parallel_iterations,
                    num_results = num_results, 
                    num_burnin_steps= num_burnin_steps,
                    step_size = step_size,
                    num_leapfrog_steps = num_leapfrog_steps,
                    num_adaptation_steps = num_adaptation_steps,
                    num_chains = num_chains)
    
    return res


class fit():
    def __init__():
        pass

    
    def run_model(self, observed_data,params,
                init = None,
                bijectors = None,
                parallel_iterations=1,
                num_results=2000,
                num_burnin_steps=500,
                step_size=0.065,
                num_leapfrog_steps=5,
                num_adaptation_steps=400,
                num_chains=4):
        
        self.parallel_iterations = parallel_iterations
        self.num_results = num_results
        self.num_burnin_steps = num_burnin_steps
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.num_adaptation_steps = num_adaptation_steps
        self.num_chains = num_chains 
        self.params = self.priors
        self.hmc_results = None
        #self.hmc_sample_stats = None
        self.hmc_posterior = None

        res = run_modelH(self.tensor, 
                        observed_data,
                        params = params,
                        init = init,
                        bijectors = bijectors,
                        parallel_iterations = parallel_iterations,
                        num_results = num_results, 
                        num_burnin_steps=num_burnin_steps,
                        step_size = step_size,
                        num_leapfrog_steps = num_leapfrog_steps,
                        num_adaptation_steps = num_adaptation_steps,
                        num_chains = num_chains)
        self.res = res
        posterior, sample_stats = res
        p = dict(zip(self.tensor._flat_resolve_names(), posterior))
        az_trace = tfp_trace_to_arviz(posterior, sample_stats, p)    
        return dict(p), az_trace, sample_stats







# %%