# %%
import tensorflow_probability as tfp
import tensorflow as tf
import functools
import arviz as az
import numpy as np

def trace_fn(_, pkr):  
    return (
        pkr.inner_results.inner_results.accepted_results.target_log_prob,
        pkr.inner_results.inner_results.accepted_results.num_leapfrog_steps,
        #pkr.inner_results.inner_results.accepted_results.has_divergence,
        #pkr.inner_results.inner_results.accepted_results.energy,
        pkr.inner_results.inner_results.log_accept_ratio
    )

def target_log_prob_fn(model, observed_data, *args):
    param_dict = {name: value for name, value in zip(model._flat_resolve_names(), args)}
    param_dict= {**param_dict, **observed_data}
    return model.log_prob(model.sample(**param_dict))   

def _trace_to_arviz(
    trace=None,
    sample_stats=None,
    observed_data=None,
    prior_predictive=None,
    posterior_predictive=None,
    inplace=True,
):

    if trace is not None and isinstance(trace, dict):
        trace = {k: v.numpy() for k, v in trace.items()}
    if sample_stats is not None and isinstance(sample_stats, dict):
        sample_stats = {k: v.numpy().T for k, v in sample_stats.items()}
    if prior_predictive is not None and isinstance(prior_predictive, dict):
        prior_predictive = {k: v[np.newaxis] for k, v in prior_predictive.items()}
    if posterior_predictive is not None and isinstance(posterior_predictive, dict):
        if isinstance(trace, az.InferenceData) and inplace == True:
            return trace + az.from_dict(posterior_predictive=posterior_predictive)
        else:
            trace = None

    return az.from_dict(
        posterior=trace,
        sample_stats=sample_stats,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
    )

@tf.function(autograph=False)
def sampleH(model,
           observed_data,
           parallel_iterations,
           num_results = 2000, 
           num_burnin_steps=500,
           step_size = 0.065,
           num_leapfrog_steps = 5,
           num_adaptation_steps = 400,
           num_chains = 4):
    unnormalized_posterior_log_prob = functools.partial(target_log_prob_fn, model, observed_data)

    # Assuming that 'model' is defined elsewhere
    initial_state = list(model.sample(num_chains).values())[:-1]

    bijectors = [tfp.bijectors.Identity() for _ in initial_state]

    results, sample_stats =  tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_state,
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
  
def model_to_azH(sample_stats, results, params):    
    stat_names = ["mean_tree_accept"]
    sampler_stats = dict(zip(stat_names, [sample_stats]))
    transposed_results = []
    for r in results:
        if len(r.shape) == 2:
            transposed_shape = [1, 0]
        elif len(r.shape) == 3:
            transposed_shape = [1, 0, 2]
        else:
            transposed_shape = [1, 0, 2, 3]
        transposed_results.append(tf.transpose(r, transposed_shape))
    posterior = dict(zip(params, transposed_results))

    az_trace = _trace_to_arviz(trace=posterior, sample_stats = sampler_stats)
    return az_trace

def tfp_trace_to_arviz(
    tfp_trace,
    var_names=None, 
    sample_stats_name=['log_likelihood','tree_size','mean_tree_accept']):
    
    samps, trace = tfp_trace
    if var_names is None:
        var_names = ["var " + str(x) for x in range(len(samps))]
        
    sample_stats = {k:v.numpy().T for k, v in zip(sample_stats_name, trace)}
    posterior = {name : tf.transpose(samp, [1, 0, 2]).numpy() for name, samp in zip(var_names, samps)}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)

def run_modelH(model, 
              observed_data,
              parallel_iterations = 1,
              num_results = 2000, 
              num_burnin_steps=500,
              step_size = 0.065,
              num_leapfrog_steps = 5,
              num_adaptation_steps = 400,
              num_chains = 4):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    

    
    res  =  sampleH(model = model, observed_data = observed_data,
                                      parallel_iterations = parallel_iterations,
                                      num_results = num_results, 
                                      num_burnin_steps= num_burnin_steps,
                                      step_size = step_size,
                                      num_leapfrog_steps = num_leapfrog_steps,
                                      num_adaptation_steps = num_adaptation_steps,
                                      num_chains = num_chains)
    posterior, sample_stats = res
    p = dict(zip(model._flat_resolve_names(), posterior))
    az_trace = tfp_trace_to_arviz(res, p)
    #az_trace = model_to_azH(sample_stats, posterior, list(p.keys()))    

    #p = dict(zip(model._flat_resolve_names(), posterior))
    #az_trace = tfp_trace_to_arviz()
    
    return dict(p), az_trace, sample_stats

class fit():
    def __init__():
        pass
    
    def run_model(self, observed_data,
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
        self.hmc_sample_stats = None
        self.hmc_posterior = None

        posterior, trace, sample_stats = run_modelH(self.tensor, 
                                        observed_data,
                                        parallel_iterations = parallel_iterations,
                                        num_results = num_results, 
                                        num_burnin_steps=num_burnin_steps,
                                        step_size = step_size,
                                        num_leapfrog_steps = num_leapfrog_steps,
                                        num_adaptation_steps = num_adaptation_steps,
                                        num_chains = num_chains)
        return posterior, trace, sample_stats





