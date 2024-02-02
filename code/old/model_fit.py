# %%
import tensorflow_probability as tfp
from model_write import *
import tensorflow as tf
import functools
import arviz as az

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

def target_log_prob_fn(model, observed_data,*args):
    param_dict = {name: value for name, value in zip(model._flat_resolve_names(), args)}
    param_dict= {**param_dict, **observed_data}
    print(param_dict)
    return model.log_prob(model.sample(**param_dict))                                          

@tf.function(autograph=False)
def sample(model,
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
    trace_fn=lambda _, pkr: pkr.inner_results.inner_results.log_accept_ratio,
    parallel_iterations = parallel_iterations)  
    return results, sample_stats
  
def model_to_az(sample_stats, results, params):    
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

def run_model(model, 
              observed_data,
              parallel_iterations = 1,
              num_results = 2000, 
              num_burnin_steps=500,
              step_size = 0.065,
              num_leapfrog_steps = 5,
              num_adaptation_steps = 400,
              num_chains = 4):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    
    #saved_args = locals()
    #print("saved_args is", saved_args)
    #write_fit(args = saved_args)
    
    posterior, sample_stats  = sample(model = model, observed_data = observed_data,
                                      parallel_iterations = parallel_iterations,
                                      num_results = num_results, 
                                      num_burnin_steps= num_burnin_steps,
                                      step_size = step_size,
                                      num_leapfrog_steps = num_leapfrog_steps,
                                      num_adaptation_steps = num_adaptation_steps,
                                      num_chains = num_chains)   
    
    p = dict(zip(model._flat_resolve_names(), posterior))
    az_trace = model_to_az(sample_stats, posterior, list(p.keys()))       

    
    return dict(p), az_trace, sample_stats

def fit_model(model, 
              observed_data,
              parallel_iterations = 1,
              num_results = 2000, 
              num_burnin_steps=500,
              step_size = 0.065,
              num_leapfrog_steps = 5,
              num_adaptation_steps = 400,
              num_chains = 4,
              float = 32,
              inDF = True):
    write_HMC(model, 
              observed_data,
              parallel_iterations = parallel_iterations,
              num_results = num_results, 
              num_burnin_steps = num_burnin_steps,
              step_size = step_size,
              num_leapfrog_steps = num_leapfrog_steps,
              num_adaptation_steps = num_adaptation_steps,
              num_chains = num_chains,
              float = float,
              inDF = inDF
             )
    
    import importlib
    from output import mymodel
    importlib.reload(mymodel)
    from output.mymodel import posterior, trace, sample_stats
    return posterior, trace, sample_stats

#%%
#from model_write import *
#import arviz as az
#d = pd.read_csv('/home/sosa/BI/data/Howell1.csv', sep=';')
#
#d = d[d.age > 18]
#d.weight = d.weight - d.weight.mean()
#weight = d.weight
#
#model = tfd.JointDistributionNamed(dict(
#	sigma = tfd.Sample(tfd.Uniform(0, 50), sample_shape=1),
#	alpha = tfd.Sample(tfd.Normal(178, 20), sample_shape=1),
#	beta = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
#	height = lambda alpha,beta,sigma: tfd.Independent(tfd.Normal(alpha+beta*weight, sigma), reinterpreted_batch_ndims=1),
#))
#
##%%
#posterior, trace, sample_stats =  run_model(model, 
#                                            observed_data = dict(height =d.height.astype('float32').values),
#                                            num_chains = 1)
#import arviz as az
#az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)
## %%
#num_chains = 1
#observed_data = dict(height =d.height.astype('float32').values)
#unnormalized_posterior_log_prob = functools.partial(target_log_prob_fn, model, observed_data)
## Assuming that 'model' is defined elsewhere
#initial_state = list(model.sample(num_chains).values())[:-1]
#bijectors = [tfp.bijectors.Identity() for _ in initial_state]
#results, sample_stats =  tfp.mcmc.sample_chain(
#num_results=1,
#num_burnin_steps=1,
#current_state=initial_state,
#kernel=tfp.mcmc.SimpleStepSizeAdaptation(
#    tfp.mcmc.TransformedTransitionKernel(
#        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
#            target_log_prob_fn=unnormalized_posterior_log_prob,
#            step_size= 0.065,
#            num_leapfrog_steps= 5),
#        bijector=bijectors),
#     num_adaptation_steps= 1),
#trace_fn=lambda _, pkr: pkr.inner_results.inner_results.log_accept_ratio,
#parallel_iterations = 1)  
    
