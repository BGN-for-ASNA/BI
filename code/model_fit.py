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
    print(posterior)
    az_trace = _trace_to_arviz(trace=posterior, sample_stats=sampler_stats)
    return az_trace

def run_model(model, 
              params,
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
    
    az_trace = model_to_az(sample_stats, posterior, params)       
    p={}
    for a in range(len(params)):
        p[params[a]] = posterior[a]
    return p, az_trace, sample_stats

#%%
#from model_write import *
#import arviz as az
#d = pd.read_csv('C:/Users/sebastian_sosa/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv', sep=';')
#
#d = d[d.age > 18]
#d.weight = d.weight - d.weight.mean()
#
#model = dict(main = 'height ~ Normal(m,sigma)',
#            likelihood = 'm = alpha + beta * weight',
#            prior1 = 'sigma~Uniform(0, 50)',
#            prior2 = 'alpha ~ Normal(178,20)',
#            prior3 = 'beta ~ Normal(0,1)')    
#
#model = build_model(model, path = None, df = d, sep = ',', float=64)
#
#posterior, trace, sample_stats =  run_model(model, 
#                                            observed_data = dict(height =d.height.astype('float32').values),
#                                            params = ['sigma', 'alpha', 'beta'],
#                                            num_chains = 1)
#import arviz as az
#az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)
# %%
