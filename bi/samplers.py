import time as tm
import tensorflow_probability 
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
import inspect
import re

def get_model_distributions(model):
    source_code = inspect.getsource(model)
    lines = source_code.split('\n')
    variables = {}
    for line in lines:
        if not line or line.startswith('def') or 'independent' in line.lower() or not 'yield' in line:
            continue
        # Split the line into key and value
        key, value = line.split('=', 1)
        # Remove leading and trailing whitespace
        key = key.strip()
        # Find all words before the brackets
        words = re.findall(r'\b\w+\b(?=\()', value)
        # Create a dictionary with 'distribution' as the key and words as the value
        distribution = {
            'distribution': words[0]
        }
        # Add the key-value pair to the dictionary
        variables[key] = distribution
    return variables


def initialise(infos, init_params):
    init_params2 = []
    bijectors = []
    i = 0
    for key in infos.keys():  
        tmp = infos[key]['distribution'].lower()
        if 'lkj' in tmp:
            infos[key]['shape'] = int(init_params[i].shape[0])
            init_params2.append(jnp.array(jnp.eye(infos[key]['shape'])))            
            bijectors.append(tfb.CorrelationCholesky())
        elif 'exponential' in tmp:
             init_params2.append(jnp.array(jnp.ones_like(init_params[i])))
             infos[key]['shape'] = init_params[i].shape
             bijectors.append(tfb.Exp())
        else:
            init_params2.append(jnp.array(jnp.ones_like(init_params[i])))
            infos[key]['shape'] = init_params[i].shape
            bijectors.append(tfb.Identity())
        i+=1
    return init_params2, bijectors



def NUTS(model,  obs, infos,  n_chains = 1, init = None, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, seed=0, name=None):

    #init_key, key = jax.random.split(jax.random.PRNGKey(int(seed)))
    #init_key = jnp.array(init_key)

    #tensor = JointDistributionCoroutine(model)
    ##infos = get_model_distributions(model)
    #init_params = tensor.sample(seed = init_key)
    
    #_, bijectors = initialise(infos, init_params)
    #init_params = list(init_params)[:-1]

    names = infos.keys()
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio
        )
    
    if target_log_prob_fn == None:
        def target_log_prob(*params):
            return tensor.log_prob(params + (obs,))
    else:
        def target_log_prob(*params):
            return target_log_prob_fn(params + (obs,))
    @jit
    def run_chain(key):
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob,
            step_size= 1e-3
        )

        kernel = tensorflow_probability.substrates.jax.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_kernel,
                bijector=bijectors
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
                                     current_state= init_params,
                                     kernel=hmc,
                                     trace_fn=trace_fn,
                                     num_burnin_steps=num_burnin_steps,
                                     parallel_iterations = parallel_iterations,
                                     seed=key)
    
    Ndevices = jax.local_device_count(backend=None)

    if(n_chains > Ndevices):
        runs = jnp.ceil(n_chains/Ndevices)
        print(runs)
        result = []
        for run in range(int(runs)):
            rng_keys = jax.random.split( jax.random.PRNGKey(0), Ndevices)
            result.append(jax.pmap(run_chain)(rng_keys))

        return result
    else:
        start = tm.time()  
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_chains)
        result =  jax.pmap(run_chain)(rng_keys)
        end = tm.time()    
        print(f"HonnorMode took: {end - start:.4f} seconds")
        posterior, sample_stats = result

    return posterior, sample_stats 