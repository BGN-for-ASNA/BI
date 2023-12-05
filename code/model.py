#%%
from operator import length_hint
from pickletools import float8
from tracemalloc import stop
from matplotlib.patheffects import Normal
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import re
import numpy as np

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


## Distribution functions -----------------------------------------------------
def get_distribution_classes():
    # Get all names defined in the distributions module
    all_names = dir(tfd)
    
    # Filter to include only classes
    class_names = [name for name in all_names if isinstance(getattr(tfd, name), type)]
    
    # Create a dictionary of class names and corresponding classes
    class_dict = {name: getattr(tfd, name) for name in class_names}
    
    return class_dict

tf_classes = get_distribution_classes()

def exportTFD(tf_classes):
    for key in tf_classes.keys():
        globals()[key] = tf_classes[key]
        
exportTFD(tf_classes)

## Formula functions -----------------------------------------------------
def get_formula(formula = "y~Normal(0,1)", type = 'likelihood'):

    y, x = re.split(r'[~=]',formula)
    y = y.replace(" ", "")
    
    if 'likelihood' in type:
        y, x = re.split(r'[=]',formula)
        y = y.replace(" ", "")    
        args = re.split(r'[+*()]',x)
        for i in range(len(args)):
            args[i] = args[i].replace(" ", "") 
        return [y, args]     
    else:
        dist, args = x.split('(')
        dist = dist.replace(" ", "")
        args = args.replace("(", "")
        args = args.replace(")", "")
        args = args.split(",") 
        #args = args.replace(" ", "")
        return [y, dist, args]
    
def get_var(model):
    full_model = {}
    # Extract variables
    for  key in model.keys():
         full_model[key] = dict(
             input = model[key],
             var = get_formula(formula=model[key], type=key)
             ) 
    return full_model   

def get_undeclared_params(model, df):
    Vars = []
    params = []
    for key in model.keys():
        if 'main' in key:
            if df.empty: 
                tmp = model[key]['var'][2:]
            else:
                tmp = model[key]['var']
        else:
            tmp = model[key]['var']
        for a in range(len(tmp)):
            if isinstance(tmp[a], list):
                params.append(tmp[a])
            else:
                if a == 0:
                    Vars.append(tmp[0].replace(' ', ''))                
    params = [item.replace(' ', '') for sublist in params for item in sublist]
    undeclared_params = list(set(Vars) ^ set(params))
    undeclared_params2 = []
    for a in range(len(undeclared_params)):
        if undeclared_params[a].isdigit() != True:
            undeclared_params2.append(undeclared_params[a])
    
    if df.empty:  
        return undeclared_params2
    else:
        test = pd.Index(undeclared_params2).difference(df.columns).tolist()
        test2 =  list(set(undeclared_params2) & set(df.columns))
        return {'undeclared_params': test, 'params_in_data' : test2}  
        

## Write model functions-----------------------------------------------------
def get_likelihood(model, main_name):
    result = []
    for key in model.keys():
        if 'likelihood' in key:
            name = model[key]['var'][0]
            if name in model[main_name]['var'][2]:
                index = model[main_name]['var'][2].index(name)
                y, x = re.split(r'[=]',model[key]['input'])
                result.append(x)
                result.append(index)
    if len(result) >= 1:
        return result
    else:
        return None
    
def write_header(output_file):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        #file.write("import tensorflow as tf")    
        #file.write('\n')
        #file.write("import pandas as pd")    
        #file.write('\n')
        #file.write("from bayesian.distribution import dist")    
        #file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

def write_header_with_dataFrame(output_file, DFpath, data):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        file.write("import pandas as pd")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        
        file.write("d = pd.read_csv('" + DFpath +"',sep = ';')" )    
        file.write('\n')
        
    for a in range(len(data)):
        with open(output_file,'a') as file:
            file.write(data[a] + "= d." + data[a])  
            file.write('\n')
        
    with open(output_file,'a') as file:   
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

def write_priors(model, output_file):    
    p = [] # Store model parameters name
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['var']
        if 'prior' in key.lower():
            p.append(var[0])
            with open(output_file,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + 
                           " = tfd.Sample(tfd." + var[1] + "(" + 
                            str(','.join(var[2])) + ")),")
                file.write('\n')
    return p           

def write_main(model, output_file, p ):    
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['var']
        if 'main' in key.lower():
            with open(output_file,'a') as file:
                file.write('\t')                
                
                file.write(str(var[0]) + " = lambda " + 
                            str(','.join(p)) + ":" +
                            " tfd.Independent(tfd."+ var[1] + "(")
                
                if 'likelihood' in model.keys():
                    print(key)
                    formula = get_likelihood(model, key)
                    print(str(formula))
                    if formula is not None:
                        var[2][formula[1]] = formula[0]
                        file.write(str(','.join(var[2]))+ ")),")
                    else:
                        file.write(str(','.join(var[2]))+ ")),")
                file.write('\n')
                
    with open(output_file,'a') as file:
        file.write('))')

def write_model(model, path = 'mymodel.py', withDF = False, DFpath = None, data  = None):
    if withDF == False:
        write_header(path)
    else:
        write_header_with_dataFrame(path, DFpath, data)
        
    p = write_priors(model, path)
    write_main(model, path, p)

## Mains -----------------------------------------------------
def build_model(model, path = None):
    
    if  path is None:
        df = pd.DataFrame({'A' : []})
    else:        
        df = pd.read_csv(path, sep = ';')
        
    full_model = get_var(model)
    
    issues = get_undeclared_params(full_model, df = df)
    
    if df.empty :
        if len(issues) == 0:
            print('Non missing variables')
            write_model(full_model)
        else:
            print("Arguments are missing: " + ''.join(issues))
            return None
    else:
        data = get_undeclared_params(full_model, df = df)
        if len(data['undeclared_params']) == 0:
            data = data['params_in_data']
            write_model(full_model, withDF = True, DFpath = path, data  = data)
        else:
           print("Arguments are missing: " + ''.join(data['undeclared_params'])) 
           return None        

    import importlib
    import mymodel
    importlib.reload(mymodel)
    from mymodel import m
    
    return m


#%%
## test No data frame -----------------------------------------------------
model = dict(main = 'y~Normal(m,s)',
            likelihood = 'm = alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)',
            
            main1 = 'z~Normal(m2,s2)',
            likelihood2 = 'm2 = alpha2 + beta2',
            prior4 = 's2~Exponential(1)',
            prior5 = 'alpha2 ~ Normal(0,1)',
            prior6 = 'beta2 ~ Normal(0,1)')    
model = build_model(model)        
model
# %%
model.sample()
# %%
model.sample(2)
# %%
model.sample(y = [5, 6 ])
# %%
model.log_prob(model.sample())


#%%
## test with data frame -----------------------------------------------------
model = dict(main = 'weight~Normal(m,s)',
            likelihood = 'm = alpha + beta * height',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')    

model2 = build_model(model, 
            path = "D:/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv")
model2

# %%
model2.sample()
# %%
model2.sample(544)

# %%
model2.log_prob(model2.sample())


# %%
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf
import functools
from tensorflow_probability import distributions as tfd

d = pd.read_csv('C:/Users/sebastian_sosa/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv', sep=';')
d = d[d.age > 18]
weight = d.weight - d.weight.mean()
observed_data= d.height.values
params = ['sigma', 'alpha', 'beta']
# Your model definition here
model = tfp.distributions.JointDistributionNamed(dict(
    sigma= tfd.Sample(tfp.distributions.Uniform(low=0.0, high=50.0), sample_shape=1),
    alpha= tfd.Sample(tfp.distributions.Normal(loc=178.0, scale=20.0), sample_shape=1),
    beta= tfd.Sample(tfp.distributions.LogNormal(loc=0.0, scale=1.0), sample_shape=1),
    y=lambda sigma, alpha, beta: tfp.distributions.Independent(
        tfp.distributions.Normal(alpha + beta*weight, sigma), reinterpreted_batch_ndims=1
        )
    )
)                                            

def target_log_prob_fn(observed_data, *args):
    param_dict = {name: value for name, value in zip(model._flat_resolve_names(), args)}
    # Assuming that 'y' is a key in the model
    return model.log_prob(model.sample(y=observed_data, **param_dict))

# Add explicit shape information to observed_data
#observed_data = tf.constant(observed_data, dtype=tf.float32, shape=[len(observed_data)])

unnormalized_posterior_log_prob = functools.partial(target_log_prob_fn, observed_data)

# Assuming that 'model' is defined elsewhere
initial_state = list(model.sample().values())[:-1]

bijectors = [tfp.bijectors.Identity() for _ in initial_state]

@tf.function(autograph=False)
def sample():
  return tfp.mcmc.sample_chain(
    num_results=2000,
    num_burnin_steps=500,
    current_state=initial_state,
    kernel=tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                step_size=0.065,
                num_leapfrog_steps=5),
            bijector=bijectors),
         num_adaptation_steps=400),
    trace_fn=lambda _, pkr: pkr.inner_results.inner_results.log_accept_ratio)
  
results, sample_stats  = sample()


#%%
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
az_trace = _trace_to_arviz(trace=posterior, sample_stats=sampler_stats)


_, ax = plt.subplots(1, len(params), figsize=(8, 4))
az.plot_posterior(az_trace, var_names=params, ax=ax);

# %%
az.summary(az_trace, round_to=2, kind="stats", hdi_prob=0.89)

# %%
az.plot_trace(az_trace, compact=True)
plt.tight_layout()
# %%
