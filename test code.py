#%%
from code.model_diagnostic import *
from code.model_fit import *
from code.model_write import *
from code.data_manip import *
import pandas as pd

d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

#%% Test No data frame single likelihood -----------------------------------------------------
model = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')    

model = build_model(model, float = 16)        


posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(y =  d.weight.astype('float32').values),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test No data frame multiple likelihood-----------------------------------------------------
model = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)',
            
            main1 = 'z~Normal(m2,s2)',
            likelihood2 = 'm2 ~ alpha2 + beta2',
            prior4 = 's2~Exponential(1)',
            prior5 = 'alpha2 ~ Normal(0,1)',
            prior6 = 'beta2 ~ Normal(0,1)')    

model = build_model(model, float = 16)        


posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(y = d.weight.astype('float32').values),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Test with data frame in likelihood-----------------------------------------------------
d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

formula = dict(main = 'height ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * weight',
            prior1 = 'sigma~Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,20)',
            prior3 = 'beta ~ Normal(0,1)')    

model = build_model(formula, df = d, sep = ',', float=32)


posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(height =  'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Test  with data frame only as output-----------------------------------------------------
model = dict(main = 'height ~ Normal(mu,sigma)',
            prior1 = 'mu ~ Normal(178.0, 0.1)',            
            prior2 = 'sigma ~ Uniform(0.0, 50.0)')    


model = build_model(model, path = None, df = d, sep = ',', float=32)

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(height = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test with multiple likelihood-----------------------------------------------------
model = dict(main = 'z ~ Poisson(alpha)',
            likelihood = 'alpha ~ a + beta * weight',
            prior1 = 'a ~ Normal(178,20)',
            prior2 = 'beta ~ Normal(0,1)',
            
            main2 = 'y ~ Normal(mu, sigma)',
            likelihood2 = 'mu ~ alpha2 + beta * age',
            prior4 = 'sigma ~ Normal(0.0, 1.0)',
            prior5 = 'alpha2 ~ Normal(0,1)'
            )    

model = build_model(model, path = None, df = d, sep = ',', float=32)

model.sample()

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test error modeling-----------------------------------------------------
# Initialize a single 2-variate Gaussian.
mvn = tfd.MultivariateNormalTriL(
    loc=[1., 10, -10],
    scale_tril = [[1., 0.9, -0.8],
                  [0.9, 1, 0.4],
                  [-0.8, 0.4, 1]])


df = pd.DataFrame(mvn.sample(30).numpy())
df.columns = ['y', 'att', 'exposure']
df.plot()
df.corr()

model = dict(main = 'y ~ Normal(mu, sigma)',
            likelihood = 'mu ~ a + b * att + exposureM',
            prior1 = 'a ~ Normal(0,1)',
            prior2 = 'b ~ Normal(0,1)',
            prior3 = 'sigma~Normal(0,1)',
            
            main2 = 'exposureM ~ Normal(mu2, sigma2)',
            likelihood2 = 'mu2 ~ a2 + b2 * att + b3 * exposure',
            prior4 = 'a2 ~ Normal(0,1)',
            prior5 = 'b2 ~ Normal(0,1)',
            prior6 = 'b3 ~ Normal(0,1)',
            prior7 = 'sigma2~Normal(0,1)'
            )

model = build_model(model, path = None, df = df, sep = ',', float=32)

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'y'),
                                            num_chains = 4, inDF = True)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Categorical models with OHE code 5.45
d = pd.read_csv('./data/Howell1.csv', sep=';')
formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * male',
            prior1 = 'sigma~Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,100)',
            prior3 = 'beta ~ Normal(0,10)')     

model = build_model(formula, df = d, sep = ',', float=32)
posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Categorical models with OHE multiple categories code 5.52
d = pd.read_csv('./data/milk.csv', sep=';')
d = OHE(d, ['clade'])
d.columns

formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + B1*clade_New_World_Monkey + B2*clade_Old_World_Monkey + B3*clade_Strepsirrhine',
            prior1 = 'sigma~Uniform(0,10)',
            prior2 = 'alpha ~ Normal(0.6,10)',
            prior3 = 'B1 ~ Normal(0,1)',
            prior4 = 'B2 ~ Normal(0,1)',
            prior5 = 'B3 ~ Normal(0,1)',)     

model = build_model(formula, df = d, sep = ',', float=32)
posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'kcal_per_g'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Categorical models with index and multiple categories code 5.45
d = pd.read_csv('./data/milk.csv', sep=';')
d.rename(columns={'kcal.per.g':'kcal'}, inplace=True)
d['cladeID'] = d.clade.astype("category").cat.codes
d.cladeID = d.cladeID.astype(np.int64)
CLADE_ID_LEN = len(set(d.cladeID.values))

m = tfd.JointDistributionNamed(dict(
    sigma = tfd.Sample(tfd.Uniform(0, 50), sample_shape=1),
    alpha = tfd.Sample(tfd.Normal(178, 20), sample_shape=4),
    alpha_cladeID = lambda alpha: tfd.Deterministic(tf.transpose(tf.gather(tf.transpose(alpha), tf.cast(d.cladeID , dtype=tf.int64)))),
    y = lambda alpha_cladeID, sigma: tfd.Independent(tfd.Normal(alpha_cladeID, sigma), 
                                                  reinterpreted_batch_ndims=1),
))
observed_data = dict(y = d.kcal.astype('float32').values,)
model = m
def target_log_prob_fn(model, observed_data,*args):
    param_dict = {name: value for name, value in zip(model._flat_resolve_names(), args)}
    param_dict= {**param_dict, **observed_data}
    print(args)
    return model.log_prob(param_dict)

unnormalized_posterior_log_prob = functools.partial(target_log_prob_fn, model, observed_data)
# Assuming that 'model' is defined elsewhere
initial_state = list(model.sample(4).values())[:-1]
bijectors = [tfp.bijectors.Identity() for _ in initial_state]
results, sample_stats =  tfp.mcmc.sample_chain(
num_results=2000,
num_burnin_steps=200,
current_state=initial_state,
kernel=tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size= 0.065,
            num_leapfrog_steps= (5)),
        bijector=bijectors),
     num_adaptation_steps= 400),
trace_fn=lambda _, pkr: pkr.inner_results.inner_results.log_accept_ratio,
parallel_iterations = 1)  


#%%
posterior, trace, sample_stats =  run_model(model = m,
parallel_iterations=1,
num_results=2000,
num_burnin_steps=500,
step_size=0.065,
num_leapfrog_steps=5,
num_adaptation_steps=400,
num_chains=4,
observed_data = dict(y = d.kcal.astype('float32').values,))
#%%
az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Model diag -----------------------------------------------------
plot_prior_dist(model)

model_check(posterior, trace, sample_stats, params = ['sigma', 'alpha', 'beta'])   

samples = model.sample(**posterior)
plt.plot(samples['height'][1].numpy(), weight)