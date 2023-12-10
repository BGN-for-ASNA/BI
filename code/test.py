#%%
from model_diagnostic import *
from model_fit import *
from model_write import *
import pandas as pd

#%% Test No data frame -----------------------------------------------------
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
model.sample()

#%% Test with data frame in path-----------------------------------------------------
model = dict(main = 'weight~Normal(m,s)',
            likelihood = 'm ~ alpha + beta * height',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')    

model2 = build_model(model, 
            path = "../data/Howell1.csv", sep = ';')
model2.sample()

#%% Test with data frame in function-----------------------------------------------------
d = pd.read_csv('C:/Users/sebastian_sosa/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

formula = dict(main = 'height ~ Normal(m,s)',
            likelihood = 'm ~ alpha + beta * weight',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')    

model =build_model(formula, path = None, df = d, sep = ',', float=32)
model.sample()

#%% Test  with data frame only as output-----------------------------------------------------
model = dict(main = 'height ~ Normal(mu,sigma)',
            prior1 = 'mu ~ Normal(178.0, 0.1)',            
            prior2 = 'sigma ~ Uniform(0.0, 50.0)')    

model = build_model(model, path = None, df = None, sep = ';', float=32)
model.sample()

#%% HMC -----------------------------------------------------
posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(height = d.height.astype('float32').values),
                                            params = ['mu', 'sigma'],
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Model diag -----------------------------------------------------
plot_prior_dist(model)

model_check(posterior, trace, sample_stats, params = ['sigma', 'alpha', 'beta'])   

samples = model.sample(**posterior)
plt.plot(samples['height'][1].numpy(), weight)