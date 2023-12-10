
#%%
from model_diagnostic import *
from model_fit import *
from model_write import *
import pandas as pd
d = pd.read_csv('C:/Users/sebastian_sosa/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

#%%
model = dict(main = 'height ~ Normal(mu,sigma)',
            prior1 = 'mu ~ Normal(178.0, 0.1)',            
            prior2 = 'sigma ~ Uniform(0.0, 50.0)')    

model = build_model(model, path = None, df = None, sep = ',', float=32)

#%%
model.sample()

#%%
posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(height = d.height.astype('float32').values),
                                            params = ['mu', 'sigma'],
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%%
plot_prior_dist(model)

#%%
model_check(posterior, trace, sample_stats, params = ['sigma', 'alpha', 'beta'])   

# %%
samples = model.sample(**posterior)
plt.plot(samples['height'][1].numpy(), weight)