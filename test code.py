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
d = index(d,["clade"])
d["kcal_per_g"] = d["kcal_per_g"].pipe(lambda x: (x - x.mean()) / x.std())
CLADE_ID_LEN = len(set(d.index_clade.values))
formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha[index_clade]',
            prior1 = 'sigma~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,0.5)')   
#%%
# index must be within main equation to handle symbolic tensor
m = tfd.JointDistributionNamed(dict(
    sigma = tfd.Sample(tfd.Exponential(1), sample_shape=1),
    alpha = tfd.Sample(tfd.Normal(0, 0.5), sample_shape=4),
    y = lambda alpha, sigma: tfd.Independent(tfd.Normal(
        loc=tf.transpose(tf.gather(tf.transpose(alpha), tf.cast(d.index_clade , dtype= tf.int32))),
        scale=sigma
    ), reinterpreted_batch_ndims=1),
))
#%%
posterior, trace, sample_stats =  run_model(model = m,
parallel_iterations=1,
num_results=2000,
num_burnin_steps=500,
step_size=0.065,
num_leapfrog_steps=5,
num_adaptation_steps=400,
num_chains=4,
observed_data = dict(y = d.kcal_per_g.astype('float32').values,))

#%%
# alpha indeces must be gather
ape_alpha, nwm_alpha, owm_alpha, strep_alpha = tf.split(
    posterior["alpha"][0], 4, axis=1
)

updated_posterior = {
    "ape_alpha": ape_alpha.numpy(),
    "nwm_alpha": nwm_alpha.numpy(),
    "owm_alpha": owm_alpha.numpy(),
    "strep_alpha": strep_alpha.numpy(),
    "sigma": posterior["sigma"][0],
}


az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)



# %%
from  main import *
my_formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha[index_clade]',
            prior1 = 'sigma~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,0.5)')  
self = model()
self.import_csv('./data/milk.csv', sep = ';')
self.index(cols = "clade")
self.formula(f = my_formula)

self.build_model()
