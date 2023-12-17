#%%
from code.model_diagnostic import *
from code.model_fit import *
from code.model_write import *
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

#%% Categorical models
from code.model_diagnostic import *
from code.model_fit import *
from code.model_write import *
import pandas as pd
import random
from pandas.api.types import is_string_dtype


d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

d.insert(2, 'cat', d.male.apply(lambda x: random.choice(['old', 'adult', 'juveniles']) ) )

# If category in model then generate OHE and modify data and formula
d2 = OHE(d)
d2
cat_juveniles = d2.cat_juveniles
cat_old = d2.cat_old
cat_adult = d2.cat_adult
cat = d.cat


formula = dict(main = 'height ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * cat',
            prior1 = 'sigma~Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,20)',
            prior3 = 'beta ~ Normal(0,1)')    

model = build_model(formula, df = d, sep = ',', float=32)

#%%
full_model = {'main': {'input': 'height ~ Normal(mu,sigma)', 'var': ['height', 'Normal', ['mu', 'sigma']]}, 'likelihood': {'input': 'mu ~ alpha + beta * cat', 'var': ['mu', ['alpha', 'beta', 'cat']]}, 'prior1': {'input': 'sigma~Uniform(0,50)', 'var': ['sigma', 'Uniform', ['0', '50']]}, 'prior2': {'input': 'alpha ~ Normal(178,20)', 'var': ['alpha', 'Normal', ['178', '20']]}, 'prior3': {'input': 'beta ~ Normal(0,1)', 'var': ['beta', 'Normal', ['0', '1']]}}

issues = get_undeclared_params(full_model, df = d)
# change df
newdf = d
if len(issues['params_in_data']):
    print(issues['params_in_data'])
    colCat = list(d.select_dtypes(['object']).columns)
    var_in_df = issues['params_in_data']
    for var in var_in_df:
        if is_string_dtype(d[var]):
            newdf = pd.get_dummies(newdf, columns=[var], dtype=int)

# Change model likelihood
new_param = list(newdf.columns.difference(d.columns))
old_param = list(d.columns.difference(newdf.columns))
new_param_with_coef = []
for a in range(len(new_param)):
    if a != 0 :
        new_param_with_coef.append('Bcat' + str(a) + ' * ' + new_param[a] + ' + ')
    else:
        new_param_with_coef.append(new_param[a] + ' + ')
for var in old_param:
    for key in full_model.keys():
        if 'likelihood' in key:
            input = full_model[key]['input']
            variables = full_model[key]['var']
            if var in input:
                print('Categorical variable in likelihood')
                full_model[key]['input'] = input.replace(var, ''.join(new_param_with_coef))
                full_model[key]['var'][1].remove(var)
                full_model[key]['var'][1] = full_model[key]['var'][1] + new_param

# Add new priors, but how to identify the prior linked to original cat?

print(full_model)               
        
## Find where old_param is
#%%
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Uniform(0, 50), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(178, 20), sample_shape=1),
	beta = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

	height = lambda alpha,beta, sigma: tfd.Independent(
        tfd.Normal(alpha+beta*cat, sigma), 
        reinterpreted_batch_ndims=1
    ),
))

#%%
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Uniform(0, 50), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(178, 20), sample_shape=1),
	beta = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
    beta2 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
    beta3 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

	height = lambda alpha,beta, beta2, beta3, sigma: tfd.Independent(
        tfd.Normal(alpha+beta*cat_adult + beta2*cat_juveniles +	beta3*cat_old, sigma), 
        reinterpreted_batch_ndims=1
    ),
))
m.sample()

#%% Model diag -----------------------------------------------------
plot_prior_dist(model)

model_check(posterior, trace, sample_stats, params = ['sigma', 'alpha', 'beta'])   

samples = model.sample(**posterior)
plt.plot(samples['height'][1].numpy(), weight)