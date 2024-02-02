#%% Test No data frame single likelihood -----------------------------------------------------
from  main import *
formula = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')   
self = model(formula= formula) 
self.sample(10)
print('tensor DICT:')
print(self.tensor_dict)
print('tensor likelihoods:')
print(self.main_text)


#%% Test No data frame multiple likelihood-----------------------------------------------------
from  main import *
formula = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)',
            
            main1 = 'z~Normal(m2,s2)',
            likelihood2 = 'm2 ~ alpha2 + beta2',
            prior4 = 's2~Exponential(1)',
            prior5 = 'alpha2 ~ Normal(0,1)',
            prior6 = 'beta2 ~ Normal(0,1)') 
self = model(formula= formula)
self.sample(10)
print('tensor DICT:')
print(self.tensor_dict)
print('tensor likelihoods:')
print(self.main_text)

#%% Test with data frame in likelihood-----------------------------------------------------
from  main import *
## Model m4.3
d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
d.age = d.age - d.age.mean()
formula = dict(main1 = 'height ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * weight',
            prior1 = 'sigma ~ Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,20)',
            prior3 = 'beta ~ Normal(0,1)')    

self = model(formula, df = d)
print('tensor DICT:')
print(self.tensor_dict)
print('tensor likelihoods:')
print(self.main_text)
self.fit(observed_data = dict(height =d.height.astype('float32').values),
                                           num_results = 2000, num_burnin_steps=500, num_adaptation_steps=400, num_chains=4)
self.summary()

# expected 
#           Mean    StdDev  5.5%    94.5%   a   b   sigma
#   a       154.60  0.27    154.17  155.03  1   0   0
#   b       0.91    0.04    0.84    0.97    0   1   0
#   sigma   5.07    0.19    4.77    5.38    0   0   1

# got:
#           mean	    sd	    hdi_5.5%	hdi_94.5%
#sigma[0]	5.14	    0.20	4.81	        5.43
#beta[0]	0.91	    0.04	0.84	        0.97
#alpha[0]	154.65	    0.28	154.22	        155.12
# Sigma and alph are inverted!!!!!!!!!!!!!!!!!!

# %% Indices -------------------------------------
## Model m5.9 
from  main import *
self = model()
self.import_csv('./data/milk.csv', sep = ';')
self.df["K"] = self.df["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
self.index(cols = "clade")

my_formula = dict(main = 'K ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha[index_clade]',
            prior1 = 'alpha~ Normal(0,0.5)',
            prior2 = 'sigma ~ Exponential(1)')  
self.formula(f = my_formula)
self.build_model()
self.fit(observed_data = dict(K =self.df.K.astype('float32').values),
                                           num_results = 2000, num_burnin_steps=500, num_adaptation_steps=400, num_chains=4)
self.summary()


# Expected:
#                mean	sd	    hdi_5.5%	hdi_94.5%
# ape_alpha	    -0.48	0.27	-0.93	    -0.08
# nwm_alpha	    0.37	0.22	0.03	    0.73
# owm_alpha	    0.65	0.30	0.26	    1.18
# strep_alpha	-0.55	0.29	-1.07	    -0.19
# sigma	        0.83	0.13	0.67	    1.06

# got:
#               mean	sd	    hdi_5.5%	hdi_94.5%
#sigma[0]	    0.80	0.12	0.61	    0.97
#alpha[0]ape	-0.47	0.24	-0.85	    -0.10
#alpha[1]nwm	0.35	0.24	0.00	    0.76
#alpha[2]owm	0.64	0.28	0.17	    1.07
#alpha[3]strep	-0.54	0.29	-1.00	    -0.07


#%%
# m8.1


#%% Test with multiple likelihood-----------------------------------------------------
from  main import *
formula = dict(main = 'z ~ Poisson(alpha)',
            likelihood = 'alpha ~ a + beta * weight',
            prior1 = 'a ~ Normal(178,20)',
            prior2 = 'beta ~ Normal(0,1)',
            
            main2 = 'y ~ Normal(mu, sigma)',
            likelihood2 = 'mu ~ alpha2 + beta * age',
            prior4 = 'sigma ~ Normal(0.0, 1.0)',
            prior5 = 'alpha2 ~ Normal(0,1)'
            )    

d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
d.age = d.age - d.age.mean()
self = model(formula,d)
print('tensor DICT:')
print(self.tensor_dict)
print('tensor likelihoods:')
print(self.main_text)

#%% Test  with data frame only as output-----------------------------------------------------
from  main import *
formula = dict(main = 'height ~ Normal(mu,sigma)',
            prior1 = 'mu ~ Normal(178.0, 0.1)',            
            prior2 = 'sigma ~ Uniform(0.0, 50.0)')    
d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
d.age = d.age - d.age.mean()
self = model(formula,d)

#%% Old code-------------------
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
        loc=tf.transpose(tf.gather(tf.transpose(alpha), 
                                   tf.cast(d.index_clade , dtype= tf.int32))),
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