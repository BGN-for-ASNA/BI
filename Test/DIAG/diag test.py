#%%
from patch_diag import patch_diag_class
from BI import bi
m = bi(platform='cpu')

# Import Data & Data Manipulation ------------------------------------------------
# Import
from importlib.resources import files
data_path = m.load.howell1(only_path = True)
m.data(data_path, sep=';') 
m.df = m.df[m.df.age > 18] # Subset data to adults
m.scale(['weight']) # Normalize

# Define model ------------------------------------------------
def model(weight, height):    
    a = m.dist.normal(178, 20, name = 'a') 
    b = m.dist.log_normal(0, 1, name = 'b') 
    s = m.dist.uniform(0, 50, name = 's') 
    m.dist.normal(a + b * weight , s, obs = height) 

# Run mcmc ------------------------------------------------
m.fit(model, progress_bar=False)  # Optimize model parameters through MCMC sampling
#%%
from jax_diagnostics import *

# Compute diagnostics
summary(m.posteriors)
#%%
m.summary()
#%%
from jax_diagnostics import *

# Compute diagnostics
summary(m.posteriors)
#%%
m.summary()
# %%
from BI.Diagnostic.jax_summary import *
jax_summary(m.posteriors)


