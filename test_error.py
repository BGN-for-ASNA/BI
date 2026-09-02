import os
os.environ["BF_NSIM"] = "1"
import sys
sys.path.insert(0, "/home/sosa/work/BF/Test/1. Rethinking/Numpyro backend")
from Utils import *
from BayesForge import bf
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

model_name = "2.Categorical variable"
m = bf(platform='cpu')
data_path = m.load.milk(only_path=True)
m.data(data_path, sep=';') 
m.index(["clade"])
m.scale(['kcal_per_g'])

def model_BF(kcal_per_g, index_clade):
    a = m.dist.normal(0, 0.5, shape=(4,))
    s = m.dist.exponential(1)    
    mu = a[index_clade]
    m.dist.normal(mu, s, obs=kcal_per_g)

m.data_to_model(['kcal_per_g', "index_clade"])
m.fit(model_BF) 

import arviz as az
try:
    az.from_numpyro(m.sampler, log_likelihood=True)
except Exception as e:
    import traceback
    traceback.print_exc()
