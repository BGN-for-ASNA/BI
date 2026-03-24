import os
import sys
import pandas as pd
import jax.numpy as jnp
import jax

# Add BI to path
sys.path.append(os.path.abspath("."))

from BI import bi

print("Initializing BI with TFP backend...")
m = bi(platform='cpu', backend='tfp')

# Load data - using path from notebook context
data_path = os.path.join("BI", "Resources", "howell1.csv")
print(f"Loading data from {data_path}")
m.data(data_path, sep=';') 
m.df = m.df[m.df.age > 18]

# Model from notebook
def model(weight, height):
    a = yield m.dist.normal(178, 20)
    b = yield m.dist.log_normal(0, 1)  
    s = yield m.dist.uniform(0, 50)   
    y = yield m.dist.normal(a+b*weight, s, shape = (1,), obs = height)

print("Fitting model...")
try:
    m.fit(model = model, obs = 'height', num_chains = 1) 
    print("Summary:")
    print(m.summary())
except Exception as e:
    print("Execution failed!")
    print(e)
