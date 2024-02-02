#%%
import time
start_time = time.time()
import tensorflow as tf
# GPU configuration ----------------------------
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#	tf.config.experimental.set_memory_growth(physical_devices[0], True)


import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
from model_fit import *
from model_diagnostic import *



# Import data (with modification if maded) ----------------------------
d = pd.read_csv('Howell1.csv', sep = ';')
height= d.height
weight= d.weight

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Uniform(tf.cast(0, tf.float16), tf.cast(50, tf.float16)), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(tf.cast(178, tf.float16), tf.cast(20, tf.float16)), sample_shape=1),
	beta = tfd.Sample(tfd.Normal(tf.cast(0, tf.float16), tf.cast(1, tf.float16)), sample_shape=1),

	height = lambda alpha,beta,sigma: 
    tfd.Independent(tfd.Normal(alpha+beta*weight, sigma), reinterpreted_batch_ndims=1),
))

# Run HMC ----------------------------
posterior, trace, sample_stats =  run_model(model = m,
parallel_iterations=1,
num_results=2000,
num_burnin_steps=500,
step_size=0.065,
num_leapfrog_steps=5,
num_adaptation_steps=400,
num_chains=4,
observed_data = dict(height = d.height.astype('float16').values,))
end_time = time.time()
print( end_time - start_time)
# %%
az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)