# Import dependencies ----------------------------
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
from code.model_fit import *
from code.model_diagnostic import *

# GPU configuration ----------------------------
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import data (with modification if maded) ----------------------------
d = pd.read_csv('output/mydf.csv', sep = ',')
clade_Old_World_Monkey= d.clade_Old_World_Monkey
clade_New_World_Monkey= d.clade_New_World_Monkey
clade_Strepsirrhine= d.clade_Strepsirrhine

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Uniform(0, 10), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(0.6, 10), sample_shape=1),
	B1 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	B2 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	B3 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

	y = lambda alpha,B1,B2,B3,sigma: tfd.Independent(tfd.Normal(alpha+B1*clade_New_World_Monkey+B2*clade_Old_World_Monkey+B3*clade_Strepsirrhine, sigma), reinterpreted_batch_ndims=1),
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
observed_data = dict(y = d.kcal_per_g.astype('float32').values,))