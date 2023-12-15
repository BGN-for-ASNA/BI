# Import dependencies ----------------------------
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
from model_fit import *
from model_diagnostic import *

# GPU configuration ----------------------------
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import data (with modification if maded) ----------------------------
d = pd.read_csv('mydf', sep = ',')
exposure= d.exposure
y= d.y
att= d.att

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	a = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	b = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	sigma = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	a2 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	b2 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	b3 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),
	sigma2 = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

	y = lambda a,b,exposureM,sigma: tfd.Independent(tfd.Normal(a+b*att+exposureM, sigma), reinterpreted_batch_ndims=1),

	exposureM = lambda a2,b2,b3,sigma2: tfd.Independent(tfd.Normal(a2+b2*att+b3*exposure, sigma2), reinterpreted_batch_ndims=1),
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
observed_data = dict(y = d.y.astype('float32').values,))