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
d = pd.read_csv('output/mydf.csv')

# Set up parameters ----------------------------

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	s = tfd.Sample(tfd.Exponential(1), sample_shape= 1),
	alpha = tfd.Sample(tfd.Normal(0, 1), sample_shape= 1),
	beta = tfd.Sample(tfd.Normal(0, 1), sample_shape= 1),
	s2 = tfd.Sample(tfd.Exponential(1), sample_shape= 1),
	alpha2 = tfd.Sample(tfd.Normal(0, 1), sample_shape= 1),
	beta2 = tfd.Sample(tfd.Normal(0, 1), sample_shape= 1),
y = lambda s, alpha, beta: tfd.Independent(tfd.Normal(alpha+beta,s), reinterpreted_batch_ndims=1),
z = lambda s2, alpha2, beta2: tfd.Independent(tfd.Normal(alpha2+beta2,s2), reinterpreted_batch_ndims=1),
))