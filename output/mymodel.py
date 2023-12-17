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
cat= d.cat
height= d.height

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Uniform(0, 50), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(178, 20), sample_shape=1),
	beta = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

	height = lambda alpha,beta,sigma: tfd.Independent(tfd.Normal(alpha+beta*cat, sigma), reinterpreted_batch_ndims=1),
))