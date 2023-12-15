# Import dependencies ----------------------------
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from code.model_fit import *
from code.model_diagnostic import *

# GPU configuration ----------------------------
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	m = tfd.Sample(tfd.Normal(0, 10), sample_shape=1),
	s = tfd.Sample(tfd.Normal(0, 1), sample_shape=1),

y = lambda m,s: tfd.Independent(tfd.Normal(m, s), reinterpreted_batch_ndims=1),
))