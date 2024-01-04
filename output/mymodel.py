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
index_clade= d.index_clade
len_index_clade= len(set(d.index_clade.values))

# Model ----------------------------
m = tfd.JointDistributionNamed(dict(
	sigma = tfd.Sample(tfd.Exponential(1), sample_shape= 1),
	alpha = tfd.Sample(tfd.Normal(0, 0.5), sample_shape= len_index_clade),
