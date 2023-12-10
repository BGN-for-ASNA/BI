import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from model_fit import *
from model_diagnostic import *
m = tfd.JointDistributionNamed(dict(
	mu = tfd.Sample(tfd.Normal(178.0, 0.1), sample_shape=1),
	sigma = tfd.Sample(tfd.Uniform(0.0, 50.0), sample_shape=1),
	height = lambda mu,sigma: tfd.Independent(tfd.Normal(mu, sigma), reinterpreted_batch_ndims=1),
))