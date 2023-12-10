import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
from bayesian.distribution import dist
tfd = tfp.distributions
m = tfd.JointDistributionNamed(dict(
