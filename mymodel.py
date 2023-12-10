import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
d = pd.read_csv('mydf',sep = ',')
weight= d.weight
height= d.height
m = tfd.JointDistributionNamed(dict(
	s = tfd.Sample(tfd.Exponential(1), sample_shape=1),
	alpha = tfd.Sample(tfd.Normal(0,1), sample_shape=1),
	beta = tfd.Sample(tfd.Normal(0,1), sample_shape=1),
	height = lambda s,alpha,beta: tfd.Independent(tfd.Normal( alpha + beta * weight,s), reinterpreted_batch_ndims=1),
))