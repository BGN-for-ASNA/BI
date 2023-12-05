import tensorflow_probability as tfp
tfd = tfp.distributions
m = tfd.JointDistributionNamed(dict(
	s = tfd.Sample(tfd.Exponential(1)),
	alpha = tfd.Sample(tfd.Normal(0,1)),
	beta = tfd.Sample(tfd.Normal(0,1)),
	s2 = tfd.Sample(tfd.Exponential(1)),
	alpha2 = tfd.Sample(tfd.Normal(0,1)),
	beta2 = tfd.Sample(tfd.Normal(0,1)),
	y = lambda s,alpha,beta,s2,alpha2,beta2: tfd.Independent(tfd.Normal( alpha + beta,s)),
	z = lambda s,alpha,beta,s2,alpha2,beta2: tfd.Independent(tfd.Normal( alpha2 + beta2,s2)),
))