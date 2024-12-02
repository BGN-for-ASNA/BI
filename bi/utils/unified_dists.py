import jax 
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import*
import tensorflow_probability.substrates.jax.distributions as tfd
tfb = tfp.bijectors
root = tfd.JointDistributionCoroutine.Root

class tfpLight:

    def __init__(self):
        pass

    @staticmethod
    def autocompositetensordistribution(dtype, reparameterization_type, validate_args, allow_nan_stats, parameters=None, graph_parents=None, name=None, shape=(), sample = False, seed = 0):
        """
        AutoCompositeTensorDistribution distribution.
    
        Arguments:
            dtype: <class 'inspect._empty'>
            reparameterization_type: <class 'inspect._empty'>
            validate_args: <class 'inspect._empty'>
            allow_nan_stats: <class 'inspect._empty'>
            parameters: None
            graph_parents: None
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.AutoCompositeTensorDistribution(dtype=dtype, reparameterization_type=reparameterization_type, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, graph_parents=graph_parents, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.AutoCompositeTensorDistribution(dtype=dtype, reparameterization_type=reparameterization_type, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, graph_parents=graph_parents, name=name), shape))

    @staticmethod
    def autoregressive(distribution_fn, sample0=None, num_steps=None, validate_args=False, allow_nan_stats=True, name='Autoregressive', shape=(), sample = False, seed = 0):
        """
        Autoregressive distribution.
    
        Arguments:
            distribution_fn: <class 'inspect._empty'>
            sample0: None
            num_steps: None
            validate_args: False
            allow_nan_stats: True
            name: Autoregressive
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Autoregressive(distribution_fn=distribution_fn, sample0=sample0, num_steps=num_steps, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Autoregressive(distribution_fn=distribution_fn, sample0=sample0, num_steps=num_steps, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def batchbroadcast(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        BatchBroadcast distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.BatchBroadcast(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.BatchBroadcast(args=args, kwargs=kwargs), shape))

    @staticmethod
    def batchconcat(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        BatchConcat distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.BatchConcat(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.BatchConcat(args=args, kwargs=kwargs), shape))

    @staticmethod
    def batchreshape(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        BatchReshape distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.BatchReshape(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.BatchReshape(args=args, kwargs=kwargs), shape))

    @staticmethod
    def bates(total_count, low=0.0, high=1.0, validate_args=False, allow_nan_stats=True, name='Bates', shape=(), sample = False, seed = 0):
        """
        Bates distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            low: 0.0
            high: 1.0
            validate_args: False
            allow_nan_stats: True
            name: Bates
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Bates(total_count=total_count, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Bates(total_count=total_count, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def bernoulli(logits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='Bernoulli', shape=(), sample = False, seed = 0):
        """
        Bernoulli distribution.
    
        Arguments:
            logits: None
            probs: None
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: Bernoulli
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Bernoulli(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Bernoulli(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def beta(concentration1, concentration0, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Beta', shape=(), sample = False, seed = 0):
        """
        Beta distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            force_probs_to_zero_outside_support: False
            name: Beta
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name), shape))

    @staticmethod
    def betabinomial(total_count, concentration1, concentration0, validate_args=False, allow_nan_stats=True, name='BetaBinomial', shape=(), sample = False, seed = 0):
        """
        BetaBinomial distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: BetaBinomial
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.BetaBinomial(total_count=total_count, concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.BetaBinomial(total_count=total_count, concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def betaquotient(concentration1_numerator, concentration0_numerator, concentration1_denominator, concentration0_denominator, validate_args=False, allow_nan_stats=True, name='BetaQuotient', shape=(), sample = False, seed = 0):
        """
        BetaQuotient distribution.
    
        Arguments:
            concentration1_numerator: <class 'inspect._empty'>
            concentration0_numerator: <class 'inspect._empty'>
            concentration1_denominator: <class 'inspect._empty'>
            concentration0_denominator: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: BetaQuotient
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.BetaQuotient(concentration1_numerator=concentration1_numerator, concentration0_numerator=concentration0_numerator, concentration1_denominator=concentration1_denominator, concentration0_denominator=concentration0_denominator, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.BetaQuotient(concentration1_numerator=concentration1_numerator, concentration0_numerator=concentration0_numerator, concentration1_denominator=concentration1_denominator, concentration0_denominator=concentration0_denominator, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def binomial(total_count, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name=None, shape=(), sample = False, seed = 0):
        """
        Binomial distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Binomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Binomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def blockwise(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        Blockwise distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Blockwise(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Blockwise(args=args, kwargs=kwargs), shape))

    @staticmethod
    def categorical(logits=None, probs=None, dtype=jax.numpy.int32, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Categorical', shape=(), sample = False, seed = 0):
        """
        Categorical distribution.
    
        Arguments:
            logits: None
            probs: None
            dtype: <class 'jax.numpy.int32'>
            force_probs_to_zero_outside_support: False
            validate_args: False
            allow_nan_stats: True
            name: Categorical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Categorical(logits=logits, probs=probs, dtype=dtype, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Categorical(logits=logits, probs=probs, dtype=dtype, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def cauchy(loc, scale, validate_args=False, allow_nan_stats=True, name='Cauchy', shape=(), sample = False, seed = 0):
        """
        Cauchy distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Cauchy
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Cauchy(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Cauchy(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def chi(df, validate_args=False, allow_nan_stats=True, name='Chi', shape=(), sample = False, seed = 0):
        """
        Chi distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Chi
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Chi(df=df, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Chi(df=df, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def chi2(df, validate_args=False, allow_nan_stats=True, name='Chi2', shape=(), sample = False, seed = 0):
        """
        Chi2 distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Chi2
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Chi2(df=df, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Chi2(df=df, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def choleskylkj(dimension, concentration, validate_args=False, allow_nan_stats=True, name='CholeskyLKJ', shape=(), sample = False, seed = 0):
        """
        CholeskyLKJ distribution.
    
        Arguments:
            dimension: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: CholeskyLKJ
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.CholeskyLKJ(dimension=dimension, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.CholeskyLKJ(dimension=dimension, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def continuousbernoulli(logits=None, probs=None, dtype = jax.numpy.float32, validate_args=False, allow_nan_stats=True, name='ContinuousBernoulli', shape=(), sample = False, seed = 0):
        """
        ContinuousBernoulli distribution.
    
        Arguments:
            logits: None
            probs: None
            dtype: <class 'jax.numpy.float32'>
            validate_args: False
            allow_nan_stats: True
            name: ContinuousBernoulli
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ContinuousBernoulli(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ContinuousBernoulli(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def determinantalpointprocess(eigenvalues, eigenvectors, validate_args=False, allow_nan_stats=False, name='DeterminantalPointProcess', shape=(), sample = False, seed = 0):
        """
        DeterminantalPointProcess distribution.
    
        Arguments:
            eigenvalues: <class 'inspect._empty'>
            eigenvectors: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: False
            name: DeterminantalPointProcess
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.DeterminantalPointProcess(eigenvalues=eigenvalues, eigenvectors=eigenvectors, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.DeterminantalPointProcess(eigenvalues=eigenvalues, eigenvectors=eigenvectors, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def deterministic(loc, atol=None, rtol=None, validate_args=False, allow_nan_stats=True, name='Deterministic', shape=(), sample = False, seed = 0):
        """
        Deterministic distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            atol: None
            rtol: None
            validate_args: False
            allow_nan_stats: True
            name: Deterministic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Deterministic(loc=loc, atol=atol, rtol=rtol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Deterministic(loc=loc, atol=atol, rtol=rtol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def dirichlet(concentration, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Dirichlet', shape=(), sample = False, seed = 0):
        """
        Dirichlet distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            force_probs_to_zero_outside_support: False
            name: Dirichlet
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Dirichlet(concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Dirichlet(concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name), shape))

    @staticmethod
    def dirichletmultinomial(total_count, concentration, validate_args=False, allow_nan_stats=True, name='DirichletMultinomial', shape=(), sample = False, seed = 0):
        """
        DirichletMultinomial distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: DirichletMultinomial
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.DirichletMultinomial(total_count=total_count, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.DirichletMultinomial(total_count=total_count, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def distribution(dtype, reparameterization_type, validate_args, allow_nan_stats, parameters=None, graph_parents=None, name=None, shape=(), sample = False, seed = 0):
        """
        Distribution distribution.
    
        Arguments:
            dtype: <class 'inspect._empty'>
            reparameterization_type: <class 'inspect._empty'>
            validate_args: <class 'inspect._empty'>
            allow_nan_stats: <class 'inspect._empty'>
            parameters: None
            graph_parents: None
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Distribution(dtype=dtype, reparameterization_type=reparameterization_type, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, graph_parents=graph_parents, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Distribution(dtype=dtype, reparameterization_type=reparameterization_type, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, graph_parents=graph_parents, name=name), shape))

    @staticmethod
    def doublesidedmaxwell(loc, scale, validate_args=False, allow_nan_stats=True, name='doublesided_maxwell', shape=(), sample = False, seed = 0):
        """
        DoublesidedMaxwell distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: doublesided_maxwell
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.DoublesidedMaxwell(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.DoublesidedMaxwell(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def empirical(samples, event_ndims=0, validate_args=False, allow_nan_stats=True, name='Empirical', shape=(), sample = False, seed = 0):
        """
        Empirical distribution.
    
        Arguments:
            samples: <class 'inspect._empty'>
            event_ndims: 0
            validate_args: False
            allow_nan_stats: True
            name: Empirical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Empirical(samples=samples, event_ndims=event_ndims, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Empirical(samples=samples, event_ndims=event_ndims, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def expgamma(concentration, rate=None, log_rate=None, validate_args=False, allow_nan_stats=True, name='ExpGamma', shape=(), sample = False, seed = 0):
        """
        ExpGamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: None
            log_rate: None
            validate_args: False
            allow_nan_stats: True
            name: ExpGamma
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ExpGamma(concentration=concentration, rate=rate, log_rate=log_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ExpGamma(concentration=concentration, rate=rate, log_rate=log_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def expinversegamma(concentration, scale=None, log_scale=None, validate_args=False, allow_nan_stats=True, name='ExpInverseGamma', shape=(), sample = False, seed = 0):
        """
        ExpInverseGamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale: None
            log_scale: None
            validate_args: False
            allow_nan_stats: True
            name: ExpInverseGamma
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ExpInverseGamma(concentration=concentration, scale=scale, log_scale=log_scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ExpInverseGamma(concentration=concentration, scale=scale, log_scale=log_scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def exprelaxedonehotcategorical(temperature, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='ExpRelaxedOneHotCategorical', shape=(), sample = False, seed = 0):
        """
        ExpRelaxedOneHotCategorical distribution.
    
        Arguments:
            temperature: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            name: ExpRelaxedOneHotCategorical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def exponential(rate, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Exponential', shape=(), sample = False, seed = 0):
        """
        Exponential distribution.
    
        Arguments:
            rate: <class 'inspect._empty'>
            force_probs_to_zero_outside_support: False
            validate_args: False
            allow_nan_stats: True
            name: Exponential
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Exponential(rate=rate, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Exponential(rate=rate, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def exponentiallymodifiedgaussian(loc, scale, rate, validate_args=False, allow_nan_stats=True, name='ExponentiallyModifiedGaussian', shape=(), sample = False, seed = 0):
        """
        ExponentiallyModifiedGaussian distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            rate: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: ExponentiallyModifiedGaussian
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ExponentiallyModifiedGaussian(loc=loc, scale=scale, rate=rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ExponentiallyModifiedGaussian(loc=loc, scale=scale, rate=rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def finitediscrete(outcomes, logits=None, probs=None, rtol=None, atol=None, validate_args=False, allow_nan_stats=True, name='FiniteDiscrete', shape=(), sample = False, seed = 0):
        """
        FiniteDiscrete distribution.
    
        Arguments:
            outcomes: <class 'inspect._empty'>
            logits: None
            probs: None
            rtol: None
            atol: None
            validate_args: False
            allow_nan_stats: True
            name: FiniteDiscrete
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.FiniteDiscrete(outcomes=outcomes, logits=logits, probs=probs, rtol=rtol, atol=atol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.FiniteDiscrete(outcomes=outcomes, logits=logits, probs=probs, rtol=rtol, atol=atol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def flatdirichlet(concentration_shape, dtype=jax.numpy.float32, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='FlatDirichlet', shape=(), sample = False, seed = 0):
        """
        FlatDirichlet distribution.
    
        Arguments:
            concentration_shape: <class 'inspect._empty'>
            dtype: <class 'jax.numpy.float32'>
            validate_args: False
            allow_nan_stats: True
            force_probs_to_zero_outside_support: False
            name: FlatDirichlet
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.FlatDirichlet(concentration_shape=concentration_shape, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.FlatDirichlet(concentration_shape=concentration_shape, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name), shape))

    @staticmethod
    def gamma(concentration, rate=None, log_rate=None, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Gamma', shape=(), sample = False, seed = 0):
        """
        Gamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: None
            log_rate: None
            validate_args: False
            allow_nan_stats: True
            force_probs_to_zero_outside_support: False
            name: Gamma
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Gamma(concentration=concentration, rate=rate, log_rate=log_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Gamma(concentration=concentration, rate=rate, log_rate=log_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, name=name), shape))

    @staticmethod
    def gammagamma(concentration, mixing_concentration, mixing_rate, validate_args=False, allow_nan_stats=True, name='GammaGamma', shape=(), sample = False, seed = 0):
        """
        GammaGamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            mixing_concentration: <class 'inspect._empty'>
            mixing_rate: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: GammaGamma
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GammaGamma(concentration=concentration, mixing_concentration=mixing_concentration, mixing_rate=mixing_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GammaGamma(concentration=concentration, mixing_concentration=mixing_concentration, mixing_rate=mixing_rate, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def gaussianprocess(kernel, index_points=None, mean_fn=None, observation_noise_variance=0.0, marginal_fn=None, cholesky_fn=None, jitter=1e-06, validate_args=False, allow_nan_stats=False, parameters=None, name='GaussianProcess', _check_marginal_cholesky_fn=True, shape=(), sample = False, seed = 0):
        """
        GaussianProcess distribution.
    
        Arguments:
            kernel: <class 'inspect._empty'>
            index_points: None
            mean_fn: None
            observation_noise_variance: 0.0
            marginal_fn: None
            cholesky_fn: None
            jitter: 1e-06
            validate_args: False
            allow_nan_stats: False
            parameters: None
            name: GaussianProcess
            _check_marginal_cholesky_fn: True
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GaussianProcess(kernel=kernel, index_points=index_points, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, marginal_fn=marginal_fn, cholesky_fn=cholesky_fn, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, name=name, _check_marginal_cholesky_fn=_check_marginal_cholesky_fn).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GaussianProcess(kernel=kernel, index_points=index_points, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, marginal_fn=marginal_fn, cholesky_fn=cholesky_fn, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, parameters=parameters, name=name, _check_marginal_cholesky_fn=_check_marginal_cholesky_fn), shape))

    @staticmethod
    def gaussianprocessregressionmodel(kernel, index_points=None, observation_index_points=None, observations=None, observation_noise_variance=0.0, predictive_noise_variance=None, mean_fn=None, cholesky_fn=None, jitter=1e-06, validate_args=False, allow_nan_stats=False, name='GaussianProcessRegressionModel', _conditional_kernel=None, _conditional_mean_fn=None, shape=(), sample = False, seed = 0):
        """
        GaussianProcessRegressionModel distribution.
    
        Arguments:
            kernel: <class 'inspect._empty'>
            index_points: None
            observation_index_points: None
            observations: None
            observation_noise_variance: 0.0
            predictive_noise_variance: None
            mean_fn: None
            cholesky_fn: None
            jitter: 1e-06
            validate_args: False
            allow_nan_stats: False
            name: GaussianProcessRegressionModel
            _conditional_kernel: None
            _conditional_mean_fn: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GaussianProcessRegressionModel(kernel=kernel, index_points=index_points, observation_index_points=observation_index_points, observations=observations, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, mean_fn=mean_fn, cholesky_fn=cholesky_fn, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name, _conditional_kernel=_conditional_kernel, _conditional_mean_fn=_conditional_mean_fn).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GaussianProcessRegressionModel(kernel=kernel, index_points=index_points, observation_index_points=observation_index_points, observations=observations, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, mean_fn=mean_fn, cholesky_fn=cholesky_fn, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name, _conditional_kernel=_conditional_kernel, _conditional_mean_fn=_conditional_mean_fn), shape))

    @staticmethod
    def generalizedextremevalue(loc, scale, concentration, validate_args=False, allow_nan_stats=True, name='GeneralizedExtremeValue', shape=(), sample = False, seed = 0):
        """
        GeneralizedExtremeValue distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: GeneralizedExtremeValue
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def generalizednormal(loc, scale, power, validate_args=False, allow_nan_stats=True, name='GeneralizedNormal', shape=(), sample = False, seed = 0):
        """
        GeneralizedNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            power: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: GeneralizedNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GeneralizedNormal(loc=loc, scale=scale, power=power, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GeneralizedNormal(loc=loc, scale=scale, power=power, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def generalizedpareto(loc, scale, concentration, validate_args=False, allow_nan_stats=True, name=None, shape=(), sample = False, seed = 0):
        """
        GeneralizedPareto distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.GeneralizedPareto(loc=loc, scale=scale, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.GeneralizedPareto(loc=loc, scale=scale, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def geometric(logits=None, probs=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Geometric', shape=(), sample = False, seed = 0):
        """
        Geometric distribution.
    
        Arguments:
            logits: None
            probs: None
            force_probs_to_zero_outside_support: False
            validate_args: False
            allow_nan_stats: True
            name: Geometric
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Geometric(logits=logits, probs=probs, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Geometric(logits=logits, probs=probs, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def gumbel(loc, scale, validate_args=False, allow_nan_stats=True, name='Gumbel', shape=(), sample = False, seed = 0):
        """
        Gumbel distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Gumbel
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Gumbel(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Gumbel(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def halfcauchy(loc, scale, validate_args=False, allow_nan_stats=True, name='HalfCauchy', shape=(), sample = False, seed = 0):
        """
        HalfCauchy distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: HalfCauchy
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.HalfCauchy(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.HalfCauchy(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def halfnormal(scale, validate_args=False, allow_nan_stats=True, name='HalfNormal', shape=(), sample = False, seed = 0):
        """
        HalfNormal distribution.
    
        Arguments:
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: HalfNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.HalfNormal(scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.HalfNormal(scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def halfstudentt(df, loc, scale, validate_args=False, allow_nan_stats=True, name='HalfStudentT', shape=(), sample = False, seed = 0):
        """
        HalfStudentT distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: HalfStudentT
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.HalfStudentT(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.HalfStudentT(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def hiddenmarkovmodel(initial_distribution, transition_distribution, observation_distribution, num_steps, validate_args=False, allow_nan_stats=True, time_varying_transition_distribution=False, time_varying_observation_distribution=False, mask=None, name='HiddenMarkovModel', shape=(), sample = False, seed = 0):
        """
        HiddenMarkovModel distribution.
    
        Arguments:
            initial_distribution: <class 'inspect._empty'>
            transition_distribution: <class 'inspect._empty'>
            observation_distribution: <class 'inspect._empty'>
            num_steps: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            time_varying_transition_distribution: False
            time_varying_observation_distribution: False
            mask: None
            name: HiddenMarkovModel
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.HiddenMarkovModel(initial_distribution=initial_distribution, transition_distribution=transition_distribution, observation_distribution=observation_distribution, num_steps=num_steps, validate_args=validate_args, allow_nan_stats=allow_nan_stats, time_varying_transition_distribution=time_varying_transition_distribution, time_varying_observation_distribution=time_varying_observation_distribution, mask=mask, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.HiddenMarkovModel(initial_distribution=initial_distribution, transition_distribution=transition_distribution, observation_distribution=observation_distribution, num_steps=num_steps, validate_args=validate_args, allow_nan_stats=allow_nan_stats, time_varying_transition_distribution=time_varying_transition_distribution, time_varying_observation_distribution=time_varying_observation_distribution, mask=mask, name=name), shape))

    @staticmethod
    def horseshoe(scale, validate_args=False, allow_nan_stats=True, name='Horseshoe', shape=(), sample = False, seed = 0):
        """
        Horseshoe distribution.
    
        Arguments:
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Horseshoe
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Horseshoe(scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Horseshoe(scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def independent( shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        Independent distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Independent(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Independent(args=args, kwargs=kwargs), shape))

    @staticmethod
    def inflated(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        Inflated distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Inflated(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Inflated(args=args, kwargs=kwargs), shape))

    @staticmethod
    def inversegamma(concentration, scale=None, validate_args=False, allow_nan_stats=True, name='InverseGamma', shape=(), sample = False, seed = 0):
        """
        InverseGamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale: None
            validate_args: False
            allow_nan_stats: True
            name: InverseGamma
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.InverseGamma(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.InverseGamma(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def inversegaussian(loc, concentration, validate_args=False, allow_nan_stats=True, name='InverseGaussian', shape=(), sample = False, seed = 0):
        """
        InverseGaussian distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: InverseGaussian
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.InverseGaussian(loc=loc, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.InverseGaussian(loc=loc, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def johnsonsu(skewness, tailweight, loc, scale, validate_args=False, allow_nan_stats=True, name=None, shape=(), sample = False, seed = 0):
        """
        JohnsonSU distribution.
    
        Arguments:
            skewness: <class 'inspect._empty'>
            tailweight: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JohnsonSU(skewness=skewness, tailweight=tailweight, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def jointdistribution(dtype, validate_args, parameters, name, use_vectorized_map=False, batch_ndims=None, experimental_use_kahan_sum=False, shape=(), sample = False, seed = 0):
        """
        JointDistribution distribution.
    
        Arguments:
            dtype: <class 'inspect._empty'>
            validate_args: <class 'inspect._empty'>
            parameters: <class 'inspect._empty'>
            name: <class 'inspect._empty'>
            use_vectorized_map: False
            batch_ndims: None
            experimental_use_kahan_sum: False
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistribution(dtype=dtype, validate_args=validate_args, parameters=parameters, name=name, use_vectorized_map=use_vectorized_map, batch_ndims=batch_ndims, experimental_use_kahan_sum=experimental_use_kahan_sum).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistribution(dtype=dtype, validate_args=validate_args, parameters=parameters, name=name, use_vectorized_map=use_vectorized_map, batch_ndims=batch_ndims, experimental_use_kahan_sum=experimental_use_kahan_sum), shape))

    @staticmethod
    def jointdistributioncoroutine(model, sample_dtype=None, batch_ndims=None, use_vectorized_map=False, validate_args=False, experimental_use_kahan_sum=False, name=None, shape=(), sample = False, seed = 0):
        """
        JointDistributionCoroutine distribution.
    
        Arguments:
            model: <class 'inspect._empty'>
            sample_dtype: None
            batch_ndims: None
            use_vectorized_map: False
            validate_args: False
            experimental_use_kahan_sum: False
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionCoroutine(model=model, sample_dtype=sample_dtype, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map, validate_args=validate_args, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionCoroutine(model=model, sample_dtype=sample_dtype, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map, validate_args=validate_args, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name), shape))

    @staticmethod
    def jointdistributioncoroutineautobatched(model, sample_dtype=None, batch_ndims=0, use_vectorized_map=True, validate_args=False, experimental_use_kahan_sum=False, name=None, shape=(), sample = False, seed = 0):
        """
        JointDistributionCoroutineAutoBatched distribution.
    
        Arguments:
            model: <class 'inspect._empty'>
            sample_dtype: None
            batch_ndims: 0
            use_vectorized_map: True
            validate_args: False
            experimental_use_kahan_sum: False
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionCoroutineAutoBatched(model=model, sample_dtype=sample_dtype, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map, validate_args=validate_args, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionCoroutineAutoBatched(model=model, sample_dtype=sample_dtype, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map, validate_args=validate_args, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name), shape))

    @staticmethod
    def jointdistributionnamed(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        JointDistributionNamed distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionNamed(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionNamed(args=args, kwargs=kwargs), shape))

    @staticmethod
    def jointdistributionnamedautobatched( shape=(), sample = False, seed = 0, *args, **kwargs,):
        """
        JointDistributionNamedAutoBatched distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionNamedAutoBatched(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionNamedAutoBatched(args=args, kwargs=kwargs), shape))

    @staticmethod
    def jointdistributionsequential( shape=(), sample = False, seed = 0, *args, **kwarg):
        """
        JointDistributionSequential distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionSequential(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionSequential(args=args, kwargs=kwargs), shape))

    @staticmethod
    def jointdistributionsequentialautobatched(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        JointDistributionSequentialAutoBatched distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.JointDistributionSequentialAutoBatched(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.JointDistributionSequentialAutoBatched(args=args, kwargs=kwargs), shape))

    @staticmethod
    def kumaraswamy(concentration1=1.0, concentration0=1.0, validate_args=False, allow_nan_stats=True, name='Kumaraswamy', shape=(), sample = False, seed = 0):
        """
        Kumaraswamy distribution.
    
        Arguments:
            concentration1: 1.0
            concentration0: 1.0
            validate_args: False
            allow_nan_stats: True
            name: Kumaraswamy
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def lkj(dimension, concentration, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='LKJ', shape=(), sample = False, seed = 0):
        """
        LKJ distribution.
    
        Arguments:
            dimension: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            input_output_cholesky: False
            validate_args: False
            allow_nan_stats: True
            name: LKJ
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LKJ(dimension=dimension, concentration=concentration, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LKJ(dimension=dimension, concentration=concentration, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def lambertwdistribution(distribution, shift, scale, tailweight=None, validate_args=False, allow_nan_stats=True, name='LambertWDistribution', shape=(), sample = False, seed = 0):
        """
        LambertWDistribution distribution.
    
        Arguments:
            distribution: <class 'inspect._empty'>
            shift: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            tailweight: None
            validate_args: False
            allow_nan_stats: True
            name: LambertWDistribution
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LambertWDistribution(distribution=distribution, shift=shift, scale=scale, tailweight=tailweight, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LambertWDistribution(distribution=distribution, shift=shift, scale=scale, tailweight=tailweight, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def lambertwnormal(loc, scale, tailweight=None, validate_args=False, allow_nan_stats=True, name='LambertWNormal', shape=(), sample = False, seed = 0):
        """
        LambertWNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            tailweight: None
            validate_args: False
            allow_nan_stats: True
            name: LambertWNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LambertWNormal(loc=loc, scale=scale, tailweight=tailweight, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def laplace(loc, scale, validate_args=False, allow_nan_stats=True, name='Laplace', shape=(), sample = False, seed = 0):
        """
        Laplace distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Laplace
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Laplace(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Laplace(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def lineargaussianstatespacemodel(num_timesteps, transition_matrix, transition_noise, observation_matrix, observation_noise, initial_state_prior, initial_step=0, mask=None, experimental_parallelize=False, validate_args=False, allow_nan_stats=True, name='LinearGaussianStateSpaceModel', shape=(), sample = False, seed = 0):
        """
        LinearGaussianStateSpaceModel distribution.
    
        Arguments:
            num_timesteps: <class 'inspect._empty'>
            transition_matrix: <class 'inspect._empty'>
            transition_noise: <class 'inspect._empty'>
            observation_matrix: <class 'inspect._empty'>
            observation_noise: <class 'inspect._empty'>
            initial_state_prior: <class 'inspect._empty'>
            initial_step: 0
            mask: None
            experimental_parallelize: False
            validate_args: False
            allow_nan_stats: True
            name: LinearGaussianStateSpaceModel
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LinearGaussianStateSpaceModel(num_timesteps=num_timesteps, transition_matrix=transition_matrix, transition_noise=transition_noise, observation_matrix=observation_matrix, observation_noise=observation_noise, initial_state_prior=initial_state_prior, initial_step=initial_step, mask=mask, experimental_parallelize=experimental_parallelize, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LinearGaussianStateSpaceModel(num_timesteps=num_timesteps, transition_matrix=transition_matrix, transition_noise=transition_noise, observation_matrix=observation_matrix, observation_noise=observation_noise, initial_state_prior=initial_state_prior, initial_step=initial_step, mask=mask, experimental_parallelize=experimental_parallelize, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def loglogistic(loc, scale, validate_args=False, allow_nan_stats=True, name='LogLogistic', shape=(), sample = False, seed = 0):
        """
        LogLogistic distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: LogLogistic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LogLogistic(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LogLogistic(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def lognormal(loc, scale, validate_args=False, allow_nan_stats=True, name='LogNormal', shape=(), sample = False, seed = 0):
        """
        LogNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: LogNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LogNormal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LogNormal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def logistic(loc, scale, validate_args=False, allow_nan_stats=True, name='Logistic', shape=(), sample = False, seed = 0):
        """
        Logistic distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Logistic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Logistic(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Logistic(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def logitnormal(loc, scale, num_probit_terms_approx=2, gauss_hermite_scale_limit=None, gauss_hermite_degree=20, validate_args=False, allow_nan_stats=True, name='LogitNormal', shape=(), sample = False, seed = 0):
        """
        LogitNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            num_probit_terms_approx: 2
            gauss_hermite_scale_limit: None
            gauss_hermite_degree: 20
            validate_args: False
            allow_nan_stats: True
            name: LogitNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.LogitNormal(loc=loc, scale=scale, num_probit_terms_approx=num_probit_terms_approx, gauss_hermite_scale_limit=gauss_hermite_scale_limit, gauss_hermite_degree=gauss_hermite_degree, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.LogitNormal(loc=loc, scale=scale, num_probit_terms_approx=num_probit_terms_approx, gauss_hermite_scale_limit=gauss_hermite_scale_limit, gauss_hermite_degree=gauss_hermite_degree, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def markovchain(initial_state_prior, transition_fn, num_steps, experimental_use_kahan_sum=False, validate_args=False, name='MarkovChain', shape=(), sample = False, seed = 0):
        """
        MarkovChain distribution.
    
        Arguments:
            initial_state_prior: <class 'inspect._empty'>
            transition_fn: <class 'inspect._empty'>
            num_steps: <class 'inspect._empty'>
            experimental_use_kahan_sum: False
            validate_args: False
            name: MarkovChain
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MarkovChain(initial_state_prior=initial_state_prior, transition_fn=transition_fn, num_steps=num_steps, experimental_use_kahan_sum=experimental_use_kahan_sum, validate_args=validate_args, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MarkovChain(initial_state_prior=initial_state_prior, transition_fn=transition_fn, num_steps=num_steps, experimental_use_kahan_sum=experimental_use_kahan_sum, validate_args=validate_args, name=name), shape))

    @staticmethod
    def masked( shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        Masked distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Masked(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Masked(args=args, kwargs=kwargs), shape))

    @staticmethod
    def matrixnormallinearoperator(loc, scale_row, scale_column, validate_args=False, allow_nan_stats=True, name='MatrixNormalLinearOperator', shape=(), sample = False, seed = 0):
        """
        MatrixNormalLinearOperator distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale_row: <class 'inspect._empty'>
            scale_column: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: MatrixNormalLinearOperator
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MatrixNormalLinearOperator(loc=loc, scale_row=scale_row, scale_column=scale_column, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MatrixNormalLinearOperator(loc=loc, scale_row=scale_row, scale_column=scale_column, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def matrixtlinearoperator(df, loc, scale_row, scale_column, validate_args=False, allow_nan_stats=True, name='MatrixTLinearOperator', shape=(), sample = False, seed = 0):
        """
        MatrixTLinearOperator distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale_row: <class 'inspect._empty'>
            scale_column: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: MatrixTLinearOperator
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MatrixTLinearOperator(df=df, loc=loc, scale_row=scale_row, scale_column=scale_column, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MatrixTLinearOperator(df=df, loc=loc, scale_row=scale_row, scale_column=scale_column, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def mixture(shape=(), sample = False, seed = 0, *args, **kwargs, ):
        """
        Mixture distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Mixture(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Mixture(args=args, kwargs=kwargs), shape))

    @staticmethod
    def mixturesamefamily(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        MixtureSameFamily distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MixtureSameFamily(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MixtureSameFamily(args=args, kwargs=kwargs), shape))

    @staticmethod
    def moyal(loc, scale, validate_args=False, allow_nan_stats=True, name='Moyal', shape=(), sample = False, seed = 0):
        """
        Moyal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Moyal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Moyal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Moyal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def multinomial(total_count, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='Multinomial', shape=(), sample = False, seed = 0):
        """
        Multinomial distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            name: Multinomial
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Multinomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Multinomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def multivariatenormaldiag(loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalDiag', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalDiag distribution.
    
        Arguments:
            loc: None
            scale_diag: None
            validate_args: False
            allow_nan_stats: True
            experimental_use_kahan_sum: False
            name: MultivariateNormalDiag
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name), shape))

    @staticmethod
    def multivariatenormaldiagpluslowrank(loc=None, scale_diag=None, scale_perturb_factor=None, scale_perturb_diag=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiagPlusLowRank', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalDiagPlusLowRank distribution.
    
        Arguments:
            loc: None
            scale_diag: None
            scale_perturb_factor: None
            scale_perturb_diag: None
            validate_args: False
            allow_nan_stats: True
            name: MultivariateNormalDiagPlusLowRank
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalDiagPlusLowRank(loc=loc, scale_diag=scale_diag, scale_perturb_factor=scale_perturb_factor, scale_perturb_diag=scale_perturb_diag, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalDiagPlusLowRank(loc=loc, scale_diag=scale_diag, scale_perturb_factor=scale_perturb_factor, scale_perturb_diag=scale_perturb_diag, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def multivariatenormaldiagpluslowrankcovariance(loc=None, cov_diag_factor=None, cov_perturb_factor=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiagPlusLowRankCovariance', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalDiagPlusLowRankCovariance distribution.
    
        Arguments:
            loc: None
            cov_diag_factor: None
            cov_perturb_factor: None
            validate_args: False
            allow_nan_stats: True
            name: MultivariateNormalDiagPlusLowRankCovariance
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalDiagPlusLowRankCovariance(loc=loc, cov_diag_factor=cov_diag_factor, cov_perturb_factor=cov_perturb_factor, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalDiagPlusLowRankCovariance(loc=loc, cov_diag_factor=cov_diag_factor, cov_perturb_factor=cov_perturb_factor, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def multivariatenormalfullcovariance(loc=None, covariance_matrix=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalFullCovariance', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalFullCovariance distribution.
    
        Arguments:
            loc: None
            covariance_matrix: None
            validate_args: False
            allow_nan_stats: True
            name: MultivariateNormalFullCovariance
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=covariance_matrix, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=covariance_matrix, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def multivariatenormallinearoperator(loc=None, scale=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalLinearOperator', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalLinearOperator distribution.
    
        Arguments:
            loc: None
            scale: None
            validate_args: False
            allow_nan_stats: True
            experimental_use_kahan_sum: False
            name: MultivariateNormalLinearOperator
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalLinearOperator(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalLinearOperator(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name), shape))

    @staticmethod
    def multivariatenormaltril(loc=None, scale_tril=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalTriL', shape=(), sample = False, seed = 0):
        """
        MultivariateNormalTriL distribution.
    
        Arguments:
            loc: None
            scale_tril: None
            validate_args: False
            allow_nan_stats: True
            experimental_use_kahan_sum: False
            name: MultivariateNormalTriL
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril, validate_args=validate_args, allow_nan_stats=allow_nan_stats, experimental_use_kahan_sum=experimental_use_kahan_sum, name=name), shape))

    @staticmethod
    def multivariatestudenttlinearoperator(df, loc, scale, validate_args=False, allow_nan_stats=True, name='MultivariateStudentTLinearOperator', shape=(), sample = False, seed = 0):
        """
        MultivariateStudentTLinearOperator distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: MultivariateStudentTLinearOperator
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.MultivariateStudentTLinearOperator(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.MultivariateStudentTLinearOperator(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def negativebinomial(total_count, logits=None, probs=None, validate_args=False, allow_nan_stats=True, require_integer_total_count=True, name='NegativeBinomial', shape=(), sample = False, seed = 0):
        """
        NegativeBinomial distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            require_integer_total_count: True
            name: NegativeBinomial
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.NegativeBinomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, require_integer_total_count=require_integer_total_count, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.NegativeBinomial(total_count=total_count, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, require_integer_total_count=require_integer_total_count, name=name), shape))

    @staticmethod
    def noncentralchi2(df, noncentrality, validate_args=False, allow_nan_stats=True, name='NoncentralChi2', shape=(), sample = False, seed = 0):
        """
        NoncentralChi2 distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            noncentrality: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: NoncentralChi2
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.NoncentralChi2(df=df, noncentrality=noncentrality, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.NoncentralChi2(df=df, noncentrality=noncentrality, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def normal(loc, scale, validate_args=False, allow_nan_stats=True, name='Normal', shape=(), sample = False, seed = 0):
        """
        Normal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Normal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Normal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Normal(loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def normalinversegaussian(loc, scale, tailweight, skewness, validate_args=False, allow_nan_stats=True, name='NormalInverseGaussian', shape=(), sample = False, seed = 0):
        """
        NormalInverseGaussian distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            tailweight: <class 'inspect._empty'>
            skewness: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: NormalInverseGaussian
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.NormalInverseGaussian(loc=loc, scale=scale, tailweight=tailweight, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.NormalInverseGaussian(loc=loc, scale=scale, tailweight=tailweight, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def onehotcategorical(logits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='OneHotCategorical', shape=(), sample = False, seed = 0):
        """
        OneHotCategorical distribution.
    
        Arguments:
            logits: None
            probs: None
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: OneHotCategorical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.OneHotCategorical(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.OneHotCategorical(logits=logits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def orderedlogistic(cutpoints, loc, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='OrderedLogistic', shape=(), sample = False, seed = 0):
        """
        OrderedLogistic distribution.
    
        Arguments:
            cutpoints: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: OrderedLogistic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def pert(low, peak, high, temperature=4.0, validate_args=False, allow_nan_stats=False, name='PERT', shape=(), sample = False, seed = 0):
        """
        PERT distribution.
    
        Arguments:
            low: <class 'inspect._empty'>
            peak: <class 'inspect._empty'>
            high: <class 'inspect._empty'>
            temperature: 4.0
            validate_args: False
            allow_nan_stats: False
            name: PERT
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.PERT(low=low, peak=peak, high=high, temperature=temperature, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.PERT(low=low, peak=peak, high=high, temperature=temperature, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def pareto(concentration, scale=1.0, validate_args=False, allow_nan_stats=True, name='Pareto', shape=(), sample = False, seed = 0):
        """
        Pareto distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale: 1.0
            validate_args: False
            allow_nan_stats: True
            name: Pareto
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Pareto(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Pareto(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def plackettluce(scores, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='PlackettLuce', shape=(), sample = False, seed = 0):
        """
        PlackettLuce distribution.
    
        Arguments:
            scores: <class 'inspect._empty'>
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: PlackettLuce
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.PlackettLuce(scores=scores, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.PlackettLuce(scores=scores, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def poisson(rate=None, log_rate=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Poisson', shape=(), sample = False, seed = 0):
        """
        Poisson distribution.
    
        Arguments:
            rate: None
            log_rate: None
            force_probs_to_zero_outside_support: False
            validate_args: False
            allow_nan_stats: True
            name: Poisson
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Poisson(rate=rate, log_rate=log_rate, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Poisson(rate=rate, log_rate=log_rate, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def poissonlognormalquadraturecompound(loc, scale, quadrature_size=8, quadrature_fn=tfp.distributions.quadrature_scheme_lognormal_quantiles, validate_args=False, allow_nan_stats=True, name='PoissonLogNormalQuadratureCompound', shape=(), sample = False, seed = 0):
        """
        PoissonLogNormalQuadratureCompound distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            quadrature_size: 8
            quadrature_fn: <function quadrature_scheme_lognormal_quantiles at 0x7f4a8be1b370>
            validate_args: False
            allow_nan_stats: True
            name: PoissonLogNormalQuadratureCompound
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.PoissonLogNormalQuadratureCompound(loc=loc, scale=scale, quadrature_size=quadrature_size, quadrature_fn=quadrature_fn, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.PoissonLogNormalQuadratureCompound(loc=loc, scale=scale, quadrature_size=quadrature_size, quadrature_fn=quadrature_fn, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def powerspherical(mean_direction, concentration, validate_args=False, allow_nan_stats=True, name='PowerSpherical', shape=(), sample = False, seed = 0):
        """
        PowerSpherical distribution.
    
        Arguments:
            mean_direction: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: PowerSpherical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.PowerSpherical(mean_direction=mean_direction, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.PowerSpherical(mean_direction=mean_direction, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def probitbernoulli(probits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='ProbitBernoulli', shape=(), sample = False, seed = 0):
        """
        ProbitBernoulli distribution.
    
        Arguments:
            probits: None
            probs: None
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: ProbitBernoulli
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ProbitBernoulli(probits=probits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ProbitBernoulli(probits=probits, probs=probs, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def quantizeddistribution( shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        QuantizedDistribution distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.QuantizedDistribution(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.QuantizedDistribution(args=args, kwargs=kwargs), shape))

    @staticmethod
    def registerkl(dist_cls_a, dist_cls_b, shape=(), sample = False, seed = 0):
        """
        RegisterKL distribution.
    
        Arguments:
            dist_cls_a: <class 'inspect._empty'>
            dist_cls_b: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.RegisterKL(dist_cls_a=dist_cls_a, dist_cls_b=dist_cls_b).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.RegisterKL(dist_cls_a=dist_cls_a, dist_cls_b=dist_cls_b), shape))

    @staticmethod
    def relaxedbernoulli(temperature, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='RelaxedBernoulli', shape=(), sample = False, seed = 0):
        """
        RelaxedBernoulli distribution.
    
        Arguments:
            temperature: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            name: RelaxedBernoulli
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.RelaxedBernoulli(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.RelaxedBernoulli(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def relaxedonehotcategorical(temperature, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='RelaxedOneHotCategorical', shape=(), sample = False, seed = 0):
        """
        RelaxedOneHotCategorical distribution.
    
        Arguments:
            temperature: <class 'inspect._empty'>
            logits: None
            probs: None
            validate_args: False
            allow_nan_stats: True
            name: RelaxedOneHotCategorical
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.RelaxedOneHotCategorical(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.RelaxedOneHotCategorical(temperature=temperature, logits=logits, probs=probs, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def reparameterizationtype(rep_type, shape=(), sample = False, seed = 0):
        """
        ReparameterizationType distribution.
    
        Arguments:
            rep_type: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ReparameterizationType(rep_type=rep_type).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ReparameterizationType(rep_type=rep_type), shape))

    @staticmethod
    def sample(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        Sample distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Sample(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Sample(args=args, kwargs=kwargs), shape))

    @staticmethod
    def sigmoidbeta(concentration1, concentration0, validate_args=False, allow_nan_stats=True, name='SigmoidBeta', shape=(), sample = False, seed = 0):
        """
        SigmoidBeta distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: SigmoidBeta
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.SigmoidBeta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.SigmoidBeta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def sinharcsinh(loc, scale, skewness=None, tailweight=None, distribution=None, validate_args=False, allow_nan_stats=True, name='SinhArcsinh', shape=(), sample = False, seed = 0):
        """
        SinhArcsinh distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            skewness: None
            tailweight: None
            distribution: None
            validate_args: False
            allow_nan_stats: True
            name: SinhArcsinh
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.SinhArcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight, distribution=distribution, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.SinhArcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight, distribution=distribution, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def skellam(rate1=None, rate2=None, log_rate1=None, log_rate2=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Skellam', shape=(), sample = False, seed = 0):
        """
        Skellam distribution.
    
        Arguments:
            rate1: None
            rate2: None
            log_rate1: None
            log_rate2: None
            force_probs_to_zero_outside_support: False
            validate_args: False
            allow_nan_stats: True
            name: Skellam
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Skellam(rate1=rate1, rate2=rate2, log_rate1=log_rate1, log_rate2=log_rate2, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Skellam(rate1=rate1, rate2=rate2, log_rate1=log_rate1, log_rate2=log_rate2, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def sphericaluniform(dimension, batch_shape=(), dtype=jax.numpy.float32, validate_args=False, allow_nan_stats=True, name='SphericalUniform', shape=(), sample = False, seed = 0):
        """
        SphericalUniform distribution.
    
        Arguments:
            dimension: <class 'inspect._empty'>
            batch_shape: ()
            dtype: <class 'jax.numpy.float32'>
            validate_args: False
            allow_nan_stats: True
            name: SphericalUniform
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.SphericalUniform(dimension=dimension, batch_shape=batch_shape, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.SphericalUniform(dimension=dimension, batch_shape=batch_shape, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def stoppingratiologistic(cutpoints, loc, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='StoppingRatioLogistic', shape=(), sample = False, seed = 0):
        """
        StoppingRatioLogistic distribution.
    
        Arguments:
            cutpoints: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            dtype: <class 'jax.numpy.int32'>
            validate_args: False
            allow_nan_stats: True
            name: StoppingRatioLogistic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.StoppingRatioLogistic(cutpoints=cutpoints, loc=loc, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.StoppingRatioLogistic(cutpoints=cutpoints, loc=loc, dtype=dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def studentt(df, loc, scale, validate_args=False, allow_nan_stats=True, name='StudentT', shape=(), sample = False, seed = 0):
        """
        StudentT distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: StudentT
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def studenttprocess(df, kernel, index_points=None, mean_fn=None, observation_noise_variance=0.0, marginal_fn=None, cholesky_fn=None, jitter=1e-06, always_yield_multivariate_student_t=None, validate_args=False, allow_nan_stats=False, name='StudentTProcess', shape=(), sample = False, seed = 0):
        """
        StudentTProcess distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            kernel: <class 'inspect._empty'>
            index_points: None
            mean_fn: None
            observation_noise_variance: 0.0
            marginal_fn: None
            cholesky_fn: None
            jitter: 1e-06
            always_yield_multivariate_student_t: None
            validate_args: False
            allow_nan_stats: False
            name: StudentTProcess
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.StudentTProcess(df=df, kernel=kernel, index_points=index_points, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, marginal_fn=marginal_fn, cholesky_fn=cholesky_fn, jitter=jitter, always_yield_multivariate_student_t=always_yield_multivariate_student_t, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.StudentTProcess(df=df, kernel=kernel, index_points=index_points, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, marginal_fn=marginal_fn, cholesky_fn=cholesky_fn, jitter=jitter, always_yield_multivariate_student_t=always_yield_multivariate_student_t, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def studenttprocessregressionmodel(df, kernel, index_points=None, observation_index_points=None, observations=None, observation_noise_variance=0.0, predictive_noise_variance=None, mean_fn=None, cholesky_fn=None, marginal_fn=None, always_yield_multivariate_student_t=None, validate_args=False, allow_nan_stats=False, name='StudentTProcessRegressionModel', _conditional_kernel=None, _conditional_mean_fn=None, shape=(), sample = False, seed = 0):
        """
        StudentTProcessRegressionModel distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            kernel: <class 'inspect._empty'>
            index_points: None
            observation_index_points: None
            observations: None
            observation_noise_variance: 0.0
            predictive_noise_variance: None
            mean_fn: None
            cholesky_fn: None
            marginal_fn: None
            always_yield_multivariate_student_t: None
            validate_args: False
            allow_nan_stats: False
            name: StudentTProcessRegressionModel
            _conditional_kernel: None
            _conditional_mean_fn: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.StudentTProcessRegressionModel(df=df, kernel=kernel, index_points=index_points, observation_index_points=observation_index_points, observations=observations, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, mean_fn=mean_fn, cholesky_fn=cholesky_fn, marginal_fn=marginal_fn, always_yield_multivariate_student_t=always_yield_multivariate_student_t, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name, _conditional_kernel=_conditional_kernel, _conditional_mean_fn=_conditional_mean_fn).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.StudentTProcessRegressionModel(df=df, kernel=kernel, index_points=index_points, observation_index_points=observation_index_points, observations=observations, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, mean_fn=mean_fn, cholesky_fn=cholesky_fn, marginal_fn=marginal_fn, always_yield_multivariate_student_t=always_yield_multivariate_student_t, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name, _conditional_kernel=_conditional_kernel, _conditional_mean_fn=_conditional_mean_fn), shape))

    @staticmethod
    def transformeddistribution(shape=(), sample = False, seed = 0, *args, **kwargs):
        """
        TransformedDistribution distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.TransformedDistribution(args=args, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.TransformedDistribution(args=args, kwargs=kwargs), shape))

    @staticmethod
    def triangular(low=0.0, high=1.0, peak=0.5, validate_args=False, allow_nan_stats=True, name='Triangular', shape=(), sample = False, seed = 0):
        """
        Triangular distribution.
    
        Arguments:
            low: 0.0
            high: 1.0
            peak: 0.5
            validate_args: False
            allow_nan_stats: True
            name: Triangular
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Triangular(low=low, high=high, peak=peak, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Triangular(low=low, high=high, peak=peak, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def truncatedcauchy(loc, scale, low, high, validate_args=False, allow_nan_stats=True, name='TruncatedCauchy', shape=(), sample = False, seed = 0):
        """
        TruncatedCauchy distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            low: <class 'inspect._empty'>
            high: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: TruncatedCauchy
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def truncatednormal(loc, scale, low, high, validate_args=False, allow_nan_stats=True, name='TruncatedNormal', shape=(), sample = False, seed = 0):
        """
        TruncatedNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            low: <class 'inspect._empty'>
            high: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: TruncatedNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def twopiecenormal(loc, scale, skewness, validate_args=False, allow_nan_stats=True, name='TwoPieceNormal', shape=(), sample = False, seed = 0):
        """
        TwoPieceNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            skewness: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: TwoPieceNormal
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.TwoPieceNormal(loc=loc, scale=scale, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.TwoPieceNormal(loc=loc, scale=scale, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def twopiecestudentt(df, loc, scale, skewness, validate_args=False, allow_nan_stats=True, name='TwoPieceStudentT', shape=(), sample = False, seed = 0):
        """
        TwoPieceStudentT distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            skewness: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: TwoPieceStudentT
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.TwoPieceStudentT(df=df, loc=loc, scale=scale, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.TwoPieceStudentT(df=df, loc=loc, scale=scale, skewness=skewness, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def uniform(low=0.0, high=1.0, validate_args=False, allow_nan_stats=True, name='Uniform', shape=(), sample = False, seed = 0):
        """
        Uniform distribution.
    
        Arguments:
            low: 0.0
            high: 1.0
            validate_args: False
            allow_nan_stats: True
            name: Uniform
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Uniform(low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Uniform(low=low, high=high, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def variationalgaussianprocess(kernel, index_points, inducing_index_points, variational_inducing_observations_loc, variational_inducing_observations_scale, mean_fn=None, observation_noise_variance=None, predictive_noise_variance=None, cholesky_fn=None, use_whitening_transform=False, jitter=1e-06, validate_args=False, allow_nan_stats=False, name='VariationalGaussianProcess', shape=(), sample = False, seed = 0):
        """
        VariationalGaussianProcess distribution.
    
        Arguments:
            kernel: <class 'inspect._empty'>
            index_points: <class 'inspect._empty'>
            inducing_index_points: <class 'inspect._empty'>
            variational_inducing_observations_loc: <class 'inspect._empty'>
            variational_inducing_observations_scale: <class 'inspect._empty'>
            mean_fn: None
            observation_noise_variance: None
            predictive_noise_variance: None
            cholesky_fn: None
            use_whitening_transform: False
            jitter: 1e-06
            validate_args: False
            allow_nan_stats: False
            name: VariationalGaussianProcess
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.VariationalGaussianProcess(kernel=kernel, index_points=index_points, inducing_index_points=inducing_index_points, variational_inducing_observations_loc=variational_inducing_observations_loc, variational_inducing_observations_scale=variational_inducing_observations_scale, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, cholesky_fn=cholesky_fn, use_whitening_transform=use_whitening_transform, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.VariationalGaussianProcess(kernel=kernel, index_points=index_points, inducing_index_points=inducing_index_points, variational_inducing_observations_loc=variational_inducing_observations_loc, variational_inducing_observations_scale=variational_inducing_observations_scale, mean_fn=mean_fn, observation_noise_variance=observation_noise_variance, predictive_noise_variance=predictive_noise_variance, cholesky_fn=cholesky_fn, use_whitening_transform=use_whitening_transform, jitter=jitter, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def vectordeterministic(loc, atol=None, rtol=None, validate_args=False, allow_nan_stats=True, name='VectorDeterministic', shape=(), sample = False, seed = 0):
        """
        VectorDeterministic distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            atol: None
            rtol: None
            validate_args: False
            allow_nan_stats: True
            name: VectorDeterministic
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.VectorDeterministic(loc=loc, atol=atol, rtol=rtol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.VectorDeterministic(loc=loc, atol=atol, rtol=rtol, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def vonmises(loc, concentration, validate_args=False, allow_nan_stats=True, name='VonMises', shape=(), sample = False, seed = 0):
        """
        VonMises distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: VonMises
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.VonMises(loc=loc, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.VonMises(loc=loc, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def vonmisesfisher(mean_direction, concentration, validate_args=False, allow_nan_stats=True, name='VonMisesFisher', shape=(), sample = False, seed = 0):
        """
        VonMisesFisher distribution.
    
        Arguments:
            mean_direction: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: VonMisesFisher
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def weibull(concentration, scale, validate_args=False, allow_nan_stats=True, name='Weibull', shape=(), sample = False, seed = 0):
        """
        Weibull distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: False
            allow_nan_stats: True
            name: Weibull
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Weibull(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Weibull(concentration=concentration, scale=scale, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def wishartlinearoperator(df, scale, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='WishartLinearOperator', shape=(), sample = False, seed = 0):
        """
        WishartLinearOperator distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            input_output_cholesky: False
            validate_args: False
            allow_nan_stats: True
            name: WishartLinearOperator
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.WishartLinearOperator(df=df, scale=scale, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.WishartLinearOperator(df=df, scale=scale, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def wisharttril(df, scale_tril=None, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='WishartTriL', shape=(), sample = False, seed = 0):
        """
        WishartTriL distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            scale_tril: None
            input_output_cholesky: False
            validate_args: False
            allow_nan_stats: True
            name: WishartTriL
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.WishartTriL(df=df, scale_tril=scale_tril, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.WishartTriL(df=df, scale_tril=scale_tril, input_output_cholesky=input_output_cholesky, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def zeroinflatednegativebinomial(inflated_loc_logits=None, inflated_loc_probs=None, name='ZeroInflatedNegativeBinomial', shape=(), sample = False, seed = 0, **kwargs):
        """
        ZeroInflatedNegativeBinomial distribution.
    
        Arguments:
            inflated_loc_logits: None
            inflated_loc_probs: None
            name: ZeroInflatedNegativeBinomial
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.ZeroInflatedNegativeBinomial(inflated_loc_logits=inflated_loc_logits, inflated_loc_probs=inflated_loc_probs, name=name, kwargs=kwargs).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.ZeroInflatedNegativeBinomial(inflated_loc_logits=inflated_loc_logits, inflated_loc_probs=inflated_loc_probs, name=name, kwargs=kwargs), shape))

    @staticmethod
    def zipf(power, dtype=jax.numpy.int32, force_probs_to_zero_outside_support=False, sample_maximum_iterations=100, validate_args=False, allow_nan_stats=False, name='Zipf', shape=(), sample = False, seed = 0):
        """
        Zipf distribution.
    
        Arguments:
            power: <class 'inspect._empty'>
            dtype: <class 'jax.numpy.int32'>
            force_probs_to_zero_outside_support: False
            sample_maximum_iterations: 100
            validate_args: False
            allow_nan_stats: False
            name: Zipf
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.Zipf(power=power, dtype=dtype, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, sample_maximum_iterations=sample_maximum_iterations, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.Zipf(power=power, dtype=dtype, force_probs_to_zero_outside_support=force_probs_to_zero_outside_support, sample_maximum_iterations=sample_maximum_iterations, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def independent_joint_distribution_from_structure(structure_of_distributions, batch_ndims=None, validate_args=False, shape=(), sample = False, seed = 0):
        """
        independent_joint_distribution_from_structure distribution.
    
        Arguments:
            structure_of_distributions: <class 'inspect._empty'>
            batch_ndims: None
            validate_args: False
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.independent_joint_distribution_from_structure(structure_of_distributions=structure_of_distributions, batch_ndims=batch_ndims, validate_args=validate_args).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.independent_joint_distribution_from_structure(structure_of_distributions=structure_of_distributions, batch_ndims=batch_ndims, validate_args=validate_args), shape))

    @staticmethod
    def kl_divergence(distribution_a, distribution_b, allow_nan_stats=True, name=None, shape=(), sample = False, seed = 0):
        """
        kl_divergence distribution.
    
        Arguments:
            distribution_a: <class 'inspect._empty'>
            distribution_b: <class 'inspect._empty'>
            allow_nan_stats: True
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.kl_divergence(distribution_a=distribution_a, distribution_b=distribution_b, allow_nan_stats=allow_nan_stats, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.kl_divergence(distribution_a=distribution_a, distribution_b=distribution_b, allow_nan_stats=allow_nan_stats, name=name), shape))

    @staticmethod
    def mvn_conjugate_linear_update(prior_scale, linear_transformation, likelihood_scale, observation, prior_mean=None, name=None, shape=(), sample = False, seed = 0):
        """
        mvn_conjugate_linear_update distribution.
    
        Arguments:
            prior_scale: <class 'inspect._empty'>
            linear_transformation: <class 'inspect._empty'>
            likelihood_scale: <class 'inspect._empty'>
            observation: <class 'inspect._empty'>
            prior_mean: None
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.mvn_conjugate_linear_update(prior_scale=prior_scale, linear_transformation=linear_transformation, likelihood_scale=likelihood_scale, observation=observation, prior_mean=prior_mean, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.mvn_conjugate_linear_update(prior_scale=prior_scale, linear_transformation=linear_transformation, likelihood_scale=likelihood_scale, observation=observation, prior_mean=prior_mean, name=name), shape))

    @staticmethod
    def normal_conjugates_known_scale_posterior(prior, scale, s, n, shape=(), sample = False, seed = 0):
        """
        normal_conjugates_known_scale_posterior distribution.
    
        Arguments:
            prior: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            s: <class 'inspect._empty'>
            n: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.normal_conjugates_known_scale_posterior(prior=prior, scale=scale, s=s, n=n).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.normal_conjugates_known_scale_posterior(prior=prior, scale=scale, s=s, n=n), shape))

    @staticmethod
    def normal_conjugates_known_scale_predictive(prior, scale, s, n, shape=(), sample = False, seed = 0):
        """
        normal_conjugates_known_scale_predictive distribution.
    
        Arguments:
            prior: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            s: <class 'inspect._empty'>
            n: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.normal_conjugates_known_scale_predictive(prior=prior, scale=scale, s=s, n=n).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.normal_conjugates_known_scale_predictive(prior=prior, scale=scale, s=s, n=n), shape))

    @staticmethod
    def quadrature_scheme_lognormal_gauss_hermite(loc, scale, quadrature_size, validate_args=False, name=None, shape=(), sample = False, seed = 0):
        """
        quadrature_scheme_lognormal_gauss_hermite distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            quadrature_size: <class 'inspect._empty'>
            validate_args: False
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.quadrature_scheme_lognormal_gauss_hermite(loc=loc, scale=scale, quadrature_size=quadrature_size, validate_args=validate_args, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.quadrature_scheme_lognormal_gauss_hermite(loc=loc, scale=scale, quadrature_size=quadrature_size, validate_args=validate_args, name=name), shape))

    @staticmethod
    def quadrature_scheme_lognormal_quantiles(loc, scale, quadrature_size, validate_args=False, name=None, shape=(), sample = False, seed = 0):
        """
        quadrature_scheme_lognormal_quantiles distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            quadrature_size: <class 'inspect._empty'>
            validate_args: False
            name: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = jax.random.PRNGKey(seed)
            return tfd.quadrature_scheme_lognormal_quantiles(loc=loc, scale=scale, quadrature_size=quadrature_size, validate_args=validate_args, name=name).sample(sample_shape = shape, seed = seed)
        else: 
            return root(tfd.Sample(tfd.quadrature_scheme_lognormal_quantiles(loc=loc, scale=scale, quadrature_size=quadrature_size, validate_args=validate_args, name=name), shape))

