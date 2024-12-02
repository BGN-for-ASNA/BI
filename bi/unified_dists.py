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

    def autocompositetensordistribution(dtype, reparameterization_type, validate_args, allow_nan_stats, shape=(), sample=False, seed=0, parameters=None, graph_parents=None, name=None, *args, **kwargs):
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

    def autoregressive(distribution_fn, shape=(), sample=False, seed=0, sample0=None, num_steps=None, validate_args=False, allow_nan_stats=True, name='Autoregressive', *args, **kwargs):
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

    def batchbroadcast(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def batchconcat(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def batchreshape(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def bates(total_count, shape=(), sample=False, seed=0, low=0.0, high=1.0, validate_args=False, allow_nan_stats=True, name='Bates', *args, **kwargs):
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

    def bernoulli(shape=(), sample=False, seed=0, logits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='Bernoulli', *args, **kwargs):
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

    def beta(concentration1, concentration0, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Beta', *args, **kwargs):
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

    def betabinomial(total_count, concentration1, concentration0, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='BetaBinomial', *args, **kwargs):
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

    def betaquotient(concentration1_numerator, concentration0_numerator, concentration1_denominator, concentration0_denominator, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='BetaQuotient', *args, **kwargs):
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

    def binomial(total_count, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name=None, *args, **kwargs):
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

    def blockwise(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def categorical(shape=(), sample=False, seed=0, logits=None, probs=None, dtype=jax.numpy.int32, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Categorical', *args, **kwargs):
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

    def cauchy(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Cauchy', *args, **kwargs):
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

    def chi(df, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Chi', *args, **kwargs):
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

    def chi2(df, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Chi2', *args, **kwargs):
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

    def choleskylkj(dimension, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='CholeskyLKJ', *args, **kwargs):
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

    def continuousbernoulli(shape=(), sample=False, seed=0, logits=None, probs=None, dtype=jax.numpy.float32, validate_args=False, allow_nan_stats=True, name='ContinuousBernoulli', *args, **kwargs):
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

    def determinantalpointprocess(eigenvalues, eigenvectors, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=False, name='DeterminantalPointProcess', *args, **kwargs):
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

    def deterministic(loc, shape=(), sample=False, seed=0, atol=None, rtol=None, validate_args=False, allow_nan_stats=True, name='Deterministic', *args, **kwargs):
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

    def dirichlet(concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Dirichlet', *args, **kwargs):
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

    def dirichletmultinomial(total_count, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='DirichletMultinomial', *args, **kwargs):
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

    def distribution(dtype, reparameterization_type, validate_args, allow_nan_stats, shape=(), sample=False, seed=0, parameters=None, graph_parents=None, name=None, *args, **kwargs):
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

    def doublesidedmaxwell(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='doublesided_maxwell', *args, **kwargs):
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

    def empirical(samples, shape=(), sample=False, seed=0, event_ndims=0, validate_args=False, allow_nan_stats=True, name='Empirical', *args, **kwargs):
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

    def expgamma(concentration, shape=(), sample=False, seed=0, rate=None, log_rate=None, validate_args=False, allow_nan_stats=True, name='ExpGamma', *args, **kwargs):
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

    def expinversegamma(concentration, shape=(), sample=False, seed=0, scale=None, log_scale=None, validate_args=False, allow_nan_stats=True, name='ExpInverseGamma', *args, **kwargs):
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

    def exprelaxedonehotcategorical(temperature, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='ExpRelaxedOneHotCategorical', *args, **kwargs):
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

    def exponential(rate, shape=(), sample=False, seed=0, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Exponential', *args, **kwargs):
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

    def exponentiallymodifiedgaussian(loc, scale, rate, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='ExponentiallyModifiedGaussian', *args, **kwargs):
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

    def finitediscrete(outcomes, shape=(), sample=False, seed=0, logits=None, probs=None, rtol=None, atol=None, validate_args=False, allow_nan_stats=True, name='FiniteDiscrete', *args, **kwargs):
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

    def flatdirichlet(concentration_shape, shape=(), sample=False, seed=0, dtype=jax.numpy.float32, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='FlatDirichlet', *args, **kwargs):
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

    def gamma(concentration, shape=(), sample=False, seed=0, rate=None, log_rate=None, validate_args=False, allow_nan_stats=True, force_probs_to_zero_outside_support=False, name='Gamma', *args, **kwargs):
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

    def gammagamma(concentration, mixing_concentration, mixing_rate, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='GammaGamma', *args, **kwargs):
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

    def gaussianprocess(kernel, shape=(), sample=False, seed=0, index_points=None, mean_fn=None, observation_noise_variance=0.0, marginal_fn=None, cholesky_fn=None, jitter=1e-06, validate_args=False, allow_nan_stats=False, parameters=None, name='GaussianProcess', _check_marginal_cholesky_fn=True, *args, **kwargs):
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

    def gaussianprocessregressionmodel(kernel, shape=(), sample=False, seed=0, index_points=None, observation_index_points=None, observations=None, observation_noise_variance=0.0, predictive_noise_variance=None, mean_fn=None, cholesky_fn=None, jitter=1e-06, validate_args=False, allow_nan_stats=False, name='GaussianProcessRegressionModel', _conditional_kernel=None, _conditional_mean_fn=None, *args, **kwargs):
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

    def generalizedextremevalue(loc, scale, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='GeneralizedExtremeValue', *args, **kwargs):
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

    def generalizednormal(loc, scale, power, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='GeneralizedNormal', *args, **kwargs):
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

    def generalizedpareto(loc, scale, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name=None, *args, **kwargs):
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

    def geometric(shape=(), sample=False, seed=0, logits=None, probs=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Geometric', *args, **kwargs):
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

    def gumbel(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Gumbel', *args, **kwargs):
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

    def halfcauchy(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='HalfCauchy', *args, **kwargs):
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

    def halfnormal(scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='HalfNormal', *args, **kwargs):
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

    def halfstudentt(df, loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='HalfStudentT', *args, **kwargs):
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

    def hiddenmarkovmodel(initial_distribution, transition_distribution, observation_distribution, num_steps, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, time_varying_transition_distribution=False, time_varying_observation_distribution=False, mask=None, name='HiddenMarkovModel', *args, **kwargs):
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

    def horseshoe(scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Horseshoe', *args, **kwargs):
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

    def independent(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def inflated(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def inversegamma(concentration, shape=(), sample=False, seed=0, scale=None, validate_args=False, allow_nan_stats=True, name='InverseGamma', *args, **kwargs):
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

    def inversegaussian(loc, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='InverseGaussian', *args, **kwargs):
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

    def johnsonsu(skewness, tailweight, loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name=None, *args, **kwargs):
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

    def jointdistribution(dtype, validate_args, parameters, name, shape=(), sample=False, seed=0, use_vectorized_map=False, batch_ndims=None, experimental_use_kahan_sum=False, *args, **kwargs):
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

    def jointdistributioncoroutine(model, shape=(), sample=False, seed=0, sample_dtype=None, batch_ndims=None, use_vectorized_map=False, validate_args=False, experimental_use_kahan_sum=False, name=None, *args, **kwargs):
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

    def jointdistributioncoroutineautobatched(model, shape=(), sample=False, seed=0, sample_dtype=None, batch_ndims=0, use_vectorized_map=True, validate_args=False, experimental_use_kahan_sum=False, name=None, *args, **kwargs):
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

    def jointdistributionnamed(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def jointdistributionnamedautobatched(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def jointdistributionsequential(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def jointdistributionsequentialautobatched(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def kumaraswamy(shape=(), sample=False, seed=0, concentration1=1.0, concentration0=1.0, validate_args=False, allow_nan_stats=True, name='Kumaraswamy', *args, **kwargs):
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

    def lkj(dimension, concentration, shape=(), sample=False, seed=0, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='LKJ', *args, **kwargs):
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

    def lambertwdistribution(distribution, shift, scale, shape=(), sample=False, seed=0, tailweight=None, validate_args=False, allow_nan_stats=True, name='LambertWDistribution', *args, **kwargs):
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

    def lambertwnormal(loc, scale, shape=(), sample=False, seed=0, tailweight=None, validate_args=False, allow_nan_stats=True, name='LambertWNormal', *args, **kwargs):
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

    def laplace(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Laplace', *args, **kwargs):
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

    def lineargaussianstatespacemodel(num_timesteps, transition_matrix, transition_noise, observation_matrix, observation_noise, initial_state_prior, shape=(), sample=False, seed=0, initial_step=0, mask=None, experimental_parallelize=False, validate_args=False, allow_nan_stats=True, name='LinearGaussianStateSpaceModel', *args, **kwargs):
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

    def loglogistic(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='LogLogistic', *args, **kwargs):
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

    def lognormal(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='LogNormal', *args, **kwargs):
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

    def logistic(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Logistic', *args, **kwargs):
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

    def logitnormal(loc, scale, shape=(), sample=False, seed=0, num_probit_terms_approx=2, gauss_hermite_scale_limit=None, gauss_hermite_degree=20, validate_args=False, allow_nan_stats=True, name='LogitNormal', *args, **kwargs):
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

    def markovchain(initial_state_prior, transition_fn, num_steps, shape=(), sample=False, seed=0, experimental_use_kahan_sum=False, validate_args=False, name='MarkovChain', *args, **kwargs):
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

    def masked(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def matrixnormallinearoperator(loc, scale_row, scale_column, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='MatrixNormalLinearOperator', *args, **kwargs):
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

    def matrixtlinearoperator(df, loc, scale_row, scale_column, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='MatrixTLinearOperator', *args, **kwargs):
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

    def mixture(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def mixturesamefamily(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def moyal(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Moyal', *args, **kwargs):
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

    def multinomial(total_count, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='Multinomial', *args, **kwargs):
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

    def multivariatenormaldiag(shape=(), sample=False, seed=0, loc=None, scale_diag=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalDiag', *args, **kwargs):
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

    def multivariatenormaldiagpluslowrank(shape=(), sample=False, seed=0, loc=None, scale_diag=None, scale_perturb_factor=None, scale_perturb_diag=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiagPlusLowRank', *args, **kwargs):
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

    def multivariatenormaldiagpluslowrankcovariance(shape=(), sample=False, seed=0, loc=None, cov_diag_factor=None, cov_perturb_factor=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiagPlusLowRankCovariance', *args, **kwargs):
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

    def multivariatenormalfullcovariance(shape=(), sample=False, seed=0, loc=None, covariance_matrix=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalFullCovariance', *args, **kwargs):
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

    def multivariatenormallinearoperator(shape=(), sample=False, seed=0, loc=None, scale=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalLinearOperator', *args, **kwargs):
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

    def multivariatenormaltril(shape=(), sample=False, seed=0, loc=None, scale_tril=None, validate_args=False, allow_nan_stats=True, experimental_use_kahan_sum=False, name='MultivariateNormalTriL', *args, **kwargs):
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

    def multivariatestudenttlinearoperator(df, loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='MultivariateStudentTLinearOperator', *args, **kwargs):
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

    def negativebinomial(total_count, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, require_integer_total_count=True, name='NegativeBinomial', *args, **kwargs):
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

    def noncentralchi2(df, noncentrality, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='NoncentralChi2', *args, **kwargs):
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

    def normal(loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Normal', *args, **kwargs):
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

    def normalinversegaussian(loc, scale, tailweight, skewness, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='NormalInverseGaussian', *args, **kwargs):
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

    def onehotcategorical(shape=(), sample=False, seed=0, logits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='OneHotCategorical', *args, **kwargs):
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

    def orderedlogistic(cutpoints, loc, shape=(), sample=False, seed=0, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='OrderedLogistic', *args, **kwargs):
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

    def pert(low, peak, high, shape=(), sample=False, seed=0, temperature=4.0, validate_args=False, allow_nan_stats=False, name='PERT', *args, **kwargs):
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

    def pareto(concentration, shape=(), sample=False, seed=0, scale=1.0, validate_args=False, allow_nan_stats=True, name='Pareto', *args, **kwargs):
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

    def plackettluce(scores, shape=(), sample=False, seed=0, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='PlackettLuce', *args, **kwargs):
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

    def poisson(shape=(), sample=False, seed=0, rate=None, log_rate=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Poisson', *args, **kwargs):
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

    def poissonlognormalquadraturecompound(loc, scale, shape=(), sample=False, seed=0, quadrature_size=8, quadrature_fn=tfp.distributions.quadrature_scheme_lognormal_quantiles, validate_args=False, allow_nan_stats=True, name='PoissonLogNormalQuadratureCompound', *args, **kwargs):
        """
        PoissonLogNormalQuadratureCompound distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            quadrature_size: 8
            quadrature_fn: <function quadrature_scheme_lognormal_quantiles at 0x0000014B29A33F60>
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

    def powerspherical(mean_direction, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='PowerSpherical', *args, **kwargs):
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

    def probitbernoulli(shape=(), sample=False, seed=0, probits=None, probs=None, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='ProbitBernoulli', *args, **kwargs):
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

    def quantizeddistribution(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def registerkl(dist_cls_a, dist_cls_b, shape=(), sample=False, seed=0, *args, **kwargs):
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

    def relaxedbernoulli(temperature, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='RelaxedBernoulli', *args, **kwargs):
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

    def relaxedonehotcategorical(temperature, shape=(), sample=False, seed=0, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='RelaxedOneHotCategorical', *args, **kwargs):
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

    def reparameterizationtype(rep_type, shape=(), sample=False, seed=0, *args, **kwargs):
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

    def sample(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def sigmoidbeta(concentration1, concentration0, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='SigmoidBeta', *args, **kwargs):
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

    def sinharcsinh(loc, scale, shape=(), sample=False, seed=0, skewness=None, tailweight=None, distribution=None, validate_args=False, allow_nan_stats=True, name='SinhArcsinh', *args, **kwargs):
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

    def skellam(shape=(), sample=False, seed=0, rate1=None, rate2=None, log_rate1=None, log_rate2=None, force_probs_to_zero_outside_support=False, validate_args=False, allow_nan_stats=True, name='Skellam', *args, **kwargs):
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

    def sphericaluniform(dimension, shape=(), sample=False, seed=0, batch_shape=(), dtype=jax.numpy.float32, validate_args=False, allow_nan_stats=True, name='SphericalUniform', *args, **kwargs):
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

    def stoppingratiologistic(cutpoints, loc, shape=(), sample=False, seed=0, dtype=jax.numpy.int32, validate_args=False, allow_nan_stats=True, name='StoppingRatioLogistic', *args, **kwargs):
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

    def studentt(df, loc, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='StudentT', *args, **kwargs):
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

    def studenttprocess(df, kernel, shape=(), sample=False, seed=0, index_points=None, mean_fn=None, observation_noise_variance=0.0, marginal_fn=None, cholesky_fn=None, jitter=1e-06, always_yield_multivariate_student_t=None, validate_args=False, allow_nan_stats=False, name='StudentTProcess', *args, **kwargs):
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

    def studenttprocessregressionmodel(df, kernel, shape=(), sample=False, seed=0, index_points=None, observation_index_points=None, observations=None, observation_noise_variance=0.0, predictive_noise_variance=None, mean_fn=None, cholesky_fn=None, marginal_fn=None, always_yield_multivariate_student_t=None, validate_args=False, allow_nan_stats=False, name='StudentTProcessRegressionModel', _conditional_kernel=None, _conditional_mean_fn=None, *args, **kwargs):
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

    def transformeddistribution(shape=(), sample=False, seed=0, *args, **kwargs):
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

    def triangular(shape=(), sample=False, seed=0, low=0.0, high=1.0, peak=0.5, validate_args=False, allow_nan_stats=True, name='Triangular', *args, **kwargs):
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

    def truncatedcauchy(loc, scale, low, high, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='TruncatedCauchy', *args, **kwargs):
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

    def truncatednormal(loc, scale, low, high, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='TruncatedNormal', *args, **kwargs):
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

    def twopiecenormal(loc, scale, skewness, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='TwoPieceNormal', *args, **kwargs):
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

    def twopiecestudentt(df, loc, scale, skewness, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='TwoPieceStudentT', *args, **kwargs):
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

    def uniform(shape=(), sample=False, seed=0, low=0.0, high=1.0, validate_args=False, allow_nan_stats=True, name='Uniform', *args, **kwargs):
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

    def variationalgaussianprocess(kernel, index_points, inducing_index_points, variational_inducing_observations_loc, variational_inducing_observations_scale, shape=(), sample=False, seed=0, mean_fn=None, observation_noise_variance=None, predictive_noise_variance=None, cholesky_fn=None, use_whitening_transform=False, jitter=1e-06, validate_args=False, allow_nan_stats=False, name='VariationalGaussianProcess', *args, **kwargs):
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

    def vectordeterministic(loc, shape=(), sample=False, seed=0, atol=None, rtol=None, validate_args=False, allow_nan_stats=True, name='VectorDeterministic', *args, **kwargs):
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

    def vonmises(loc, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='VonMises', *args, **kwargs):
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

    def vonmisesfisher(mean_direction, concentration, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='VonMisesFisher', *args, **kwargs):
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

    def weibull(concentration, scale, shape=(), sample=False, seed=0, validate_args=False, allow_nan_stats=True, name='Weibull', *args, **kwargs):
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

    def wishartlinearoperator(df, scale, shape=(), sample=False, seed=0, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='WishartLinearOperator', *args, **kwargs):
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

    def wisharttril(df, shape=(), sample=False, seed=0, scale_tril=None, input_output_cholesky=False, validate_args=False, allow_nan_stats=True, name='WishartTriL', *args, **kwargs):
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

    def zeroinflatednegativebinomial(shape=(), sample=False, seed=0, inflated_loc_logits=None, inflated_loc_probs=None, name='ZeroInflatedNegativeBinomial', *args, **kwargs):
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

    def zipf(power, shape=(), sample=False, seed=0, dtype=jax.numpy.int32, force_probs_to_zero_outside_support=False, sample_maximum_iterations=100, validate_args=False, allow_nan_stats=False, name='Zipf', *args, **kwargs):
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

    def independent_joint_distribution_from_structure(structure_of_distributions, shape=(), sample=False, seed=0, batch_ndims=None, validate_args=False, *args, **kwargs):
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

    def kl_divergence(distribution_a, distribution_b, shape=(), sample=False, seed=0, allow_nan_stats=True, name=None, *args, **kwargs):
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

    def mvn_conjugate_linear_update(prior_scale, linear_transformation, likelihood_scale, observation, shape=(), sample=False, seed=0, prior_mean=None, name=None, *args, **kwargs):
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

    def normal_conjugates_known_scale_posterior(prior, scale, s, n, shape=(), sample=False, seed=0, *args, **kwargs):
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

    def normal_conjugates_known_scale_predictive(prior, scale, s, n, shape=(), sample=False, seed=0, *args, **kwargs):
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

    def quadrature_scheme_lognormal_gauss_hermite(loc, scale, quadrature_size, shape=(), sample=False, seed=0, validate_args=False, name=None, *args, **kwargs):
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

    def quadrature_scheme_lognormal_quantiles(loc, scale, quadrature_size, shape=(), sample=False, seed=0, validate_args=False, name=None, *args, **kwargs):
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

