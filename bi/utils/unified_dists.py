from functools import partial
from jax import random
from jax import jit
import numpyro as numpyro

class UnifiedDist:

    def __init__(self):
        pass

    @staticmethod
    def asymmetriclaplace(loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from or defines the AsymmetricLaplace distribution.

        This function can either return a sample from the AsymmetricLaplace distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            loc (float, optional): Location parameter. Defaults to 0.0.
            scale (float, optional): Scale parameter. Defaults to 1.0.
            asymmetry (float, optional): Asymmetry parameter. Defaults to 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args).expand(shape))

    @staticmethod
    def asymmetriclaplacequantile(loc=0.0, scale=1.0, quantile=0.5, validate_args=None, shape=(), sample=False, seed=0, name='x'):
        """
        Generates samples from or defines the AsymmetricLaplaceQuantile distribution.

        This function can either return a sample from the AsymmetricLaplaceQuantile distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            loc (float, optional): Location parameter. Defaults to 0.0.
            scale (float, optional): Scale parameter. Defaults to 1.0.
            quantile (float, optional): Quantile parameter. Defaults to 0.5.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
                seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args).sample(seed, shape)
        else:
            return numpyro.sample(name, numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoulli(probs=None, logits=None, validate_args=None, shape=(), sample=False, seed=0, name='x'):
        """
        Generates samples from or defines the Bernoulli distribution.

        This function can either return a sample from the Bernoulli distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            probs (float, optional): Probability of success. Defaults to None.
            logits (float, optional): Logit probability of success. Defaults to None.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else:
            return numpyro.sample(name, numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoullilogits(logits=None, validate_args=None, shape=(), sample=False, seed=0, name='x'):
        """
        Generates samples from or defines the BernoulliLogits distribution.

        This function can either return a sample from the BernoulliLogits distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            logits (float, optional): Logit probability of success. Defaults to None.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else:
            return numpyro.sample(name, numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoulliprobs(probs, validate_args=None, shape=(), sample=False, seed=0, name='x'):
        """
        Generates samples from or defines the BernoulliProbs distribution.

        This function can either return a sample from the BernoulliProbs distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            probs (float): Probability of success.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else:
            return numpyro.sample(name, numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def beta(concentration1, concentration0, validate_args=None, shape=(), sample=False, seed=0, name='x'):
        """
        Generates samples from or defines the Beta distribution.

        This function can either return a sample from the Beta distribution
        or define the distribution itself, depending on the value of the 'sample'
        parameter.

        Args:
            concentration1 (float): First concentration parameter.
            concentration0 (float): Second concentration parameter.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to None.
            shape (tuple, optional): Shape of the samples to be drawn. Defaults to ().
            sample (bool, optional): Whether to return a sample. Defaults to False.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
            name (str, optional): Name of the sample. Defaults to 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).sample(seed, shape)
        else:
            return numpyro.sample(name, numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(shape))
    
    @staticmethod
    def betabinomial(concentration1, concentration0, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a Beta-Binomial distribution.

        Args:
            concentration1: The first concentration parameter (alpha) of the Beta distribution.
            concentration0: The second concentration parameter (beta) of the Beta distribution.
            total_count: The total number of Bernoulli trials.
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def betaproportion(mean, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a BetaProportion distribution.

        Args:
            mean: The mean parameter of the Beta distribution.
            concentration: The concentration parameter of the Beta distribution.
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.          
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomial(total_count=1, probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a Binomial distribution.

        Args:
            total_count: The total number of Bernoulli trials.
            probs: The probability of success in each trial (mutually exclusive with logits).
            logits: The log-odds of success in each trial (mutually exclusive with probs).
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.        
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomiallogits(logits, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a BinomialLogits distribution.

        Args:
            logits: The log-odds of success in each trial.
            total_count: The total number of Bernoulli trials.
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.            
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomialprobs(probs, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a BinomialProbs distribution.

        Args:
            probs: The probability of success in each trial.
            total_count: The total number of Bernoulli trials.
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.            
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def car(loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a Conditional Autoregressive (CAR) distribution.

        Args:
            loc: The location parameter.
            correlation: The correlation parameter.
            conditional_precision: The conditional precision parameter.
            adj_matrix: The adjacency matrix defining the graph structure.
            is_sparse: Whether the adjacency matrix is sparse (default: False).
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.       
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args).expand(shape))

    @staticmethod
    def categorical(probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a Categorical distribution.

        Args:
            probs: The probabilities associated with each category (mutually exclusive with logits).
            logits: The log-probabilities associated with each category (mutually exclusive with probs).
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.
            name: The name of the random variable.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.      
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def categoricallogits(logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates a CategoricalLogits distribution.
    
        Args:
            logits: The log-probabilities associated with each category.
            validate_args: Whether to validate the arguments (default: None).
            shape: The shape of the samples to be drawn.
            sample: Whether to return samples (True) or the distribution object (False).
            seed: The seed for random number generation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def categoricalprobs(probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Categorical distribution with probabilities defined by probabilities.
        
        Args:
            probs: Tensor of probabilities. Must sum to 1 along the last dimension.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def cauchy(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Cauchy distribution.

        Args:
            loc: Location parameter. Default is 0.0.
            scale: Scale parameter. Must be positive. Default is 1.0.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def chi2(df, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Chi-squared distribution.

        Args:
            df: Degrees of freedom. Must be positive.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Chi2(df=df, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Chi2(df=df, validate_args=validate_args).expand(shape))

    @staticmethod
    def delta(v=0.0, log_density=0.0, event_dim=0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Delta distribution.

        Args:
            v: Value at which the delta distribution is centered. Must be a scalar or broadcastable with the    batch shape.
            log_density: Log density at the center. Default is 0.0.
            event_dim: Number of event dimensions. Default is 0.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args).expand(shape))

    @staticmethod
    def dirichlet(concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Dirichlet distribution.

        Args:
            concentration: Concentration parameters. Must be positive. Must have shape (K,) where K is the  number of categories.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def dirichletmultinomial(concentration, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        DirichletMultinomial distribution.

        Args:
            concentration: Concentration parameters. Must be positive. Must have shape (K,) where K is the  number of categories.
            total_count: Total number of trials. Default is 1.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If  `sample` is False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def discreteuniform(low=0, high=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        DiscreteUniform distribution.

        Args:
            low: Lower bound (inclusive). Default is 0.
            high: Upper bound (inclusive). Default is 1.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If `sample` is  False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def eulermaruyama(t, sde_fn, init_dist, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        EulerMaruyama distribution.

        Args:
            t: Time points to evaluate the SDE.
            sde_fn: Stochastic differential equation function.
            init_dist: Initial distribution.
            validate_args: If True, validate the arguments; else, skip validation.
            shape: Shape of samples to be drawn. If `sample` is True, this is passed to `sample()`. If `sample` is  False, this is passed to `expand()`.
            sample: If True, samples from the distribution. If False, returns the distribution object.
            seed: Seed for random number generation.
            name: Name for the distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args).expand(shape))

    @staticmethod
    def expandeddistribution(base_dist, batch_shape=(), shape=(), sample = False, seed = 0, name = 'x'):
        """
        ExpandedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            batch_shape: ()
            shape: Shape of samples to be drawn.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape).expand(shape))

    @staticmethod
    def exponential(rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Exponential distribution.

        Samples or constructs an Exponential distribution with specified rate.

        Args:
            rate (float): The rate parameter of the distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Exponential(rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Exponential(rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def foldeddistribution(base_dist, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        FoldedDistribution.

        Constructs a distribution that folds a base distribution to be non-negative.

        Args:
            base_dist (numpyro.distributions.Distribution): The base distribution to be folded.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args).expand(shape))

    @staticmethod
    def gamma(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gamma distribution.

        Samples or constructs a Gamma distribution with specified concentration and rate parameters.

        Args:
            concentration (float): Shape parameter of the Gamma distribution.
            rate (float, optional): Rate parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gammapoisson(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gamma-Poisson distribution.

        Constructs a Gamma-Poisson distribution, which is a mixture of Gamma and Poisson distributions.

        Args:
            concentration (float): Shape parameter of the Gamma distribution.
            rate (float, optional): Rate parameter of the Gamma distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussiancopula(marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gaussian Copula distribution.

        Constructs a Gaussian Copula distribution using a specified marginal distribution and correlation   structure.

        Args:
            marginal_dist (numpyro.distributions.Distribution): Marginal distribution for the copula.
            correlation_matrix (jax.numpy.ndarray, optional): Correlation matrix. Default is None.
            correlation_cholesky (jax.numpy.ndarray, optional): Cholesky decomposition of the correlation matrix.   Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussiancopulabeta(concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gaussian Copula Beta distribution.

        Constructs a Gaussian Copula Beta distribution using Beta marginals.

        Args:
            concentration1 (float): First concentration parameter of the Beta distribution.
            concentration0 (float): Second concentration parameter of the Beta distribution.
            correlation_matrix (jax.numpy.ndarray, optional): Correlation matrix. Default is None.
            correlation_cholesky (jax.numpy.ndarray, optional): Cholesky decomposition of the correlation matrix.   Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is False.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussianrandomwalk(scale=1.0, num_steps=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gaussian Random Walk distribution.

        Constructs a Gaussian Random Walk distribution, modeling a sequence of steps with Gaussian noise.

        Args:
            scale (float, optional): Scale parameter of the Gaussian distribution. Default is 1.0.
            num_steps (int, optional): Number of steps in the random walk. Default is 1.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometric(probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Geometric distribution.

        Samples or constructs a Geometric distribution modeling the number of trials until the first success.

        Args:
            probs (float, optional): Probability of success in each trial. Default is None.
            logits (float, optional): Log-odds of success in each trial. Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 

        Note:
            Either `probs` or `logits` must be specified, but not both.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometriclogits(logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Geometric Logits distribution.

        Samples or constructs a Geometric distribution using logits instead of probabilities.

        Args:
            logits (float): Log-odds of success in each trial.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """ 
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometricprobs(probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Geometric Probs distribution.

        Samples or constructs a Geometric distribution using probabilities.

        Args:
            probs (float): Probability of success in each trial.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def gompertz(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gompertz distribution.

        Samples or constructs a Gompertz distribution, often used to model mortality rates.

        Args:
            concentration (float): Shape parameter of the distribution.
            rate (float, optional): Rate parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gumbel distribution.

        Samples or constructs a Gumbel distribution, often used to model extreme values.

        Args:
            loc (float, optional): Location parameter. Default is 0.0.
            scale (float, optional): Scale parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def halfcauchy(scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Half-Cauchy distribution.

        Samples or constructs a Half-Cauchy distribution, which is the right half of the Cauchy distribution.

        Args:
            scale (float, optional): Scale parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): Shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample (True) or the distribution (False). Default is  False.
            seed (int, optional): Seed for random number generation. Default is 0.
            name (str, optional): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def halfnormal(scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        HalfNormal distribution.

        Arguments:
        scale (float): Scale parameter of the distribution. Default is 1.0.
        validate_args (bool, optional): Whether to validate the arguments. Default is None.
        shape (tuple): Shape of the samples to be drawn.
        sample (bool): Whether to sample from the distribution. Default is False.
        seed (int): Seed for random number generation. Default is 0.
        name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def improperuniform(support, batch_shape, event_shape, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ImproperUniform distribution.

        Arguments:
            support: Support of the distribution.
            batch_shape: Batch shape of the distribution.
            event_shape: Event shape of the distribution.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def independent(base_dist, reinterpreted_batch_ndims, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Independent distribution.

        Arguments:
            base_dist: Base distribution.
            reinterpreted_batch_ndims (int): Number of batch dimensions to reinterpret.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args).expand(shape))

    @staticmethod
    def inversegamma(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        InverseGamma distribution.

        Arguments:
            concentration (float): Concentration parameter.
            rate (float): Rate parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def kumaraswamy(concentration1, concentration0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Kumaraswamy distribution.

        Arguments:
            concentration1 (float): First concentration parameter.
            concentration (float): Second concentration parameter. Default is an empty tuple, which should be replaced.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(shape))

    @staticmethod
    def lkj(dimension, concentration=1.0, sample_method='onion', validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LKJ distribution.

        Arguments:
            dimension (int): Dimension of the distribution.
            concentration (float): Concentration parameter. Default is 1.0.
            sample_method (str): Sampling method. Default is 'onion'.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(shape))

    @staticmethod
    def lkjcholesky(dimension, concentration=1.0, sample_method='onion', validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LKJCholesky distribution.

        Arguments:
            dimension (int): Dimension of the distribution.
            concentration (float): Concentration parameter. Default is 1.0.
            sample_method (str): Sampling method. Default is 'onion'.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(shape))

    @staticmethod
    def laplace(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Laplace distribution.

        Arguments:
            loc (float): Location parameter. Default is 0.0.
            scale (float): Scale parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def lefttruncateddistribution(base_dist, low=0.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LeftTruncatedDistribution distribution.

        Arguments:
            base_dist: Base distribution to be truncated.
            low (float): Lower truncation point. Default is 0.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args).expand(shape))

    @staticmethod
    def lognormal(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LogNormal distribution.

        Arguments:
            loc (float): Mean of the normal distribution. Default is 0.0.
            scale (float): Standard deviation of the normal distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def loguniform(low, high, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LogUniform distribution.

        Arguments:
            low (float): Lower bound of the distribution.
            high (float): Upper bound of the distribution.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def logistic(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Logistic distribution.

        Arguments:
            loc (float): Location parameter. Default is 0.0.
            scale (float): Scale parameter. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def lowrankmultivariatenormal(loc, cov_factor, cov_diag, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LowRankMultivariateNormal distribution.

        Arguments:
            loc: Location vector.
            cov_factor: Factor for the low-rank covariance matrix.
            cov_diag: Diagonal part of the covariance matrix.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple): Shape of the samples to be drawn.
            sample (bool): Whether to sample from the distribution. Default is False.
            seed (int): Seed for random number generation. Default is 0.
            name (str): Name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args).expand(shape))

    @staticmethod
    def maskeddistribution(base_dist, mask, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MaskedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            mask: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask).expand(shape))

    @staticmethod
    def matrixnormal(loc, scale_tril_row, scale_tril_column, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MatrixNormal distribution.

        Arguments:
            loc: The mean matrix of the distribution.
            scale_tril_row: The lower triangular matrix A such that covariance matrix is A * A.T.
            scale_tril_column: The lower triangular matrix B such that covariance matrix is B.T * B.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixture(mixing_distribution, component_distributions, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Mixture distribution.

        Arguments:
            mixing_distribution: Distribution over the mixture components.
            component_distributions: List of distributions for each mixture component.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixturegeneral(mixing_distribution, component_distributions, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MixtureGeneral distribution.

        Arguments:
            mixing_distribution: Distribution over the mixture components.
            component_distributions: List of distributions for each mixture component.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixturesamefamily(mixing_distribution, component_distribution, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MixtureSameFamily distribution.

        Arguments:
            mixing_distribution: Distribution over the mixture components.
            component_distribution: Single distribution used for all mixture components.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
            name: Name for the sample operation.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomial(total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Multinomial distribution.

        Arguments:
            total_count: Number of trials.
            probs: Probabilities of each outcome.
            logits: Logits corresponding to each outcome.
            total_count_max: Maximum total_count to consider.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomiallogits(logits, total_count=1, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultinomialLogits distribution.

        Arguments:
            logits: Logits corresponding to each outcome.
            total_count: Number of trials.
            total_count_max: Maximum total_count to consider.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomialprobs(probs, total_count=1, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultinomialProbs distribution.

        Arguments:
            probs: Probabilities of each outcome.
            total_count: Number of trials.
            total_count_max: Maximum total_count to consider.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multivariatenormal(loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultivariateNormal distribution.

        Arguments:
            loc: Mean vector.
            covariance_matrix: Covariance matrix.
            precision_matrix: Precision matrix.
            scale_tril: Lower triangular matrix such that covariance matrix is scale_tril * scale_tril.T.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def multivariatestudentt(df, loc=0.0, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultivariateStudentT distribution.

        Arguments:
            df: Degrees of freedom.
            loc: Mean vector.
            scale_tril: Lower triangular matrix for the scale.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.
        
        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomial2(mean, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomial2 distribution.

        Arguments:
            mean: Mean number of successes.
            concentration: Concentration parameter.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomiallogits(total_count, logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomialLogits distribution.

        Arguments:
            total_count: Total number of trials.
            logits: Logits for the probability of success.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomialprobs(total_count, probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomialProbs distribution.

        Arguments:
            total_count: Total number of trials.
            probs: Probability of success.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def normal(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Normal distribution.

        Arguments:
            loc: Mean of the distribution.
            scale: Standard deviation of the distribution.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models. 
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def orderedlogistic(predictor, cutpoints, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        OrderedLogistic distribution.

        Arguments:
            predictor: Predictor variable.
            cutpoints: Cutpoints separating the categories.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.            
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args).expand(shape))

    @staticmethod
    def pareto(scale, alpha, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Pareto distribution.

        Arguments:
            scale: Scale parameter.
            alpha: Shape parameter.
            validate_args: Whether to validate the arguments.
            shape: Shape of samples to be drawn.
            sample: Whether to draw samples (True) or return the distribution (False).
            seed: Seed for random number generation.
            name: Name for the sample operation.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.            
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args).expand(shape))

    @staticmethod
    def poisson(rate, is_sparse=False, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a Poisson distribution.

        Args:
            rate: The rate parameter of the Poisson distribution. Required.
            is_sparse: Whether to return a sparse sample. Default: False.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args).expand(shape))

    @staticmethod
    def projectednormal(concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a ProjectedNormal distribution.

        Args:
            concentration: Concentration parameter of the ProjectedNormal distribution. Required.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def relaxedbernoulli(temperature, probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a RelaxedBernoulli distribution.
    
        Args:
            temperature: Temperature parameter of the RelaxedBernoulli distribution. Required.
            probs: Probabilities of the success state. Either probs or logits must be specified. Default: None.
            logits: Logits of the success state. Either probs or logits must be specified. Default: None.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.
    
        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def relaxedbernoullilogits(temperature, logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a RelaxedBernoulliLogits distribution.

        Args:
            temperature: Temperature parameter of the RelaxedBernoulliLogits distribution. Required.
            logits: Logits of the success state. Required.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def righttruncateddistribution(base_dist, high=0.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a RightTruncatedDistribution.

        Args:
            base_dist: Base distribution to be truncated. Required.
            high: Upper bound for truncation. Default: 0.0.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def sinebivariatevonmises(phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None, weighted_correlation=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a SineBivariateVonMises distribution.

        Args:
            phi_loc: Location parameter for the first angle. Required.
            psi_loc: Location parameter for the second angle. Required.
            phi_concentration: Concentration parameter for the first angle. Required.
            psi_concentration: Concentration parameter for the second angle. Required.
            correlation: Correlation between the two angles. Default: None.
            weighted_correlation: Weighted correlation between the two angles. Default: None.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args).expand(shape))

    @staticmethod
    def sineskewed(base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a SineSkewed distribution.

        Args:
            base_dist: Base distribution to be skewed. Required.
            skewness: Skewness parameter. Required.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args).expand(shape))

    @staticmethod
    def softlaplace(loc, scale, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a SoftLaplace distribution.

        Args:
            loc: Location parameter. Required.
            scale: Scale parameter. Required.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def studentt(df, loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a StudentT distribution.

        Args:
            df: Degrees of freedom. Required.
            loc: Location parameter. Default: 0.0.
            scale: Scale parameter. Default: 1.0.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def transformeddistribution(base_distribution, transforms, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a TransformedDistribution.

        Args:
            base_distribution: Base distribution to be transformed. Required.
            transforms: Transforms to be applied. Required.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatedcauchy(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a TruncatedCauchy distribution.
    
        Args:
            loc: Location parameter. Default: 0.0.
            scale: Scale parameter. Default: 1.0.
            low: Lower bound of truncation. Default: None.
            high: Upper bound of truncation. Default: None.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.
    
        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncateddistribution(base_dist, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Generates samples from a TruncatedDistribution.

        Args:
            base_dist: Base distribution to be truncated. Required.
            low: Lower bound of truncation. Default: None.
            high: Upper bound of truncation. Default: None.
            validate_args: Whether to validate the arguments. Default: None.
            shape: Shape of the samples to be drawn. Default: ().
            sample: Whether to return a sample (True) or a distribution (False). Default: False.
            seed: Seed for random number generation. Default: 0.
            name: Name of the sample. Default: 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatednormal(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Truncated Normal distribution.

        This distribution is similar to the Normal distribution but is bounded between `low` and `high`.

        Args:
            loc (float): The mean of the normal distribution before truncation. Default is 0.0.
            scale (float): The standard deviation of the normal distribution before truncation. Default is 1.0.
            low (float, optional): The lower bound of the truncated distribution. Default is None.
            high (float, optional): The upper bound of the truncated distribution. Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatedpolyagamma(batch_shape=(), validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Truncated PolyaGamma distribution.

        Args:
            batch_shape (tuple, optional): The batch shape of the distribution. Default is ().
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def twosidedtruncateddistribution(base_dist, low=0.0, high=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TwoSidedTruncatedDistribution.

        Args:
            base_dist: The base distribution to be truncated.
            low (float, optional): The lower bound of the truncated distribution. Default is 0.0.
            high (float, optional): The upper bound of the truncated distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def uniform(low=0.0, high=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Uniform distribution.

        Args:
            low (float, optional): The lower bound of the uniform distribution. Default is 0.0.
            high (float, optional): The upper bound of the uniform distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def unit(log_factor, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Unit distribution.

        Args:
            log_factor: The log factor of the unit distribution.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args).expand(shape))

    @staticmethod
    def vonmises(loc, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        VonMises distribution.

        A circular distribution centered at `loc` with concentration parameter.

        Args:
            loc: The location parameter.
            concentration: The concentration parameter.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def weibull(scale, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Weibull distribution.

        Args:
            scale: The scale parameter.
            concentration: The concentration parameter.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflateddistribution(base_dist, gate=None, gate_logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedDistribution.

        Args:
            base_dist: The base distribution.
            gate (float, optional): The probability of zero. Default is None.
            gate_logits (float, optional): The logit of the probability of zero. Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflatednegativebinomial2(mean, concentration, gate=None, gate_logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedNegativeBinomial2 distribution.

        Args:
            mean: The mean parameter of the negative binomial distribution.
            concentration: The concentration parameter of the negative binomial distribution.
            gate (float, optional): The probability of zero. Default is None.
            gate_logits (float, optional): The logit of the probability of zero. Default is None.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflatedpoisson(gate, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedPoisson distribution.

        Args:
            gate: The probability of zero.
            rate (float, optional): The rate parameter of the Poisson distribution. Default is 1.0.
            validate_args (bool, optional): Whether to validate the arguments. Default is None.
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def kl_divergence( shape=(), sample = False, seed = 0, name = 'x', *args, **kwargs):
        """
        Kullback-Leibler (KL) Divergence distribution.

        Computes the KL divergence between two distributions.

        Args:
            shape (tuple, optional): The shape of the samples to be drawn. Default is ().
            sample (bool, optional): Whether to return a sample. Default is False.
            seed (int, optional): The random seed. Default is 0.
            name (str, optional): The name of the sample. Default is 'x'.
            *args: Arguments for the first distribution.
            **kwargs: Keyword arguments for the second distribution.

        Returns:
            If sample=True: Tensor of shape `shape` with samples from the distribution.
            If sample=False: A numpyro distribution object that can be used in probabilistic models.
        """
        if sample == True:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.kl_divergence(args=args, kwargs=kwargs).sample(seed, shape)
        else: 
            return numpyro.sample(name, numpyro.distributions.kl_divergence(args=args, kwargs=kwargs).expand(shape))

