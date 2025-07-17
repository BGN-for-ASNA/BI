import jax
from jax import random
import numpyro

class UnifiedDist:

    def __init__(self):
        pass

    def mask(self,mask):
        return numpyro.handlers.mask(mask=mask)

    def plate(self,name, shape):
        return numpyro.plate(name, shape)

    @staticmethod
    def asymmetric_laplace(loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""AsymmetricLaplace
        
        Samples from an Asymmetric Laplace distribution.
        
        The Asymmetric Laplace distribution is a generalization of the Laplace distribution,
        where the two sides of the distribution are scaled differently. It is defined by
        a location parameter (loc), a scale parameter (scale), and an asymmetry parameter (asymmetry).
        
        .. math::
            f(x) = \frac{1}{2 \text{scale}} \left( \frac{1}{\text{scale}} \exp\left(-\frac{|x - \text{loc}|}{\text{scale} \cdot \text{asymmetry}}}\right) \text{ if } x < \text{loc} + \text{scale} \cdot \text{asymmetry}
            + \frac{1}{\text{scale}} \exp\left(-\frac{|x - \text{loc}|}{\text{scale} / \text{asymmetry}}\right) \text{ if } x > \text{loc} - \text{scale} / \text{asymmetry}
        
        Args:
            loc (jnp.ndarray or float): Location parameter of the distribution.
            
            scale (jnp.ndarray or float): Scale parameter of the distribution.
        
            asymmetry (jnp.ndarray or float): Asymmetry parameter of the distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro AsymmetricLaplace distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the AsymmetricLaplace distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.asymmetric_laplace(loc=0.0, scale=1.0, asymmetry=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplace
        """

           
        d = numpyro.distributions.continuous.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=0.5, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""AsymmetricLaplaceQuantile
        
        Samples from an AsymmetricLaplaceQuantile distribution.
        
        This distribution is an alternative parameterization of the AsymmetricLaplace
        distribution, commonly used in Bayesian quantile regression. It utilizes a
        `quantile` parameter to define the balance between the left- and right-hand
        sides of the distribution, representing the proportion of probability density
        that falls to the left-hand side.
        
        .. math::
           f(x) = \frac{1}{2 \sigma} \exp\left(-\frac{|x - \mu|}{\sigma} \frac{1}{q-1}\right) \left(1 - \frac{1}{2q}\right)
        
        Args:
            loc (float): The location parameter of the distribution.
        
            scale (float): The scale parameter of the distribution.
        
            quantile (float): The quantile parameter, representing the proportion of
                probability density to the left of the median. Must be between 0 and 1.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro AsymmetricLaplaceQuantile distribution object
                (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the
                AsymmetricLaplaceQuantile distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced
                use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=0.5, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplacequantile
        """

           
        d = numpyro.distributions.continuous.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def bernoulli_logits(logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""BernoulliLogits
        
        Samples from a Bernoulli distribution parameterized by logits.
        
        The Bernoulli distribution models a single binary event (success or failure),
        parameterized by the log-odds ratio of success.  The probability of success
        is given by the sigmoid function applied to the logit.
        
        .. math::
            P(x) = \sigma(logits)
        
        Args:
            logits (jnp.ndarray, optional): Log-odds ratio of success.  Must be real-valued.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int, optional): The number of batch dimensions to reinterpret as
                event dimensions (used in model building). Defaults to 0.
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask
                observations. Defaults to None.
        
            create_obj (bool, optional): If True, returns the raw NumPyro
                distribution object instead of creating a sample site. This is
                essential for building complex distributions like `MixtureSameFamily`.
                Defaults to False.
        
        Returns:
            When `sample=False`: A NumPyro BernoulliLogits distribution object (for
                model building).
        
            When `sample=True`: A JAX array of samples drawn from the BernoulliLogits
                distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced
                use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli_logits(logits=jnp.array([0.2, 1, 2]), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#bernoulli-logits
        """

           
        d = numpyro.distributions.discrete.BernoulliLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def bernoulli_probs(probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""BernoulliProbs
        
        Samples from a Bernoulli distribution parameterized by probabilities.
        
        The Bernoulli distribution models the probability of success in a single trial,
        where the outcome is binary (success or failure). It is characterized by a single
        parameter, `probs`, representing the probability of success.
        
        .. math::
            P(X=1) = p
            
            where:
                p is the probability of success (0 <= p <= 1)
        
        Args:
            probs (jnp.ndarray): The probability of success for each Bernoulli trial.
                Must be between 0 and 1.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            BernoulliProbs: A NumPyro BernoulliProbs distribution object (for model
                building).
        
            jnp.ndarray: A JAX array of samples drawn from the BernoulliProbs
                distribution (for direct sampling).
        
            NumPyro distribution object: The raw NumPyro distribution object (for
                advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli_probs(probs=jnp.array([0.2, 0.7, 0.5]), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#bernoulliprobs
        """

           
        d = numpyro.distributions.discrete.BernoulliProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def beta(concentration1, concentration0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Beta Distribution
        
        Samples from a Beta distribution, defined on the interval [0, 1].
        The Beta distribution is a versatile distribution often used to model
        probabilities or proportions. It is parameterized by two positive shape
        parameters, often referred to as concentration parameters in the NumPyro
        context.
        
        .. math::
            f(x) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}
        
        where :num:math:`\alpha` and :num:math:`\beta` are the concentration parameters,
        and :num:math:`B(x, y)` is the Beta function.
        
        Args:
            concentration1 (jnp.ndarray): The first concentration parameter (shape parameter). Must be positive.
            concentration0 (jnp.ndarray): The second concentration parameter (shape parameter). Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            Beta: A NumPyro Beta distribution object (for model building).
            jnp.ndarray: A JAX array of samples drawn from the Beta distribution (for direct sampling).
            Beta: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.beta(concentration1=1.0, concentration0=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#beta
        """

           
        d = numpyro.distributions.continuous.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def beta_binomial(concentration1, concentration0, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""BetaBinomial Distribution
        
        Samples from a BetaBinomial distribution, a compound distribution where the probability of success in a binomial
        experiment is drawn from a Beta distribution. This models situations where the underlying probability of success
        is not fixed but varies according to a prior belief represented by the Beta distribution.
        
        .. math::
           P(X=k) = \binom{n}{k} \frac{\Gamma(\alpha + k)}{\Gamma(\alpha + \beta + n - k)} \frac{\Gamma(\beta + n - k)}{\Gamma(\beta)}
        
        Args:
            concentration1 (jnp.ndarray): The first concentration parameter (alpha) of the Beta distribution.
            concentration0 (jnp.ndarray): The second concentration parameter (beta) of the Beta distribution.
            total_count (jnp.ndarray): The number of Bernoulli trials in the Binomial part of the distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro BetaBinomial distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the BetaBinomial distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.beta_binomial(concentration1=1.0, concentration0=1.0, total_count=10, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#betabinomial
        """

           
        d = numpyro.distributions.conjugate.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def beta_proportion(mean, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a BetaProportion distribution.
        
        The BetaProportion distribution is a reparameterization of the conventional
        Beta distribution in terms of a the variate mean and a
        precision parameter. It's useful for modeling rates and proportions.
        
        .. math::
            f(x) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}
        
        Args:
            mean (jnp.ndarray): The mean of the BetaProportion distribution,
                must be between 0 and 1.
        
            concentration (jnp.ndarray): The concentration parameter of the BetaProportion distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.
        
        Returns:
            A NumPyro BetaProportion distribution object (for model building).
            A JAX array of samples drawn from the BetaProportion distribution
                (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            samples = m.dist.beta_proportion(mean=0.5, concentration=2.0, sample=True, shape=(1000,))
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#beta_proportion
        """

           
        d = numpyro.distributions.continuous.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def binomial_logits(logits, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""BinomialLogits Distribution
        
        The BinomialLogits distribution represents a binomial distribution parameterized by logits.
        It is useful when the probability of success is not directly known but is instead
        expressed as logits, which are the natural logarithm of the odds ratio.
        
        .. math::
            P(X=k) = \binom{n}{k} \frac{e^{logits_k}}{1 + e^{logits_k}}
        
        Args:
            logits (jnp.ndarray): Log-odds of each success.
            total_count (int): Number of trials.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro BinomialLogits distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the BinomialLogits distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial_logits(logits=jnp.zeros(10), total_count=5, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#binomialllogits
        """

           
        d = numpyro.distributions.discrete.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def binomial_probs(probs, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""BinomialProbs
        
        Samples from a Binomial distribution with specified probabilities for each trial.
        
        The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials, where each trial has the same probability of success.
        
        .. math::
           P(k) = \binom{n}{k} p^k (1-p)^{n-k}
        
        Args:
            probs (jnp.ndarray): The probability of success for each trial. Must be between 0 and 1.
        
            total_count (int): The number of trials in each sequence.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro BinomialProbs distribution object (for model building).
        
            JAX array of samples drawn from the BinomialProbs distribution (for direct sampling).
        
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial_probs(probs=0.5, total_count=10, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#binomialprobs
        """

           
        d = numpyro.distributions.discrete.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def car(loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):

        r"""
        Conditional Autoregressive (CAR) Distribution
        
        The CAR distribution models a vector of variables where each variable is a linear
        combination of its neighbors in a graph.
        
        .. math::
        
           p(x) = \prod_{i=1}^{K} \mathcal{N}(x_i | \mu_i, \Sigma_i)
        
        where :math:`\mu_i` is a function of the values of the neighbors of site :math:`i`
        and :math:`\Sigma_i` is the variance of site :math:`i`.
        
        .. note::
        
           The CAR distribution is a special case of the multivariate normal distribution.
           It is used to model spatial data, such as temperature or precipitation.
        
        Args:
            loc (Union[float, Array]): Mean of the distribution.
            correlation (Union[float, Array]): Correlation between variables.
            conditional_precision (Union[float, Array]): Precision of the distribution.
            adj_matrix (Union[Array, scipy.sparse.spmatrix]): Adjacency matrix defining the graph.
            is_sparse (bool): Whether the adjacency matrix is sparse. Defaults to False.
            validate_args (bool): Whether to validate arguments. Defaults to None.
        """

        
           
        d = numpyro.distributions.continuous.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def categorical_logits(logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""CategoricalLogits
        
        Samples from a Categorical distribution with logits. This distribution represents a discrete probability distribution over a finite set of outcomes, where the probabilities are determined by the logits. The probability of each outcome is given by the softmax function applied to the logits.
        
        .. math::
            P(k) = \frac{e^{logits_k}}{\sum_{j=1}^{K} e^{logits_j}}
        
        Args:
            logits (jnp.ndarray): Log-odds of each category.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro CategoricalLogits distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the CategoricalLogits distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical_logits(logits=jnp.zeros(5), sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#categoricallogits
        """

           
        d = numpyro.distributions.discrete.CategoricalLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def categorical_probs(probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""CategoricalProbs
        
        Samples from a Categorical distribution.
        
        The Categorical distribution is a discrete probability distribution that
        represents the probability of each outcome from a finite set of possibilities.
        It is often used to model the outcome of a random process with a fixed number
        of possible outcomes, such as the roll of a die or the selection of an item
        from a list.
        
        .. math::
           P(x) = \frac{probs_i}{\sum_{k=1}^{K} probs_k}
        
        Args:
            probs (jnp.ndarray): Probabilities for each category. Must sum to 1.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro CategoricalProbs distribution object (for model building).
            JAX array of samples drawn from the CategoricalProbs distribution (for
            direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical_probs(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#categoricalprobs
        """

           
        d = numpyro.distributions.discrete.CategoricalProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def cauchy(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Cauchy Distribution
        
        Samples from a Cauchy distribution.
        
        The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
        that arises frequently in various fields, including physics and statistics. It is characterized by its
        heavy tails, which extend indefinitely.
        
        .. math::
            f(x) = \frac{1}{\pi \gamma} \left[ \frac{\gamma^2}{(x - \mu)^2 + \gamma^2} \right]
        
        Args:
            loc (jnp.ndarray or float, optional): Location parameter.  Defaults to 0.0.
            scale (jnp.ndarray or float, optional): Scale parameter. Must be positive. Defaults to 1.0.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                Defaults to None.
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to False.
        
        Returns:
            When `sample=False`: A NumPyro Cauchy distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Cauchy distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.cauchy(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#cauchy
        """

           
        d = numpyro.distributions.continuous.Cauchy(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def chi2(df, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Chi-squared distribution.
        
        The Chi-squared distribution is a continuous probability distribution that arises
        frequently in hypothesis testing, particularly in ANOVA and chi-squared tests.
        It is defined by a single positive parameter, degrees of freedom (df), which
        determines the shape of the distribution.
        
        .. math::
           p(x; df) = \frac{1}{2^{df/2} \Gamma(df/2)} x^{df/2 - 1} e^{-x/2}
        
        Args:
            df (jnp.ndarray): Degrees of freedom. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Chi2 distribution object (when `sample=False`).
            JAX array of samples drawn from the Chi2 distribution (when `sample=True`).
            The raw NumPyro distribution object (when `create_obj=True`).
        
        Example Usage:
            from BI import bi
            m = bi('cpu)
            m.dist.chi2(df=3.0, sample = True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#chi2
        """

           
        d = numpyro.distributions.continuous.Chi2(df=df, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def circulant_normal(loc: jax.Array, covariance_row: jax.Array = None, covariance_rfft: jax.Array = None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Multivariate normal distribution with covariance matrix :math:`\mathbf{C}` that is
        positive-definite and circulant [1], i.e., has periodic boundary conditions. The
        density of a sample :math:`\mathbf{x}\in\mathbb{R}^n` is the standard multivariate
        normal density
        
        .. math::
        
            p\left(\mathbf{x}\mid\boldsymbol{\mu},\mathbf{C}\right) =
            \frac{\left(\mathrm{det}\,\mathbf{C}\right)^{-1/2}}{\left(2\pi\right)^{n / 2}}
            \exp\left(-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\intercal
            \mathbf{C}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}\right)\right),
        
        where :math:`\mathrm{det}` denotes the determinant and :math:`^\intercal` the
        transpose. Circulant matrices can be diagnolized efficiently using the discrete
        Fourier transform [1], allowing the log likelihood to be evaluated in
        :math:`n \log n` time for :math:`n` observations [2].
        
        :param loc: Mean of the distribution :math:`\boldsymbol{\mu}`.
        :param covariance_row: First row of the circulant covariance matrix
            :math:`\boldsymbol{C}`. Because of periodic boundary conditions, the covariance
            matrix is fully determined by its first row (see
            :func:`jax.scipy.linalg.toeplitz` for further details).
        :param covariance_rfft: Real part of the real fast Fourier transform of
            :code:`covariance_row`, the first row of the circulant covariance matrix
            :math:`\boldsymbol{C}`.
        
        References:
        
        1. Wikipedia. (n.d.). Circulant matrix. Retrieved March 6, 2025, from
           https://en.wikipedia.org/wiki/Circulant_matrix
        2. Wood, A. T. A., & Chan, G. (1994). Simulation of Stationary Gaussian Processes in
           :math:`\left[0, 1\right]^d`. *Journal of Computational and Graphical Statistics*,
           3(4), 409--432. https://doi.org/10.1080/10618600.1994.10474655
        
        Parameters
        ----------
        loc : jnp.ndarray
            Mean of the distribution :math:`\boldsymbol{\mu}`.
        covariance_row : jnp.ndarray, optional
            First row of the circulant covariance matrix :math:`\mathbf{C}`.
            Defaults to None.
        covariance_rfft : jnp.ndarray, optional
            Real part of the real fast Fourier transform of :code:`covariance_row`.
            Defaults to None.
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#normal
        """

           
        d = numpyro.distributions.continuous.CirculantNormal(loc=loc, covariance_row=covariance_row, covariance_rfft=covariance_rfft, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def delta(v=0.0, log_density=0.0, event_dim=0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Delta distribution.
        
        The Delta distribution, also known as a point mass distribution, assigns probability 1 to a single point and 0 elsewhere. It's useful for representing deterministic variables or as a building block for more complex distributions.
        
        .. math::
           P(x = v) = 1
        
        Args:
            v (jnp.ndarray): The location of the point mass.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Delta distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Delta distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.delta(v=0.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#delta
        """

           
        d = numpyro.distributions.distribution.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def dirichlet(concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Dirichlet
        
        Samples from a Dirichlet distribution.
        
        The Dirichlet distribution is a multivariate generalization of the Beta distribution.
        It is a probability distribution over a simplex, which is a set of vectors where each element is non-negative and sums to one.
        It is often used as a prior distribution for categorical distributions.
        
        .. math::
           P(x_1, ..., x_K) = \frac{\Gamma(\sum_{i=1}^K \alpha_i)}{\prod_{i=1}^K \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i - 1}
        
        Args:
            concentration (jnp.ndarray): The concentration parameter(s) of the Dirichlet distribution.
                Must be a positive array.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        
        Returns:
            NumPyro Dirichlet distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Dirichlet distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.dirichlet(concentration=jnp.array([1.0, 1.0, 1.0]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#dirichlet
        """

           
        d = numpyro.distributions.continuous.Dirichlet(concentration=concentration, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def dirichlet_multinomial(concentration, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""DirichletMultinomial
        
        Samples from a DirichletMultinomial distribution.
        
        This distribution combines a Dirichlet distribution (for the probabilities of categories)
        and a Multinomial distribution (for the counts within each category).  The Dirichlet
        distribution acts as a prior on the probabilities, allowing for a flexible and
        informative model.
        
        .. math::
           P(x | \alpha, n) = \frac{n!}{x_1! x_2! \cdots x_k!} \frac{\Gamma(\alpha)}{\Gamma(\alpha_1) \Gamma(\alpha_2) \cdots \Gamma(\alpha_k)}
        
        Args:
            concentration (jnp.ndarray): Concentration parameter (alpha) for the Dirichlet distribution.
                Shape: (..., K), where K is the number of categories.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape. When
                `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX
                array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro DirichletMultinomial distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the DirichletMultinomial distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.dirichlet_multinomial(concentration=jnp.array([1.0, 1.0, 1.0]), total_count=10, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#dirichletmultinomial
        """

           
        d = numpyro.distributions.conjugate.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def discrete_uniform(low=0, high=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""DiscreteUniform
        
        Samples from a Discrete Uniform distribution.
        
        The Discrete Uniform distribution defines a uniform distribution over a range of integers.
        It is characterized by a lower bound (`low`) and an upper bound (`high`), inclusive.
        
        .. math::
           P(X = k) = \frac{1}{high - low + 1}, \quad k \in \{low, low+1, ..., high\}
        
        Args:
            low (jnp.ndarray): The lower bound of the uniform range, inclusive.
            high (jnp.ndarray): The upper bound of the uniform range, inclusive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro DiscreteUniform distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Discrete Uniform distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.discrete_uniform(low=0, high=5, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#discreteuniform
        """

           
        d = numpyro.distributions.discrete.DiscreteUniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def doubly_truncated_power_law(alpha, low, high, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        r"""
        Doubly Truncated Power Law distribution.

        This distribution represents a continuous power law with a finite support bounded
        between `low` and `high`, and with an exponent `alpha`. It is normalized over the
        interval `[low, high]` to ensure the area under the density function is 1.

        The probability density function (PDF) is defined as:

        .. math::
            f(x; \\alpha, a, b) = \\frac{x^{\\alpha}}{Z(\\alpha, a, b)}

        where the normalization constant :math:`Z(\\alpha, a, b)` is given by:

        .. math::
            Z(\\alpha, a, b) = \\begin{cases}
                \\log(b) - \\log(a) & \\text{if } \\alpha = -1, \\\\
                \\frac{b^{1 + \\alpha} - a^{1 + \\alpha}}{1 + \\alpha} & \\text{otherwise}.
            \\end{cases}

        This distribution is useful for modeling data that follows a power-law behavior
        but is naturally bounded due to measurement or theoretical constraints (e.g.,
        finite-size systems).


        Args:
            alpha (float or array-like): Power-law exponent.

            low (float or array-like): Lower bound of the distribution (must be ≥ 0).

            high (float or array-like): Upper bound of the distribution (must be > 0).

            shape (tuple, optional): The shape of the output tensor. Defaults to None.

            sample_shape (tuple, optional): The shape of the samples to draw. Defaults to None.
            
            validate_args (bool, optional): Whether to validate the arguments. Defaults to True.

        Example usage:
            from BI import bi
            m = bi('cpu')        
            m.dist.doubly_truncated_power_law(low=0.1, high=10.0, alpha=2.0, sample=True)
        """

        d = numpyro.distributions.truncated.DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def euler_maruyama(t, sde_fn, init_dist, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,   shape=(), event=0,create_obj=False):
        

        r"""
        Euler–Maruyama method is a method for the approximate numerical solution
        of a stochastic differential equation (SDE). It simulates the solution
        to an SDE by iteratively applying the Euler method to each time step,
        incorporating a random perturbation to account for the diffusion term.
        
        .. math::
            dX_t = f(X_t, t) dt + g(X_t, t) dW_t
        
        where:
        - :math:`X_t` is the state of the system at time :math:`t`.
        - :math:`f(X_t, t)` is the drift coefficient.
        - :math:`g(X_t, t)` is the diffusion coefficient.
        - :math:`dW_t` is a Wiener process (Brownian motion).
        
        Args:
            t (jnp.ndarray): Discretized time steps.
            sde_fn (callable): A function that takes the current state and time as input and returns the drift and diffusion coefficients.
            init_dist (Distribution): The initial distribution of the system.
            shape (tuple, optional): The shape of the output tensor. Defaults to None.
            sample_shape (tuple, optional): The shape of the samples to draw. Defaults to None.
            validate_args (bool, optional): Whether to validate the arguments. Defaults to True.
        
        Returns:
            jnp.ndarray: Samples drawn from the Euler–Maruyama distribution.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.euler_maruyama(t=jnp.array([0.0, 0.1, 0.2]), sde_fn=lambda x, t: (x, 1.0), init_dist=m.dist.normal(0.0, 1.0, create_obj=True), sample = True)
        """

           
        d = numpyro.distributions.continuous.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def exponential(rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Samples from an Exponential distribution.
        
        The Exponential distribution is a continuous probability distribution that models the time until an event occurs in a Poisson process, where events occur continuously and independently at a constant average rate. It is often used to model the duration of events, such as the time until a machine fails or the length of a phone call.
        
        .. math::
           f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0
        
        Args:
            rate (jnp.ndarray): The rate parameter, :math:`\lambda`. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Exponential distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Exponential distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.exponential(rate=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#exponential
        """

           
        d = numpyro.distributions.continuous.Exponential(rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def folded_distribution(base_dist, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Samples from a Folded distribution, which is the absolute value of a base univariate distribution.
        This distribution reflects the base distribution across the origin, effectively taking the absolute value of each sample.
    
        .. math::
            p(x) = \sum_{k=-\infty}^{\infty} p(x - 2k)
    
        Args:
            loc (float, optional): Location parameter of the base distribution. Defaults to 0.0.
            scale (float, optional): Scale parameter of the base distribution. Defaults to 1.0.
    
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
    
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
    
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
    
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
    
        Returns:
            NumPyro FoldedDistribution distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the FoldedDistribution distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
    
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.folded_distribution(m.dist.normal(loc=0.0, scale=1.0, create_obj = True), sample=True)
    
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#foldeddistribution
        """

           
        d = numpyro.distributions.distribution.FoldedDistribution(base_dist=base_dist, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gamma(concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Gamma Distribution
        
        Samples from a Gamma distribution.
        
        The Gamma distribution is a continuous probability distribution that arises frequently in Bayesian statistics,
        particularly in prior distributions for variance parameters. It is defined by two positive shape parameters,
        concentration (k) and rate (theta).
        
        .. math::
            f(x) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
        
        Args:
            concentration (jnp.ndarray): The shape parameter of the Gamma distribution (k > 0).
        
            rate (jnp.ndarray): The rate parameter of the Gamma distribution (theta > 0).
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        
        
        Returns:
            Gamma: A NumPyro Gamma distribution object (for model building).
        
            jnp.ndarray: A JAX array of samples drawn from the Gamma distribution (for direct sampling).
        
            Gamma: The raw NumPyro distribution object (for advanced use cases).
        
        
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gamma(concentration=2.0, rate=0.5, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#gamma
        """

           
        d = numpyro.distributions.continuous.Gamma(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gamma_poisson(concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""GammaPoisson Distribution
        
        A compound distribution comprising of a gamma-poisson pair, also referred to as
        a gamma-poisson mixture. The ``rate`` parameter for the
        :class:`~numpyro.distributions.Poisson` distribution is unknown and randomly
        drawn from a :class:`~numpyro.distributions.Gamma` distribution.
        
        .. math::
           P(X = x) = \int_0^\infty \frac{1}{x} \exp(-x \lambda) \frac{1}{\Gamma(\alpha)} x^{\alpha - 1} e^{-x \beta} dx
        
        Args:
            concentration (jnp.ndarray): Shape parameter (alpha) of the Gamma distribution.
            rate (jnp.ndarray): Rate parameter (beta) for the Gamma distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro GammaPoisson distribution object (for model
                building).
        
            When `sample=True`: A JAX array of samples drawn from the GammaPoisson
                distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced
                use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gamma_poisson(concentration=1.0, rate=2.0, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#gammapoisson
        """

           
        d = numpyro.distributions.conjugate.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gaussian_copula(marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        A distribution that links the `batch_shape[:-1]` of a marginal distribution with a multivariate Gaussian copula,
        modelling the correlation between the axes. A copula is a multivariate distribution over the uniform distribution
        on [0, 1]. The Gaussian copula links the marginal distributions through a multivariate normal distribution.
        
        .. math::
            f(x_1, ..., x_d) = \prod_{i=1}^{d} f_i(x_i) \cdot \phi(F_1(x_1), ..., F_d(x_d); \mu, \Sigma)
        
        where:
        - $f_i$ is the probability density function of the i-th marginal distribution.
        - $F_i$ is the cumulative distribution function of the i-th marginal distribution.
        - $\phi$ is the standard normal PDF.
        - $\mu$ is the mean vector of the multivariate normal distribution.
        - $\Sigma$ is the covariance matrix of the multivariate normal distribution.
        
        Args:
            marginal_dist (Distribution): Distribution whose last batch axis is to be coupled.
            correlation_matrix (array_like, optional): Correlation matrix of the coupling multivariate normal distribution. Defaults to None.
            correlation_cholesky (array_like, optional): Correlation Cholesky factor of the coupling multivariate normal distribution. Defaults to None.
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `Mi  xtureSameFamily`. Defaults to False.
        
        Returns:
            NumPyro GaussianCopula distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            NumPyro distribution object: When `create_obj=True` (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_copula(
                marginal_dist = m.dist.beta(2.0, 5.0, create_obj = True), 
                correlation_matrix = jnp.array([[1.0, 0.7],[0.7, 1.0]]), 
                sample = True
            )
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussiancopula
        """

           
        d = numpyro.distributions.copula.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gaussian_copula_beta(concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Gaussian Copula Beta distribution.
    
        This distribution combines a Gaussian copula with a Beta distribution.
        The Gaussian copula models the dependence structure between random variables,
        while the Beta distribution defines the marginal distributions of each variable.
    
        .. math::
            f(x) = \int_{-\infty}^{\infty} g(x|u) h(u) du
    
        Where:
            - g(x|u) is the Gaussian copula density.
            - h(u) is the Beta density.
    
        Args:
            concentration1 (jnp.ndarray): The first shape parameter of the Beta distribution.
            concentration0 (jnp.ndarray): The second shape parameter of the Beta distribution.
            correlation_cholesky (jnp.ndarray): The Cholesky decomposition of the correlation matrix.
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
                of the given shape.
    
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
    
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
    
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
    
        Returns:
            GaussianCopulaBeta: A NumPyro GaussianCopulaBeta distribution object (for model building).
            jnp.ndarray: A JAX array of samples drawn from the GaussianCopulaBeta distribution (for direct sampling).
            Distribution: The raw NumPyro distribution object (for advanced use cases).
    
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_copula_beta(
                concentration1 = jnp.array([2.0, 3.0]), 
                concentration0 = jnp.array([5.0, 3.0]),
                correlation_cholesky = jnp.linalg.cholesky(jnp.array([[1.0, 0.7],[0.7, 1.0]])), 
                sample = True
            )
    
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussiancopulabetadistribution
        """

           
        d = numpyro.distributions.copula.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gaussian_random_walk(scale=1.0, num_steps=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""GaussianRandomWalk
        
        Samples from a Gaussian Random Walk distribution.
        
        A Gaussian Random Walk is a stochastic process where each step is a Gaussian-distributed increment.
        It can be thought of as a discrete-time version of a Brownian motion.
        
        .. math::
           X_{t} = \sum_{i=1}^{t} \epsilon_i
        
        where :math:`\epsilon_i \sim \mathcal{N}(0, \sigma^2)` are independent Gaussian random variables.
        
        Args:
            scale (float): The standard deviation of the Gaussian increments.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        
        Returns:
            When `sample=False`: A NumPyro GaussianRandomWalk distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the GaussianRandomWalk distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_random_walk(scale=1.0, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#gaussianrandomwalk
        """

           
        d = numpyro.distributions.continuous.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gaussian_state_space(num_steps, transition_matrix, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        GaussianStateSpace Distribution
        
        Samples from a Gaussian state space model.
        
        .. math::
            \mathbf{z}_{t} &= \mathbf{A} \mathbf{z}_{t - 1} + \boldsymbol{\epsilon}_t\\
            &=\sum_{k=1} \mathbf{A}^{t-k} \boldsymbol{\epsilon}_t,
        
        where :math:`\mathbf{z}_t` is the state vector at step :math:`t`, :math:`\mathbf{A}`
        is the transition matrix, and :math:`\boldsymbol\epsilon` is the innovation noise.
        
        Args:
            num_steps (int): Number of steps.
            transition_matrix (jnp.ndarray): State space transition matrix :math:`\mathbf{A}`.
            covariance_matrix (jnp.ndarray, optional): Covariance of the innovation noise
                :math:`\boldsymbol{\epsilon}`. Defaults to None.
            precision_matrix (jnp.ndarray, optional): Precision matrix of the innovation noise
                :math:`\boldsymbol{\epsilon}`. Defaults to None.
            scale_tril (jnp.ndarray, optional): Scale matrix of the innovation noise
                :math:`\boldsymbol{\epsilon}`. Defaults to None.
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
            event (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. Defaults to False.
        
        Returns:
            When `sample=False`: A NumPyro GaussianStateSpace distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the GaussianStateSpace distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_state_space(num_steps=5, transition_matrix=jnp.array([[0.5]]), covariance_matrix =  jnp.array([[1.0, 0.6],[0.6, 1.0]]), sample=True)

        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussianstate
        """

           
        d = numpyro.distributions.continuous.GaussianStateSpace(num_steps=num_steps, transition_matrix=transition_matrix, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def geometric_logits(logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""GeometricLogits Distribution
        
        Samples from a GeometricLogits distribution, which models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is parameterized by logits, which are transformed into probabilities using the sigmoid function.
        
        .. math::
            P(X = k) = (1 - p)^k p
        
        where:
        
        *   X is the number of failures before the first success.
        *   k is the number of failures.
        *   p is the probability of success on each trial (derived from the logits).
        
        Args:
            logits (jnp.ndarray): Log-odds parameterization of the probability of success.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro GeometricLogits distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the GeometricLogits distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric_logits(logits=jnp.zeros(10), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#geometriclogits
        """

           
        d = numpyro.distributions.discrete.GeometricLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def geometric_probs(probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""GeometricProbs
        
        Samples from a Geometric distribution.
        
        The Geometric distribution models the number of trials until the first success in a sequence of independent Bernoulli trials, where each trial has the same probability of success.
        
        .. math::
            P(X = k) = (1 - p)^k p
        
        Args:
            probs (jnp.ndarray): Probability of success on each trial. Must be between 0 and 1.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro GeometricProbs distribution object (for model building).
            JAX array of samples drawn from the GeometricProbs distribution (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric_probs(probs=0.5, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#geometricprobs
        """

           
        d = numpyro.distributions.discrete.GeometricProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gompertz(concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Gompertz Distribution.
        
        The Gompertz distribution is a distribution with support on the positive real line that is closely
        related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
        entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.
        
        The probability density function (PDF) is:
        
        .. math::
            f(x) = \frac{con}{rate} \exp \left\{ - \frac{con}{rate} \left [ \exp\{x * rate \} - 1 \right ] \right\} \exp(-x * rate)
        
        Args:
            concentration (jnp.ndarray): The concentration parameter. Must be positive.
            rate (jnp.ndarray): The rate parameter. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Gompertz distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            NumPyro distribution object: When `create_obj=True` (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gompertz(concentration=1.0, rate=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gompertz
        """

           
        d = numpyro.distributions.continuous.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Gumbel
        
        Samples from a Gumbel (or Extreme Value) distribution.
        
        The Gumbel distribution is a continuous probability distribution named after German mathematician Carl Gumbel.
        It is often used to model the distribution of maximum values in a sequence of independent random variables.
        
        .. math::
           f(x) = \frac{1}{s} e^{-(x - \mu) / s} e^{-e^{- (x - \mu) / s}}
        
        Args:
            loc (jnp.ndarray or float, optional): Location parameter. Defaults to 0.0.
            scale (jnp.ndarray or float, optional): Scale parameter. Must be positive. Defaults to 1.0.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                Defaults to 1.
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to False.
        
        Returns:
            When `sample=False`: A NumPyro Gumbel distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Gumbel distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gumbel(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gumbel
        """

           
        d = numpyro.distributions.continuous.Gumbel(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def half_cauchy(scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""HalfCauchy Distribution
        
        The HalfCauchy distribution is a probability distribution that is half of the Cauchy distribution. It is defined on the positive real numbers and is often used in situations where only positive values are relevant.
        
        .. math::
           f(x) = \frac{1}{2} \cdot \frac{1}{\pi \cdot \frac{1}{scale} \cdot (x^2 + \frac{1}{scale^2})}
        
        Args:
            scale (jnp.ndarray): The scale parameter of the Cauchy distribution. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro HalfCauchy distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the HalfCauchy distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.half_cauchy(scale=1.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#halfcauchy
        """

           
        d = numpyro.distributions.continuous.HalfCauchy(scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def half_normal(scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""HalfNormal
        
        Samples from a HalfNormal distribution.
        
        The HalfNormal distribution is a distribution of the absolute value of a normal random variable.
        It is defined by a location parameter (implicitly 0) and a scale parameter.
        
        .. math::
           f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}} \text{ for } x > 0
        
        Args:
            scale (float, array): The scale parameter of the distribution. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro HalfNormal distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the HalfNormal distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.half_normal(scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#halfnormal
        """

           
        d = numpyro.distributions.continuous.HalfNormal(scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def improper_uniform(support, batch_shape, event_shape, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""ImproperUniform
        
        A helper distribution with zero :meth:`log_prob` over the `support` domain.
        
        .. math::
           p(x) = 0
        
        Args:
            support (numpyro.distributions.constraints.Constraint): The support of this distribution.
            batch_shape (tuple): Batch shape of this distribution. It is usually safe to
                set `batch_shape=()`.
            event_shape (tuple): Event shape of this distribution.
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
            event (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro ImproperUniform distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the ImproperUniform distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from numpyro import sample
            from numpyro.distributions import ImproperUniform, Normal, constraints
        
            def model():
                x = sample('x', ImproperUniform(constraints.ordered_vector, (), event_shape=(10,)))
        
            Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#improperuniform
        """

           
        d = numpyro.distributions.distribution.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def inverse_gamma(concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""InverseGamma Distribution
        
        The InverseGamma distribution is a two-parameter family of continuous probability
        distributions. It is defined by its shape and rate parameters. It is often used as a prior distribution for
        precision parameters (inverse variance) in Bayesian statistics.
        
        .. math::
            p(x) = \frac{1}{Gamma(\alpha)} \left( \frac{\beta}{\Gamma(\alpha)} \right)^{\alpha} x^{\alpha - 1} e^{-\beta x}
            \text{ for } x > 0
        
        Args:
            concentration (jnp.ndarray): The shape parameter (\\alpha) of the InverseGamma distribution. Must be positive.
            rate (jnp.ndarray): The rate parameter (\\beta) of the InverseGamma distribution. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro InverseGamma distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the InverseGamma distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.inverse_gamma(concentration=2.0, rate=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#inversegamma
        """

           
        d = numpyro.distributions.continuous.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def kumaraswamy(concentration1, concentration0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Kumaraswamy Distribution.
        
        The Kumaraswamy distribution is a continuous probability distribution defined on the interval [0, 1].
        It is a flexible distribution that can take on various shapes depending on its parameters.
        
        .. math::
            f(x; a, b) = a b x^{a b - 1} (1 - x)^{b - 1}
        
        Args:
            concentration1 (jnp.ndarray): The first shape parameter. Must be positive.
            concentration0 (jnp.ndarray): The second shape parameter. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Kumaraswamy distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Kumaraswamy distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.kumaraswamy(concentration1=2.0, concentration0=3.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#kumaraswamy
        """

           
        d = numpyro.distributions.continuous.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def lkj(dimension, concentration=1.0, sample_method='onion', validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Samples from an LKJ (Lewandowski, Kurowicka, Joe) distribution for correlation matrices.
        
        The LKJ distribution is controlled by the concentration parameter :math:`\eta` to make the probability
        of the correlation matrix :math:`M` proportional to :math:`\det(M)^{\eta - 1}`. When :math:`\eta = 1`,
        the distribution is uniform over correlation matrices.  When :math:`\eta > 1`, the distribution favors
        samples with large determinants. When :math:`\eta < 1`, the distribution favors samples with small
        determinants.
        
        .. math::
            P(M) \propto |\det(M)|^{\eta - 1}
        
        Args:
            dimension (int): The dimension of the correlation matrices.
        
            concentration (ndarray): The concentration/shape parameter of the distribution (often referred to as eta). Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro LKJ distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the LKJ distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.lkj(dimension=2, concentration=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#lkj
        """

           
        d = numpyro.distributions.continuous.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def lkj_cholesky(dimension, concentration=1.0, sample_method='onion', validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        LKJ Cholesky Distribution
        
        The LKJ (Leonard-Kjærgaard-Jørgensen) Cholesky distribution is a family of distributions
        on symmetric matrices, often used as a prior for the Cholesky decomposition of a
        symmetric matrix. It is particularly useful in Bayesian inference for models with
        covariance structure.
        
        Args:
            concentration (float): A parameter controlling the concentration of the distribution
                around the identity matrix. Higher values indicate greater concentration.
                Must be greater than 1.
        
        Attributes:
            concentration (float): The concentration parameter.
        """
        
        d = numpyro.distributions.continuous.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def laplace(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Laplace Distribution
        
        Samples from a Laplace distribution, also known as the double exponential distribution.
        The Laplace distribution is defined by its location parameter (loc) and scale parameter (scale).
        
        .. math::
           f(x) = \frac{1}{2s} \exp\left(-\frac{|x - \mu|}{s}\right)
        
        Args:
            loc (jnp.ndarray): Location parameter of the Laplace distribution.
        
            scale (jnp.ndarray): Scale parameter of the Laplace distribution. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Laplace distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Laplace distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.laplace(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#laplace
        """

           
        d = numpyro.distributions.continuous.Laplace(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def left_truncated_distribution(base_dist, low=0.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""LeftTruncatedDistribution
        
        Samples from a left-truncated distribution.
        
        A left-truncated distribution is a probability distribution
        obtained by restricting the support of another distribution
        to values greater than a specified lower bound. This is useful
        when dealing with data that is known to be greater than a certain value.
        
        .. math::
           f(x) = \begin{cases}
                   \frac{f(x)}{P(X > \text{low})} & \text{if } x > \text{low} \\
                   0 & \text{otherwise}
                   \end{cases}
        
        Args:
            base_dist: The base distribution to truncate. Must be univariate and have real support.
            low: The lower truncation bound. Values less than this are excluded from the distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro LeftTruncatedDistribution distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the LeftTruncatedDistribution distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        

        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#lefttruncateddistribution
        """

           
        d = numpyro.distributions.truncated.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def levy(loc, scale, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Levy distribution.
        
        Samples from a Levy distribution.
        
        The probability density function is given by,
        
        .. math::
            f(x\mid \mu, c) = \sqrt{\frac{c}{2\pi(x-\mu)^{3}}} \exp\left(-\frac{c}{2(x-\mu)}\right), \qquad x > \mu
        
        where :math:`\mu` is the location parameter and :math:`c` is the scale parameter.
        
        Args:
            loc (jnp.ndarray): Location parameter.
            scale (jnp.ndarray): Scale parameter.
        
        Args:
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
            event (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
        
        Returns:
            NumPyro Levy distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            NumPyro distribution object: When `create_obj=True` (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.levy(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#levy
        """

           
        d = numpyro.distributions.continuous.Levy(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def log_normal(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""LogNormal distribution.
        
        The LogNormal distribution is a probability distribution defined for positive real-valued random variables,
        parameterized by a location parameter (loc) and a scale parameter (scale).  It arises when the logarithm
        of a random variable is normally distributed.
        
        .. math::
           f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(log(x) - \mu)^2}{2\sigma^2}}
        
        Args:
            loc (float): Location parameter.
            scale (float): Scale parameter.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
                of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro LogNormal distribution object (for model building).
            JAX array of samples drawn from the LogNormal distribution (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.log_normal(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#lognormal
        """

           
        d = numpyro.distributions.continuous.LogNormal(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def log_uniform(low, high, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""LogUniform
        
        Samples from a LogUniform distribution.
        
        The LogUniform distribution is defined over the positive real numbers and is the result of applying an exponential transformation to a uniform distribution over the interval [low, high]. It is often used when modeling parameters that must be positive.
        
        .. math::
           f(x) = \frac{1}{(high - low) \log(high / low)}
           \text{ for } low \le x \le high
        
        Args:
            low (jnp.ndarray): The lower bound of the uniform distribution's log-space. Must be positive.
        
            high (jnp.ndarray): The upper bound of the uniform distribution's log-space. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro LogUniform distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the LogUniform distribution (for direct sampling) when `sample=True`.
        
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.log_uniform(low=0.1, high=10.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#loguniform
        """

           
        d = numpyro.distributions.continuous.LogUniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def logistic(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Logistic Distribution
        
        Samples from a Logistic distribution.
        
        The Logistic distribution is a continuous probability distribution defined by two parameters: location and scale. It is often used to model growth processes and is closely related to the normal distribution.
        
        .. math::
           f(x) = \frac{1}{s} \exp\left(-\frac{(x - \mu)}{s}\right)
        
        Args:
            loc (jnp.ndarray or float): The location parameter, specifying the median of the distribution. Defaults to 0.0.
        
            scale (jnp.ndarray or float): The scale parameter, which determines the spread of the distribution. Must be positive. Defaults to 1.0.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Logistic distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Logistic distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.logistic(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#logistic
        """

           
        d = numpyro.distributions.continuous.Logistic(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def low_rank_multivariate_normal(loc, cov_factor, cov_diag, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        LowRankMultivariateNormal Distribution
        
        Represents a multivariate normal distribution with a low-rank covariance structure.
        
        .. math::
        
           p(x) = \frac{1}{\sqrt{(2\pi)^K |\Sigma|}}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
        
        where:
        
        * :math:`x` is a vector of observations.
        * :math:`\mu` is the mean vector.
        * :math:`\Sigma` is the covariance matrix, represented in a low-rank form.
        
        Parameters:
            loc (jnp.ndarray): Mean vector.
            cov_factor (jnp.ndarray): Matrix used to construct the covariance matrix.
            cov_diag (jnp.ndarray): Diagonal elements of the covariance matrix.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            event_size = 100  # Our distribution has 100 dimensions
            rank = 5   
            m.dist.low_rank_multivariate_normal(
                loc=m.dist.normal(0,1, shape = (event_size,), sample=True)*2, 
                cov_factor=m.dist.normal(0,1, shape = (event_size, rank), sample=True),
                cov_diag=jnp.exp(m.dist.normal(0,1, shape = (event_size,), sample=True)) * 0.1, sample=True
            )
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#lowrankmultivariatenormal
        """

        d = numpyro.distributions.continuous.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def lower_truncated_power_law(alpha, low, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""LowerTruncatedPowerLaw
        
        Lower truncated power law distribution with :math:`\alpha` index.
        
        The probability density function (PDF) is given by:
        
        .. math::
            f(x; \alpha, a) = (-\alpha-1)a^{-\alpha - 1}x^{-\alpha},
            \qquad x \geq a, \qquad \alpha < -1,
        
        where :math:`a` is the lower bound.
        
        Args:
            alpha (jnp.ndarray): index of the power law distribution. Must be less than -1.
            low (jnp.ndarray): lower bound of the distribution. Must be greater than 0.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro LowerTruncatedPowerLaw distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the LowerTruncatedPowerLaw distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.lower_truncated_power_law(alpha=-2.0, low=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#lowertruncatedpowerlaw
        """

           
        d = numpyro.distributions.truncated.LowerTruncatedPowerLaw(alpha=alpha, low=low, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def matrix_normal(loc, scale_tril_row, scale_tril_column, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Matrix Normal Distribution
        
        Samples from a Matrix Normal distribution, which is a multivariate normal distribution over matrices.
        The distribution is characterized by a location matrix and two lower triangular matrices that define the correlation structure.
        The distribution is related to the multivariate normal distribution in the following way.
        If :math:`X ~ MN(loc,U,V)` then :math:`vec(X) ~ MVN(vec(loc), kron(V,U) )`.
        
        .. math::
            p(x) = \frac{1}{2\pi^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
        
        Args:
            loc (array_like): Location of the distribution.
        
            scale_tril_row (array_like): Lower cholesky of rows correlation matrix.
        
            scale_tril_column (array_like): Lower cholesky of columns correlation matrix.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro MatrixNormal distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the MatrixNormal distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')            
            n_rows, n_cols = 3, 4

            loc = jnp.zeros((n_rows, n_cols))
            U_row_cov = jnp.array([[1.0, 0.5, 0.2],
                                   [0.5, 1.0, 0.3],
                                   [0.2, 0.3, 1.0]])
            scale_tril_row = jnp.linalg.cholesky(U_row_cov)

            V_col_cov = jnp.array([[2.0, -0.8, 0.1, 0.4],
                                   [-0.8, 2.0, 0.2, -0.2],
                                   [0.1, 0.2, 2.0, 0.0],
                                   [0.4, -0.2, 0.0, 2.0]])

            # The argument passed to the distribution is its Cholesky factor
            scale_tril_column = jnp.linalg.cholesky(V_col_cov)

            m.dist.matrix_normal(
                oc=loc, 
                scale_tril_row=scale_tril_row, 
                scale_tril_column=scale_tril_column, 
                sample=True
            )
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase
        """

           
        d = numpyro.distributions.continuous.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def mixture_general(mixing_distribution, component_distributions, support=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        MixtureGeneral
        
        A finite mixture of component distributions from different families.
        
        :param mixing_distribution: A :class:`~numpyro.distributions.Categorical` specifying the weights for each mixture component. The size of this distribution specifies the number of components in the mixture.
        :param component_distributions: A list of `mixture_size` :class:`~numpyro.distributions.Distribution` objects.
        :param support: A :class:`~numpyro.distributions.constraints.Constraint` object specifying the support of the mixture distribution. If not provided, the support will be inferred from the component distributions.
        
        The probability density function (PDF) of a MixtureGeneral distribution is given by:
        
        $$
        p(x) = \sum_{i=1}^{K} \pi_i p_i(x)
        $$
        
        where:
        
        *   $K$ is the number of components in the mixture.
        *   $\pi_i$ is the mixing weight for the $i$-th component, such that $\sum_{i=1}^{K} \pi_i = 1$.
        *   $p_i(x)$ is the probability density function of the $i$-th component distribution.
        
        **Parameters:**
        
        *   **mixing_distribution**:  A `Categorical` distribution representing the mixing weights.
        *   **component_distributions**: A list of distributions representing the components of the mixture.
        
        **Returns:**
        
        *   **When `sample=False`**: A NumPyro MixtureGeneral distribution object (for model building).
        *   **When `sample=True`**: A JAX array of samples drawn from the MixtureGeneral distribution (for direct sampling).
        *   **When `create_obj=True`**: The raw NumPyro distribution object (for advanced use cases).
        
        **Example Usage:**      

            from BI import bi
            m = bi('cpu')
            m.dist.mixture_general(
                mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True),     
                component_distributions=[m.dist.normal(loc=0.0, scale=1.0, create_obj=True),m.dist.normal  (loc=0.0, scale=1.0,  create_obj=True)],
                sample = True
            )           

        Wrapper of: [https://num.pyro.ai/en/stable/distributions.html#mixturegeneral](https://num.pyro.ai/en/stable/distributions.html#mixturegeneral)
        """
        
           
        d = numpyro.distributions.mixtures.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def mixture_same_family(mixing_distribution, component_distribution, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        A finite mixture of component distributions from the same family.
        
        This mixture only supports a mixture of component distributions that are all
        of the same family. The different components are specified along the last
        batch dimension of the input ``component_distribution``. If you need a
        mixture of distributions from different families, use the more general
        implementation in :class:`~numpyro.distributions.MixtureGeneral`.
        
        .. math::
           p(x) = \sum_{k=1}^{K} w_k p_k(x)
        
        where:
        
        *   $K$ is the number of mixture components.
        *   $w_k$ is the mixing weight for component $k$.
        *   $p_k(x)$ is the probability density function (PDF) of the $k$-th component distribution.
        
        Args:
            *   **Distribution Args**:
                *   `loc` (jnp.ndarray): The location parameter of the component distribution.
                *   `scale` (jnp.ndarray): The scale parameter of the component distribution.
        
            *   **Sampling / Modeling Args**:
                *   `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                    with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                    used as `sample_shape` to draw a raw JAX array of the given shape.
                *   `event` (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                *   `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.
                *   `create_obj` (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            *   **When `sample=False`**: A NumPyro MixtureSameFamily distribution object (for model building).
            *   **When `sample=True`**: A JAX array of samples drawn from the MixtureSameFamily distribution (for direct sampling).
            *   **When `create_obj=True`**: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.mixture_same_family(
                mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True), 
                component_distribution=m.dist.normal(loc=0.0, scale=1.0, shape = (2,), create_obj=True),
                sample = True
            )
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#mixture-same-family
        """

           
        d = numpyro.distributions.mixtures.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def multinomial_logits(logits, total_count=1, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""MultinomialLogits
        
        Samples from a MultinomialLogits distribution.
        
        This distribution represents the probability of observing a specific outcome from a multinomial experiment,
        given the logits for each outcome. The logits are the natural logarithm of the odds of each outcome.
        
        .. math::
           P(k | \mathbf{\pi}) = \frac{n!}{k! (n-k)!} \prod_{i=1}^k \pi_i
        
        Args:
            logits (jnp.ndarray): Logits for each outcome. Must be at least one-dimensional.
        
            total_count (jnp.ndarray): The total number of trials.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro MultinomialLogits distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the MultinomialLogits distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial_logits(logits=jnp.array([1.0, 0.5], dtype=jnp.float32), total_count=jnp.array(5, dtype=jnp.int32), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#multinomiallogits
        """

           
        d = numpyro.distributions.discrete.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def multinomial_probs(probs, total_count=1, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""MultinomialProbs
        
        Samples from a Multinomial distribution.
        
        The Multinomial distribution models the number of times each of several discrete outcomes occurs in a fixed number of trials.  Each trial independently results in one of several outcomes, and each outcome has a probability of occurring.
        
        .. math::
           P(X = x) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}
        
        where:
        
        *   n is the total number of trials.
        *   x is a vector of counts for each outcome.
        *   p is a vector of probabilities for each outcome.
        
        Args:
            probs (jnp.ndarray): Vector of probabilities for each outcome. Must sum to 1.
            total_count (jnp.ndarray): The number of trials.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro MultinomialProbs distribution object (for model building).
            JAX array of samples drawn from the MultinomialProbs distribution (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial_probs(probs=jnp.array([0.2, 0.3, 0.5]), total_count=10, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#multinomialprobs
        """

           
        d = numpyro.distributions.discrete.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def multivariate_normal(loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Multivariate Normal distribution.
        
        The Multivariate Normal distribution, also known as the Gaussian distribution in multiple dimensions,
        is a probability distribution that arises frequently in statistics and machine learning. It is
        defined by its mean vector and covariance matrix, which describe the central tendency and
        spread of the distribution, respectively.
        
        .. math::
            p(x) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)
        
        where:
        - :math:`x` is a :math:`n`-dimensional vector of random variables.
        - :math:`\mu` is the mean vector.
        - :math:`\Sigma` is the covariance matrix.
        
        Args:
            loc (tuple): The mean vector of the distribution.
        
            covariance_matrix (jnp.ndarray, optional): The covariance matrix of the distribution. Must be positive definite.
        
            precision_matrix (jnp.ndarray, optional): The precision matrix (inverse of the covariance matrix) of the distribution. Must be positive definite.
        
            scale_tril (jnp.ndarray, optional): The lower triangular Cholesky decomposition of the covariance matrix.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro MultivariateNormal distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the MultivariateNormal distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:

        from BI import bi
        m = bi('cpu')
        m.dist.multivariate_normal(
            loc=jnp.array([1.0, 0.0, -2.0]), 
            covariance_matrix=jnp.array([[ 2.0,  0.7, -0.3],
                                        [ 0.7,  1.0,  0.5],
                                        [-0.3,  0.5,  1.5]]), 
            sample=True
        )

        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#multivariate-normal
        """

           
        d = numpyro.distributions.continuous.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def multivariate_student_t(df, loc=0.0, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Multivariate Student's t Distribution
        
        The Multivariate Student's t distribution is a generalization of the Student's t
        distribution to multiple dimensions. It is a heavy-tailed distribution that is
        often used to model data that is not normally distributed.
        
        .. math::
            p(x) = \frac{1}{B(df/2, n/2)} \frac{\Gamma(df/2 + n/2)}{\Gamma(df/2)}
            \left(1 + \frac{(x - \mu)^T \Sigma^{-1} (x - \mu)}{df}\right)^{-(df + n)/2}
        
        Args:
            df (jnp.ndarray): Degrees of freedom, must be positive.
            loc (jnp.ndarray): Location vector, representing the mean of the distribution.
            scale_tril (jnp.ndarray): Lower triangular matrix defining the scale.
        
        Distribution Args:
            df (jnp.ndarray): Degrees of freedom, must be positive.
            loc (jnp.ndarray): Location vector, representing the mean of the distribution.
            scale_tril (jnp.ndarray): Lower triangular matrix defining the scale.
        
        Sampling / Modeling Args:
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro MultivariateStudentT distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the MultivariateStudentT distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')
            m.dist.multivariate_student_t(
                df = 2,
                loc=jnp.array([1.0, 0.0, -2.0]), 
                scale_tril=jnp.linalg.cholesky(
                    jnp.array([[ 2.0,  0.7, -0.3],
                                [ 0.7,  1.0,  0.5],
                                [-0.3,  0.5,  1.5]])), 
                sample=True
            )   

        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#multivariatestudentt
        """

           
        d = numpyro.distributions.continuous.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def negative_binomial2(mean, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
            Samples from a NegativeBinomial2 distribution.
        
            This distribution is parameterized as a GammaPoisson with a modified rate.
            It represents the number of events occurring in a fixed amount of time or trials,
            where each event has a probability of success.
        
            .. math::
                P(k) = \frac{\Gamma(k + \alpha)}{\Gamma(k + 1) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^k \left(1 - \frac{\beta}{\alpha + \beta}\right)^k
        
            Args:
                mean (jnp.ndarray or float): The mean of the distribution.  This is equivalent to the `mu` parameter.
        
                concentration (jnp.ndarray or float): The concentration parameter. This is equivalent to the `alpha` parameter.
        
                shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
                event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
            Returns:
                NumPyro NegativeBinomial2 distribution object: When `sample=False` (for model building).
                jnp.ndarray: A JAX array of samples drawn from the NegativeBinomial2 distribution (for direct sampling).
                NumPyro NegativeBinomial2 distribution object: When `create_obj=True` (for advanced use cases).
        
            Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.negative_binomial2(mean=2.0, concentration=3.0, sample=True)
        
            Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomial2
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def negative_binomial_logits(total_count, logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""NegativeBinomialLogits
        
        Samples from a NegativeBinomialLogits distribution.
        
        The NegativeBinomialLogits distribution is a generalization of the Negative Binomial distribution where the parameter 'r' (number of successes) is expressed as a function of a logit parameter. This allows for more flexible modeling of count data.
        
        .. math::
        P(k) = \frac{e^{-n \cdot \text{softplus}(x)} \cdot \text{softplus}(-x)^k}{k!}
        
        Args:
            total_count (jnp.ndarray): The parameter controlling the shape of the distribution.  Represents the total number of trials.
        
            logits (jnp.ndarray): The log-odds parameter.  Related to the probability of success.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NegativeBinomialLogits: A NumPyro NegativeBinomialLogits distribution object (for model building).
        
            jnp.ndarray: A JAX array of samples drawn from the NegativeBinomialLogits distribution (for direct sampling).
        
            NegativeBinomialLogits: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial_logits(total_count=5.0, logits=0.0, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#negativebinomiallogits
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def negative_binomial_probs(total_count, probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a NegativeBinomial distribution with probabilities.
        
        The NegativeBinomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is characterized by two parameters: 'concentration' (r) and 'rate' (p).  In this implementation, the 'concentration' parameter is derived from 'total_count' and the 'rate' parameter is derived from 'probs'.
        
        .. math::
           P(k) = \binom{k + r - 1}{k} p^r (1 - p)^k
        
        Args:
            concentration (jnp.ndarray): The concentration parameter, derived from total_count.
            rate (jnp.ndarray): The rate parameter, derived from probs.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample  site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro NegativeBinomialProbs distribution object (for model building).
            JAX array of samples drawn from the NegativeBinomialProbs distribution (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial_probs(total_count=10.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomialprobs
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def normal(loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Normal Distribution
        
        Samples from a Normal (Gaussian) distribution.
        
        The Normal distribution is characterized by its mean (loc) and standard deviation (scale).
        It's a continuous probability distribution that arises frequently in statistics and
        probability theory.
        
        .. math::
           p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
        
        Args:
            loc (jnp.ndarray): The mean of the distribution.
        
            scale (jnp.ndarray): The standard deviation of the distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Normal distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the Normal distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.normal(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#normal
        """

           
        d = numpyro.distributions.continuous.Normal(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def ordered_logistic(predictor, cutpoints, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""OrderedLogistic
        
        A categorical distribution with ordered outcomes. This distribution represents the probability of an event falling into one of several ordered categories, based on a predictor variable and a set of cutpoints. The probability of an event falling into a particular category is determined by the number of categories above it.
        
        .. math::
           P(Y = k) = \begin{cases}
                1 & \text{if } k = 0 \\
                \frac{1}{k} & \text{if } k > 0
            \end{cases}
        
        Args:
            predictor (jnp.ndarray): Prediction in real domain; typically this is output of a linear model.
        
        Distribution Args:
            cutpoints (jnp.ndarray): Positions in real domain to separate categories.
        
        Sampling / Modeling Args:
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro OrderedLogistic distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the OrderedLogistic distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.ordered_logistic(predictor=jnp.array([0.2, 0.5, 0.8]), cutpoints=jnp.array([-1.0, 0.0, 1.0]), sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#orderedlogistic
        """

           
        d = numpyro.distributions.discrete.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def pareto(scale, alpha, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Pareto Distribution
        
        Samples from a Pareto distribution.
        
        The Pareto distribution is a power-law probability distribution that is often
        used to model income, wealth, and the size of cities. It is defined by two
        parameters: alpha (shape) and scale.
        
        .. math::
            f(x) = \frac{\alpha \cdot \text{scale}^{\alpha}}{x^{\alpha + 1}}
            \text{ for } x \geq \text{scale}
        
        Args:
            scale (jnp.ndarray or float): Scale parameter of the Pareto distribution.
                Must be positive.
        
            alpha (jnp.ndarray or float): Shape parameter of the Pareto distribution.
                Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Pareto distribution object (for model
                building).
        
            When `sample=True`: A JAX array of samples drawn from the Pareto
                distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced
                use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.pareto(scale=2.0, alpha=3.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#pareto
        """

           
        d = numpyro.distributions.continuous.Pareto(scale=scale, alpha=alpha, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def poisson(rate, is_sparse=False, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Creates a Poisson distribution, a discrete probability distribution that models the number of events occurring in a fixed interval of time or space if these events occur with a known average rate and independently of the time since the last event.
        
        .. math::
          \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}
        
        Args:
            rate (jnp.ndarray): The rate parameter, representing the average number of events.
        
        Args:
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Poisson distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Poisson distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.poisson(rate=2.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#poisson
        """

           
        d = numpyro.distributions.discrete.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def projected_normal(concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a ProjectedNormal distribution.
        
        This distribution over directional data is qualitatively similar to the von
        Mises and von Mises-Fisher distributions, but permits tractable variational
        inference via reparametrized gradients.
        
        .. math::
           p(x) = \frac{1}{Z} \exp\left(-\frac{1}{2\sigma^2} ||x - \mu||^2\right)
        
        Args:
            concentration (jnp.ndarray): The concentration parameter, representing the
                direction towards which the samples are concentrated.  Must be a
                JAX array with at least one dimension.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro ProjectedNormal distribution object (for
                model building).
        
            When `sample=True`: A JAX array of samples drawn from the
                ProjectedNormal distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for
                advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.projected_normal(concentration=jnp.array([1.0, 3.0, 2.0]), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#projectednormal
        """

           
        d = numpyro.distributions.directional.ProjectedNormal(concentration=concentration, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def relaxed_bernoulli_logits(temperature, logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Relaxed Bernoulli Logits Distribution.
        
            Represents a relaxed version of the Bernoulli distribution, parameterized by logits and a temperature.
            The temperature parameter controls the sharpness of the distribution.  The distribution is defined
            by transforming the output of a Logistic distribution through a sigmoid function.
        
            .. math::
                P(x) = \sigma\left(\frac{x}{\text{temperature}}\right)
        
            Args:
                temperature (jnp.ndarray): The temperature parameter, must be positive.
                logits (jnp.ndarray): The logits parameter.
        
                shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                    this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
                    (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
                event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.
        
            Returns:
                RelaxedBernoulliLogits: A NumPyro RelaxedBernoulliLogits distribution object (for model building).
                jnp.ndarray: A JAX array of samples drawn from the RelaxedBernoulliLogits distribution (for direct sampling).
                RelaxedBernoulliLogits: The raw NumPyro distribution object (for advanced use cases).
        
            Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.relaxed_bernoulli_logits(temperature=1.0, logits=0.0, sample=True)
        
            Wrapper of:
                https://num.pyro.ai/en/stable/distributions.html#relaxed-bernoulli-logits
        """

           
        d = numpyro.distributions.continuous.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def right_truncated_distribution(base_dist, high=0.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0, shape=(), event=0,create_obj=False):
        

        r"""RightTruncatedDistribution
        
        Samples from a right-truncated distribution.
        
        This distribution truncates the base distribution at a specified high value.  Values greater than `high` are discarded,
        effectively creating a distribution that is only supported up to that point. This is useful for modeling data
        where observations are only possible within a certain range.
        
        .. math::
           f(x) = \frac{f(x)}{P(X \le high)}
        
        where :math:`f(x)` is the probability density function (PDF) of the base distribution and :math:`P(X \le high)` is the
        cumulative distribution function (CDF) of the base distribution evaluated at `high`.
        
        Args:
            base_dist: The base distribution to truncate.  Must be a univariate distribution with real support.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro RightTruncatedDistribution distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the RightTruncatedDistribution distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.right_truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#righttruncateddistribution
        """

           
        d = numpyro.distributions.truncated.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def sine_skewed(base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        
        r"""
        Sine-skewing [1] is a procedure for producing a distribution that breaks pointwise symmetry on a torus distribution. The new distribution is called the Sine Skewed X distribution, where X is the name of the (symmetric) base distribution. Torus distributions are distributions with support on products of circles (i.e., :math:`\otimes S^1` where :math:`S^1 = [-pi,pi)`).
        So, a 0-torus is a point, the 1-torus is a circle, and the 2-torus is commonly associated with the donut shape.
        
        .. note: This distribution is available in NumPyro: [https://num.pyro.ai/en/stable/distributions.html#sineskewed](https://num.pyro.ai/en/stable/distributions.html#sineskewed)
        
        **Parameters:**
        
        *   **base\_dist:** Base density on a d-dimensional torus. Supported base distributions include: 1D :class:`~numpyro.distributions.VonMises`, :class:`~numnumpyro.distributions.SineBivariateVonMises`, 1D :class:`~numpyro.distributions.ProjectedNormal`, and :class:`~numpyro.distributions.Uniform` (-pi, pi).
        *   **skewness:** Skewness of the distribution.
        
        **Attributes:**
        
        *   **mean:** Mean of the base distribution.
        
        **PDF:**
        
        The probability density function (PDF) of the Sine Skewed X distribution is not explicitly defined here, but it is derived from the base distribution and the skewness parameter.
        
        **Example Usage:**
        

        from num.pyro import distributions as dist
        import num.pyro as pyro
        import num.numpy as np
        
        m = pyro.distributions.Normal(loc=0.0, scale=1.0)
        skewness = np.array([0.5, 0.5])
        sine_skewed = dist.SineSkewed(base_dist=m, skewness=skewness)
        samples = sine_skewed.sample((1000,))
        """
           
        d = numpyro.distributions.directional.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def soft_laplace(loc, scale, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""SoftLaplace
        
        Samples from a SoftLaplace distribution.
        
        This distribution is a smooth approximation of a Laplace distribution,
        characterized by its log-convex density. It offers Laplace-like tails
        while being infinitely differentiable, making it suitable for HMC and
        Laplace approximation.
        
        .. math::
            f(x) = \log(2 / \pi) - \log(scale) - \logaddexp((x - loc) / scale, -(x - loc) / scale)
        
        Args:
            loc: Location parameter.
            scale: Scale parameter.
        
        Args:
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro SoftLaplace distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the SoftLaplace distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.soft_laplace(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#softlaplace
        """

           
        d = numpyro.distributions.continuous.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def student_t(df, loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Student's t-distribution.
        
        The Student's t-distribution is a probability distribution that arises in hypothesis testing involving the mean of a normally distributed population when the population standard deviation is unknown. It is similar to the normal distribution, but has heavier tails, making it more robust to outliers.
        
        .. math::
            f(x) = \frac{1}{\Gamma(\nu/2) \sqrt{\nu \pi}} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}
        
        Args:
            df (jnp.ndarray): Degrees of freedom, must be positive.
            loc (jnp.ndarray): Location parameter, defaults to 0.0.
            scale (jnp.ndarray): Scale parameter, defaults to 1.0.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro StudentT distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the StudentT distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.student_t(df = 2, loc=0.0, scale=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#studentt
        """

           
        d = numpyro.distributions.continuous.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def truncated_polya_gamma(batch_shape=(), validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""TruncatedPolyaGamma Distribution
        
        Samples from a TruncatedPolyaGamma distribution.
        
        This distribution is a truncated version of the PolyaGamma distribution,
        defined over the interval [0, truncation_point]. It is often used in
        Bayesian non-parametric models.
        
        .. math::
           p(x) = \frac{1}{Z} \exp\left( \sum_{n=0}^{N} \left( \log(2n+1) - 1.5 \log(x) - \frac{(2n+1)^2}{4x} \right) \right)
        
        Args:
            batch_shape (tuple): The shape of the batch dimension.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions.
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
        
        Returns:
            When `sample=False`: A NumPyro TruncatedPolyaGamma distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the TruncatedPolyaGamma distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_polya_gamma(batch_shape=(), sample=True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#truncatedpolygammadistribution
        """

           
        d = numpyro.distributions.truncated.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def two_sided_truncated_distribution(base_dist, low=0.0, high=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        TwoSidedTruncatedDistribution
        
        This distribution truncates a base distribution between two specified lower and upper bounds.
        
        .. math::
            f(x) = \begin{cases}
                \frac{p(x)}{P(\text{low} \le X \le \text{high})} & \text{if } \text{low} \le x \le \text{high} \\
                0 & \text{otherwise}
            \end{cases}
        
        where :math:`p(x)` is the probability density function of the base distribution.
        
        Args:
            base_dist: The base distribution to truncate.
            low: The lower bound for truncation.
            high: The upper bound for truncation.
        
        Returns:
            When `sample=False`: A NumPyro TwoSidedTruncatedDistribution distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the TwoSidedTruncatedDistribution distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution
        """
           
        d = numpyro.distributions.truncated.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def uniform(low=0.0, high=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Uniform Distribution
        
        Samples from a Uniform distribution, which is a continuous probability distribution
        where all values within a given interval are equally likely.
        
        .. math::
           f(x) = \frac{1}{b - a}, \text{ for } a \le x \le b
        
        Args:
            low (jnp.ndarray): The lower bound of the uniform interval.
        
            high (jnp.ndarray): The upper bound of the uniform interval.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Uniform distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the Uniform distribution (for direct
            sampling) when `sample=True`.
        
            The raw NumPyro distribution object (for advanced use cases) when
            `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.uniform(low=0.0, high=1.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#uniform
        """

           
        d = numpyro.distributions.continuous.Uniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def unit(log_factor, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
            Samples from a Unit distribution.
        
            The Unit distribution is a trivial, non-normalized distribution representing the unit type.
            It has a single value with no data, effectively a placeholder often used in probabilistic programming
            for situations where no actual data is involved.
        
            .. math::
                p(x) = 1
        
            Args:
                log_factor (jnp.ndarray): Log factor for the unit distribution. This parameter determines the
                    shape and batch size of the distribution.
        
                shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                    this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
                    (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
                event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.
        
            Returns:
                NumPyro Unit distribution object: When `sample=False` (for model building).
                jnp.ndarray: A JAX array of samples drawn from the Unit distribution (for direct sampling).
                NumPyro Unit distribution object: When `create_obj=True` (for advanced use cases).
        
            Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.unit(log_factor=jnp.ones(5), sample=True)
        
            Wrapper of:
                https://num.pyro.ai/en/stable/distributions.html#unit
        """

           
        d = numpyro.distributions.distribution.Unit(log_factor=log_factor, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def weibull(scale, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Weibull Distribution
        
        Samples from a Weibull distribution.
        
        The Weibull distribution is a versatile distribution often used to model failure rates in engineering and reliability studies. It is characterized by its shape and scale parameters.
        
        .. math::
            f(x) = \frac{\beta}{\alpha} \left(\frac{x}{\alpha}\right)^{\beta - 1} e^{-\left(\frac{x}{\alpha}\right)^{\beta}} \text{ for } x \ge 0
        
        where :math:`\alpha` is the scale parameter and :math:`\beta` is the shape parameter.
        
        Args:
            scale (jnp.ndarray): The scale parameter of the Weibull distribution. Must be positive.
            concentration (jnp.ndarray): The shape parameter of the Weibull distribution. Must be positive.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Weibull distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Weibull distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.weibull(scale=1.0, concentration=2.0, sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#weibull
        """

           
        d = numpyro.distributions.continuous.Weibull(scale=scale, concentration=concentration, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def wishart(concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        Wishart distribution for covariance matrices.
        
        The Wishart distribution is a multivariate distribution used to model positive definite matrices,
        often representing covariance matrices. It's commonly used in Bayesian statistics and machine learning,
        particularly in models involving covariance estimation.
        
        .. math::
           p(X) = \frac{1}{W^{p/2} \Gamma_p(concentration/2)} \left|X\right|^{-concentration/2} \exp\left(-\frac{1}{2} \text{tr}(X^{-1} X)\right)
        
        Args:
            concentration (jnp.ndarray): Positive concentration parameter analogous to the
                concentration of a :class:`Gamma` distribution. The concentration must be larger
                than the dimensionality of the scale matrix.
        
            scale_matrix (jnp.ndarray, optional): Scale matrix analogous to the inverse rate of a :class:`Gamma`
                distribution.
        
            rate_matrix (jnp.ndarray, optional): Rate matrix anaologous to the rate of a :class:`Gamma`
                distribution.
        
            scale_tril (jnp.ndarray, optional): Cholesky decomposition of the :code:`scale_matrix`.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Wishart distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Wishart distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.wishart(concentration=5.0, scale_matrix=jnp.eye(2), sample=True)
        
        Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#wishart
        """

           
        d = numpyro.distributions.continuous.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def wishart_cholesky(concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
        WishartCholesky Distribution
        
        The Wishart distribution is a multivariate distribution used as a prior distribution
        for covariance matrices. This implementation represents the distribution in terms
        of its Cholesky decomposition.
        
        .. rubric:: Probability Density Function
        
        The probability density function (PDF) is given by:
        
        PDF = (1 / ((2 * pi)^(k * (k - 1) / 2) * Gamma(k/2)) *
               (concentration^(k/2) * det(scale_matrix))^(-1/2) *
               exp(-1/2 * trace(rate_matrix @ scale_matrix)))
        
        where:
        
        - k is the dimensionality of the covariance matrix.
        - concentration is a positive concentration parameter.
        - scale_matrix is the scale matrix.
        - rate_matrix is the rate matrix.
        - Gamma is the gamma function.
        
        .. rubric:: Parameters
        
        :param concentration: (Tensor) Positive concentration parameter analogous to the
            concentration of a :class:`Gamma` distribution. The concentration must be larger
            than the dimensionality of the scale matrix.
        :param scale_matrix: (Tensor, optional) Scale matrix analogous to the inverse rate of a :class:`Gamma`
                distribution. If not provided, `rate_matrix` or `scale_tril` must be.
        :param rate_matrix: (Tensor, optional) Rate matrix anaologous to the rate of a :class:`Gamma`
                distribution. If not provided, `scale_matrix` or `scale_tril` must be.
        :param scale_tril: (Tensor, optional) Cholesky decomposition of the :code:`scale_matrix`.
            If not provided, `scale_matrix` or `rate_matrix` must be.
        """
        
        d = numpyro.distributions.continuous.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def zero_inflated_poisson(gate, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""
            A Zero Inflated Poisson distribution.
        
            This distribution combines two Poisson processes: one with a rate parameter and another that generates only zeros.
            The probability of observing a zero is determined by the 'gate' parameter, while the probability of observing a non-zero value is governed by the 'rate' parameter of the underlying Poisson distribution.
        
            .. math::
                P(X = k) = (1 - gate) * \frac{e^{-rate} rate^k}{k!} + gate
        
            Args:
                rate (jnp.ndarray): The rate parameter of the underlying Poisson distribution.
        
                shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
                event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
            Returns:
                NumPyro ZeroInflatedPoisson distribution object (when `sample=False`).
                JAX array of samples drawn from the ZeroInflatedPoisson distribution (when `sample=True`).
                The raw NumPyro distribution object (when `create_obj=True`).
        
            Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.zero_inflated_poisson(gate = 0.3, rate=2.0, sample=True)
        
            Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflatedpoisson
        """

           
        d = numpyro.distributions.discrete.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def zero_sum_normal(scale, event_shape, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""ZeroSumNormal
        
        Samples from a ZeroSumNormal distribution, which is a Normal distribution where one or more axes are constrained to sum to zero.
        
        .. math::
            \begin{align*}
            ZSN(\sigma) = N(0, \sigma^2 (I - \tfrac{1}{n}J)) \\
            \text{where} \ ~ J_{ij} = 1 \ ~ \text{and} \\
            n = \text{number of zero-sum axes}
            \end{align*}
        
        Args:
            scale (array_like): Standard deviation of the underlying normal distribution before the zerosum constraint is enforced.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro ZeroSumNormal distribution object (for model building).
        
            When `sample=True`: A JAX array of samples drawn from the ZeroSumNormal distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_sum_normal(scale=1.0, event_shape = (2,), sample = True)
        
        Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#zerosumnormal
        """

           
        d = numpyro.distributions.continuous.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def bernoulli(probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Bernoulli distribution.
        
        The Bernoulli distribution models a single trial with two possible outcomes: success or failure.
        It is parameterized by the probability of success, often denoted as 'p'.
        
        .. math::
           P(X=1) = p \\
           P(X=0) = 1 - p
        
        Args:
            probs (jnp.ndarray, optional):  Probability of success for each Bernoulli trial. Must be between 0 and 1.
            logits (jnp.ndarray, optional): Log-odds of success for each Bernoulli trial.  `probs = sigmoid(logits)`.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro Bernoulli distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Bernoulli distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli(probs=0.7, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#bernoulli
        """

           
        d = numpyro.distributions.discrete.Bernoulli(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def binomial(total_count=1, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Binomial distribution.
        
        The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials.
        It represents the probability of obtaining exactly *k* successes in *n* trials, where each trial has a probability *p* of success.
        
        .. math::
           P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
        
        Args:
            total_count (int): The number of trials *n*.
        
            probs (jnp.ndarray, optional): The probability of success *p* for each trial. Must be between 0 and 1.
            
            logits (jnp.ndarray, optional): The log-odds of success for each trial.  `probs = jax.nn.sigmoid(logits)`.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            Binomial distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Binomial distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial(total_count=10, probs=0.5, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#binomial
        """

           
        d = numpyro.distributions.discrete.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def categorical(probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Categorical distribution.
        
        The Categorical distribution, also known as the multinomial distribution,
        describes the probability of different outcomes from a finite set of possibilities.
        It is commonly used to model discrete choices or classifications.
        
        .. math::
           P(k) = \frac{e^{\log(p_k)}}{\sum_{j=1}^{K} e^{\log(p_j)}}
        
        where :math:`p_k` is the probability of outcome :math:`k`, and the sum is over all possible outcomes.
        
        Args:
            probs (jnp.ndarray): A 1D array of probabilities for each category. Must sum to 1.
        
            Distribution Args:
                None
        
            Sampling / Modeling Args:
                shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                    (model building), this is used with `.expand(shape)` to set the
                    distribution's batch shape. When `sample=True` (direct sampling),
                    this is used as `sample_shape` to draw a raw JAX array of the
                    given shape.
        
                event (int): The number of batch dimensions to reinterpret as event
                    dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution
                    object instead of creating a sample site. This is essential for
                    building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Categorical distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Categorical distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#categorical
        """

           
        d = numpyro.distributions.discrete.Categorical(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def geometric(probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Geometric distribution.
        
        The Geometric distribution models the number of failures before the first success in a sequence of Bernoulli trials.
        It is characterized by a single parameter, the probability of success on each trial.
        
        .. math::
           P(X = k) = (1 - p)^k p
        
        Args:
            probs (jnp.ndarray, optional): Probability of success on each trial. Must be between 0 and 1.
            logits (jnp.ndarray, optional): Log-odds of success on each trial.  `probs = jax.nn.sigmoid(logits)`.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Geometric distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric(probs=0.5, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#geometric
        """

           
        d = numpyro.distributions.discrete.Geometric(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def mixture(mixing_distribution, component_distributions, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,  shape=(), event=0,create_obj=False):
        

        r"""A marginalized finite mixture of component distributions.
        
        This distribution represents a mixture of component distributions, where the
        mixing weights are determined by a Categorical distribution. The resulting
        distribution can be either a MixtureGeneral (when component distributions
        are a list) or a MixtureSameFamily (when component distributions are a single
        distribution).
        
        .. math::
           p(x) = \sum_{i=1}^{K} w_i p_i(x)
        
        Args:
            Distribution Args:
                None
        
            Sampling / Modeling Args:
                shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                    (model building), this is used with `.expand(shape)` to set the
                    distribution's batch shape. When `sample=True` (direct sampling),
                    this is used as `sample_shape` to draw a raw JAX array of the
                    given shape.
        
                event (int): The number of batch dimensions to reinterpret as event
                    dimensions (used in model building).
        
                mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
                create_obj (bool): If True, returns the raw NumPyro distribution
                    object instead of creating a sample site. This is essential for
                    building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Mixture distribution object (for model
                building).
        
            When `sample=True`: A JAX array of samples drawn from the Mixture
                distribution (for direct sampling).
        
            When `create_obj=True`: The raw NumPyro distribution object (for advanced
                use cases).
        
        Example Usage:
            from jax import random
            import numpyro as pyro
            m = pyro.distributions.Mixture(
                pyro.distributions.Categorical(torch.ones(2)),
                [pyro.distributions.Normal(0, 1), pyro.distributions.Normal(2, 1)]
            )
            samples = m.sample(sample_shape=(10,))
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#mixture
        """

           
        d = numpyro.distributions.mixtures.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def multinomial(total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Multinomial distribution.
        
        Samples from a Multinomial distribution, which models the probability of different outcomes in a sequence of independent trials, each with a fixed number of trials and a fixed set of possible outcomes.  It generalizes the binomial distribution to multiple categories.
        
        .. math::
           P(X = x) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}
        
        Args:
            total_count (int or jnp.ndarray): The number of trials.
        
            probs (jnp.ndarray, optional): Event probabilities. Must sum to 1.
        
            logits (jnp.ndarray, optional): Event log probabilities.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
            create_obj (bool, optional): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro Multinomial distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the Multinomial distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#multinomial
        """

           
        d = numpyro.distributions.discrete.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def relaxed_bernoulli(temperature, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Relaxed Bernoulli distribution.
        
        The Relaxed Bernoulli distribution is a continuous relaxation of the discrete Bernoulli distribution.
        It's useful for variational inference and other applications where a differentiable approximation of the Bernoulli is needed.
        The probability density function (PDF) is defined as:
        
        .. math::
            p(x) = \frac{1}{2} \left( 1 + \tanh\left(\frac{x - \beta \log(\frac{p}{1-p})}{1}\right) \right)
        
        Args:
            temperature (float): The temperature parameter.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro RelaxedBernoulli distribution object (for model building) when `sample=False`.
            A JAX array of samples drawn from the RelaxedBernoulli distribution (for direct sampling) when `sample=True`.
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.relaxed_bernoulli(temperature=1.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli
        """

           
        d = numpyro.distributions.continuous.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def truncated_cauchy(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Truncated Cauchy distribution.
        
        The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
        that appears frequently in various areas of mathematics and physics. It is characterized by its heavy tails,
        which extend to infinity. The truncated version limits the support of the Cauchy distribution to a specified interval.
        
        .. math::
            f(x) = \frac{1}{\pi \cdot c \cdot (1 + ((x - b) / c)^2)}  \text{ for } a < x < b
        
        Args:
            loc (float): Location parameter of the Cauchy distribution.
        
            scale (float): Scale parameter of the Cauchy distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro TruncatedCauchy distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the TruncatedCauchy distribution (for direct sampling) when `sample=True`.
        
            The raw NumPyro distribution object (for advanced use cases) when `create_obj=True`.
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_cauchy(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncatedcauchy
        """

           
        d = numpyro.distributions.truncated.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def truncated_distribution(base_dist, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""TruncatedDistribution
        
        Samples from a Truncated Distribution.
        
        This distribution represents a base distribution truncated between specified lower and upper bounds.
        The truncation modifies the probability density function (PDF) of the base distribution,
        effectively removing observations outside the defined interval.
        
        .. math::
           p(x) = \frac{p(x)}{P(\text{lower} \le x \le \text{upper})}
        
        Args:
            base_dist: The base distribution to be truncated. This should be a univariate
                distribution. Currently, only the following distributions are supported:
                Cauchy, Laplace, Logistic, Normal, and StudentT.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the distribution's
                batch shape. When `sample=True` (direct sampling), this is used as `sample_shape`
                to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of
                creating a sample site. This is essential for building complex distributions
                like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro TruncatedDistribution distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the TruncatedDistribution distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=1, low = 0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncateddistribution
        """

           
        d = numpyro.distributions.truncated.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def truncated_normal(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Truncated Normal distribution.
        
        The Truncated Normal distribution is a normal distribution truncated
        to a specified interval. It is defined by its location (`loc`), scale
        (`scale`), lower bound (`low`), and upper bound (`high`).
        
        .. math::
           f(x) = \frac{p(x)}{\alpha}
           \text{ where }
           p(x) \text{ is the PDF of the Normal distribution with location } \loc \text{ and scale } \scale,
           \alpha = \int_{low}^{high} p(x) dx
        
        Args:
            loc (float): The location parameter of the normal distribution.
            scale (float): The scale parameter of the normal distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
            event (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro TruncatedNormal distribution object (for model building).
            JAX array of samples drawn from the TruncatedNormal distribution (for
            direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_normal(loc=0.0, scale=1.0, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncatednormal_lowercase
        """

           
        d = numpyro.distributions.truncated.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def zero_inflated_distribution(base_dist, gate=None, gate_logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Generic Zero Inflated distribution.
        
        A Zero-Inflated distribution combines a base distribution with a Bernoulli distribution
        to model data with an excess of zero values. It assumes that each observation
        is either drawn from the base distribution or is a zero with probability determined
        by the Bernoulli distribution (the "gate"). This is useful for modeling data
        where zeros are more frequent than expected under a single distribution,
        often due to a different underlying process.
        
        .. math::
            P(x) = \pi \cdot I(x=0) + (1 - \pi) \cdot P_{base}(x)
        
        where:
        - $P_{base}(x)$ is the probability density function (PDF) or probability mass function (PMF) of the base distribution.
        - $\pi$ is the probability of generating a zero, governed by the Bernoulli gate.
        - $I(x=0)$ is an indicator function that equals 1 if x=0 and 0 otherwise.
        
        Args:
            base_dist (Distribution): The base distribution to be zero-inflated (e.g., Poisson,     NegativeBinomial).

            gate (jnp.ndarray, optional): Probability of extra zeros (between 0 and 1).

            gate_logits (jnp.ndarray, optional): Log-odds of extra zeros.

            validate_args (bool, optional): Whether to validate parameter values. Defaults to None.

            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro ZeroInflatedDistribution distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the ZeroInflatedDistribution distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_inflated_distribution(base_dist=m.dist.poisson(rate=5, create_obj = True), gate = 0.3, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution
        """

           
        d = numpyro.distributions.discrete.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def zero_inflated_negative_binomial2(mean, concentration, gate=None, gate_logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        r"""Samples from a Zero-Inflated Negative Binomial 2 distribution.
        
        This distribution combines a Negative Binomial 2 distribution with a binary gate variable.
        Observations are either drawn from the Negative Binomial 2 distribution with probability
        (1 - gate) or are treated as zero with probability 'gate'. This models data with excess zeros
        compared to what a standard Negative Binomial 2 distribution would predict.
        
        .. math::
            P(X = x) = (1 - gate) \cdot \frac{\Gamma(x + \alpha)}{\Gamma(x + \alpha + \beta) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^x + gate \cdot \delta_{x, 0}
        
        Args:
            mean (jnp.ndarray or float): The mean of the Negative Binomial 2 distribution.
            concentration (jnp.ndarray or float): The concentration parameter of the Negative Binomial 2 distribution.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            When `sample=False`: A NumPyro ZeroInflatedNegativeBinomial2 distribution object (for model building).
            When `sample=True`: A JAX array of samples drawn from the ZeroInflatedNegativeBinomial2 distribution (for direct sampling).
            When `create_obj=True`: The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_inflated_negative_binomial2(mean=2.0, concentration=3.0, gate = 0.3, sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflatednegativebinomial2
        """

           
        d = numpyro.distributions.conjugate.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

    @staticmethod
    def negative_binomial(total_count, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False):
        

        """Samples from a NegativeBinomial distribution.
        
        The NegativeBinomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials. It is characterized by two parameters: 'total_count' (r) and 'probs' or 'logits' (p).
        
        .. math::
            P(k) = \binom{k+r-1}{r-1} p^r (1-p)^k
        
        Args:
            total_count (jnp.ndarray): The total number of events.
        
            shape (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            mask (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        Returns:
            NumPyro NegativeBinomial distribution object (for model building).
            JAX array of samples drawn from the NegativeBinomial distribution (for direct sampling).
            The raw NumPyro distribution object (for advanced use cases).
        
        Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial(total_count=5.0,probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomial
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key,   sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if    mask is not None else None
                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)

