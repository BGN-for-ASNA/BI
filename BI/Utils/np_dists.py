import numpyro.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
import numpyro
import jax
import time
from BI.Utils.SampledData import SampledData as SampledData

class UnifiedDist:
    """A class to unify various distribution methods and provide a consistent interface for sampling and inference."""

    def __init__(self, rand_seed = True):
        self.seed = rand_seed


    def normalize_seed(self,seed):
        if seed is None:
            seed = self.seed
        # If seed is exactly a bool, return True or False
        if isinstance(seed, bool):
            if seed:
                return  int(time.time_ns())
            else:   
                return 0
        # If seed is an integer (but not a bool), return it
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
        # Otherwise, you may want to handle other types (None, floats, etc.)
        raise ValueError(f"Invalid seed type: {type(seed)}")

    @staticmethod
    def mask( mask):
        return numpyro.handlers.mask(mask=mask)

    @staticmethod
    def plate(name, shape):
        return numpyro.plate(name, shape)

    def asymmetric_laplace(self,loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Asymmetric Laplace
        
        Samples from an Asymmetric Laplace distribution.
        
        The Asymmetric Laplace distribution is a generalization of the Laplace distribution,
        where the two sides of the distribution are scaled differently. It is defined by
        a location parameter (loc), a scale parameter (scale), and an asymmetry parameter (asymmetry).
        
        $$
        f(x, \kappa) &= \frac{1}{\kappa+\kappa^{-1}}\exp(-x\kappa),\quad x\ge0\\
                    &= \frac{1}{\kappa+\kappa^{-1}}\exp(x/\kappa),\quad x<0\\

        for $-\infty < x < \infty, \kappa > 0`$.

        laplace_asymmetric takes 'kappa' as a shape parameter for
        $\kappa$. For :$\kappa = 1$, it is identical to a Laplace distribution.
        $$
        
        #### Args:

        - *loc* (jnp.ndarray or float): Location parameter of the distribution.
            
        - *scale* (jnp.ndarray or float): Scale parameter of the distribution.
        
        - *asymmetry* (jnp.ndarray or float): Asymmetry parameter of the distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.
        
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        - validate_args (bool, optional): Whether to enable validation of distribution parameters. Defaults to `None`.

        #### Returns:
        
        - When `sample=False`: A BI AsymmetricLaplace distribution object (for model building).

        - When `sample=True`: A JAX array of samples drawn from the BetaBinomial distribution (for direct sampling).

        - When `create_obj=True`: The raw BI AsymmetricLaplace distribution object (for advanced use cases).

        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.asymmetric_laplace(loc=0.0, scale=1.0, asymmetry=1.0, sample=True)

        #### Wrapper of:

            https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplace

        References
        ----------
        .. [1] "Asymmetric Laplace distribution", Wikipedia
                https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution

        .. [2] Kozubowski TJ and Podgórski K. A Multivariate and
               Asymmetric Generalization of Laplace Distribution,
               Computational Statistics 15, 531--540 (2000).
               :doi:`10.1007/PL00022717`
        """

           
        d = numpyro.distributions.continuous.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))

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

    def asymmetric_laplace_quantile(self,loc=0.0, scale=1.0, quantile=0.5, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Asymmetric Laplace Quantile
        
        Samples from an AsymmetricLaplaceQuantile distribution.
        
        This distribution is an alternative parameterization of the AsymmetricLaplace
        distribution, commonly used in Bayesian quantile regression. It utilizes a
        `quantile` parameter to define the balance between the left- and right-hand
        sides of the distribution, representing the proportion of probability density
        that falls to the left-hand side.

        #### Args:

        - *loc* (float): The location parameter of the distribution.
        
        - *sample* (float): The scale parameter of the distribution.
        
            quantile (float): The quantile parameter, representing the proportion of
                probability density to the left of the median. Must be between 0 and 1.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        #### Returns:

        - When `sample=False`: A BI AsymmetricLaplaceQuantile distribution object (for model building).

        - When `sample=True`: A JAX array of samples drawn from the BetaBinomial distribution (for direct sampling).

        - When `create_obj=True`: The raw BI AsymmetricLaplaceQuantile distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=0.5, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplacequantile
        """

           
        d = numpyro.distributions.continuous.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def bernoulli_logits(self,logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Bernoulli Logits
        
        Samples from a Bernoulli distribution parameterized by logits.
        
        The Bernoulli distribution models a single binary event (success or failure),
        parameterized by the log-odds ratio of success.  The probability of success
        is given by the sigmoid function applied to the logit.
        
        $
        p = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}
        $
        where (\eta) is the log-odds (the *logit*).

        #### Args:
        - *logits* (jnp.ndarray, optional): Log-odds ratio of success.  Must be real-valued.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int, optional): The number of batch dimensions to reinterpret as
            - *event* dimensions (used in model building). Defaults to 0.
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask
                observations. Defaults to None.
        
        - *create_obj* (bool, optional): If True, returns the raw BI
                distribution object instead of creating a sample site. This is
                essential for building complex distributions like `MixtureSameFamily`.
                Defaults to False.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        #### Returns:

      - When `sample=False`: A BI BernoulliLogits distribution object (for  model building).
        
      - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli_logits(logits=jnp.array([0.2, 1, 2]), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#bernoulli-logits


        #### References
        [1]: https://en.wikipedia.org/wiki/Bernoulli_distribution?utm_source=chatgpt.com "Bernoulli distribution"
        [2]: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli?utm_source=chatgpt.com "tfp.        distributions.Bernoulli - Probability"
        [3]: https://mc-stan.org/docs/2_19/functions-reference/bernoulli-logit-distribution.html?utm_source=chatgpt.com "11.2 Bernoulli Distribution, Logit Parameterization - Stan"

        """

           
        d = numpyro.distributions.discrete.BernoulliLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def bernoulli_probs(self,probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Bernoulli Probs
        
        Samples from a Bernoulli distribution parameterized by probabilities.
        
        The Bernoulli distribution models the probability of success in a single trial,
        where the outcome is binary (success or failure). It is characterized by a single
        parameter, `probs`, representing the probability of success.
        
        $$
        Pr(X=1) = p = 1- Pr(X=0) = 1 - q, \\
        f(x)=p \text{ if } k = 1 \text{ else } q \text{ if } k = 0
        $$
                   
        where:
            p is the probability of success (0 <= p <= 1)

        
        #### Args:
        - *probs* (jnp.ndarray): The probability of success for each Bernoulli trial.
                Must be between 0 and 1.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
        
        - When `sample=False`: A BI BernoulliProbs distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI BernoulliProbs object (for advanced use cases).

        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli_probs(probs=jnp.array([0.2, 0.7, 0.5]), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#bernoulliprobs
        """

           
        d = numpyro.distributions.discrete.BernoulliProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def beta(self,concentration1, concentration0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Beta 
        Samples from a Beta distribution, defined on the interval [0, 1].
        The Beta distribution is a versatile distribution often used to model
        probabilities or proportions. It is parameterized by two positive shape
        parameters, usually denoted $\alpha$ and $\beta>0$, control the shape of the density (how much mass is pushed toward 0, 1, or intermediate).
        
        $$
        X \sim Beta(\alpha,\beta), \\f(x)=\frac{ x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \\
        B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}, \\F(x)=I_{x}(\alpha+\beta)
        $$

        where $B(\alpha, \beta)$ is the Beta function:

        $$
        B(\alpha, \beta) = \int_0^1 x^{\alpha - 1} (1 - x)^{\beta - 1} , dx = \frac{\Gamma(\alpha),\Gamma (\beta)}{\Gamma(\alpha + \beta)}.
        $$

        where :num$\alpha` and $\beta$ are the concentration parameters,
        and $B(x, y)$ is the Beta function.

        
        #### Args:

        - *concentration1* (jnp.ndarray): The first concentration parameter (shape parameter). Must be positive.

        - *concentration0* (jnp.ndarray): The second concentration parameter (shape parameter). Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI Beta distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI Beta object (for advanced use cases).

        
        #### #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.beta(concentration1=1.0, concentration0=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#beta

        #### References:
        [1]: https://en.wikipedia.org/wiki/Beta_distribution?utm_source=chatgpt.com "Beta distribution -        Wikipedia"
        [2]: https://www.statlect.com/probability-distributions/beta-distribution?utm_source=chatgpt.com        "Beta distribution | Properties, proofs, exercises - StatLect"
        [3]: https://reference.wolfram.com/language/ref/BetaDistribution.html?utm_source=chatgpt.com        "BetaDistribution - Wolfram Language Documentation"

        """

           
        d = numpyro.distributions.continuous.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def beta_binomial(self,concentration1, concentration0, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### BetaBinomial 
        Samples from a BetaBinomial distribution, a compound distribution where the probability of success in a binomial
        experiment is drawn from a Beta distribution. This models situations where the underlying probability of success
        is not fixed but varies according to a prior belief represented by the Beta distribution. It is often used to model over-dispersion relative to the binomial distribution.
        
        For (X \sim \mathrm{BetaBinomial}(N,\alpha,\beta)), for (k = 0,1,\dots,N):

        $$
        \Pr(X = k) = \binom{N}{k} ; \frac{B(k + \alpha,; N - k + \beta)}{B(\alpha,\beta)},
        $$

        where ($B(u,v)$) is the Beta function $(B(u,v) = \frac{\Gamma(u),\Gamma(v)}{\Gamma(u+v)})$. 

        #### Args:

        - *concentration1* (jnp.ndarray): The first concentration parameter (alpha) of the Beta distribution.
        
        - *concentration0* (jnp.ndarray): The second concentration parameter (beta) of the Beta distribution.

        - *total_count* (jnp.ndarray): The number of Bernoulli trials in the Binomial part of the distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

      - When `sample=False`: A BI BetaBinomial distribution object (for model building).

      - When `sample=True`: A JAX array of samples drawn from the BetaBinomial distribution (for direct sampling).

      - When `create_obj=True`: The raw BI BetaBinomial distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.beta_binomial(concentration1=1.0, concentration0=1.0, total_count=10, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#betabinomial

        #### References:
        [1]: https://en.wikipedia.org/wiki/Beta-binomial_distribution?utm_source=chatgpt.com "Beta-binomial distribution"
        [2]: https://www.statisticshowto.com/beta-binomial-distribution/?utm_source=chatgpt.com "Beta-Binomial Distribution: Definition - Statistics How To"
        [3]: https://mc-stan.org/docs/2_19/functions-reference/beta-binomial-distribution.html?utm_source=chatgpt.com "12.3 Beta-Binomial Distribution | Stan Functions Reference"
        """

           
        d = numpyro.distributions.conjugate.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def beta_proportion(self,mean, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Beta Proportion 
        
        The Beta Proportion distribution is a reparameterization of the conventional
        Beta distribution in terms of a the variate mean and a
        precision parameter. It's useful for modeling rates and proportions. It’s essentially the same family as the standard Beta((\alpha,\beta)), but the mapping is:

        $$
        \alpha = \mu , \kappa,\quad \beta = (1 - \mu), \kappa.
        $$
        
        Thus the pdf is:

        $$
        f(\theta \mid \mu, \kappa) = \frac{1}{B(\mu\kappa,,(1-\mu)\kappa)} ; \theta^{\mu\kappa - 1} ; (1 - \theta)^{(1 - \mu)\kappa         - 1}, \\
        \theta \in (0,1). 
        $$
        
        This parameterization is useful in hierarchical models, since one can put hyperpriors on (\mu) (the “center”) and (\kappa) (the “spread”).

        #### Args:

        - *mean* (jnp.ndarray): The mean of the BetaProportion distribution,must be between 0 and 1.
        
        - *concentration* (jnp.ndarray): The concentration parameter of the BetaProportion distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI BetaProportion distribution object (for model building).

        - When `sample=True`: A JAX array of samples drawn from the BetaBinomial distribution (for direct sampling).

        - When `create_obj=True`: The raw BI BetaProportion distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            samples = m.dist.beta_proportion(mean=0.5, concentration=2.0, sample=True, shape=(1000,))
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#beta_proportion

        #### References:
        https://mc-stan.org/docs/2_19/functions-reference/beta-proportion-distribution.html?utm_source=chatgpt.com "19.2 Beta Proportion Distribution | Stan Functions Reference"
        """

           
        d = numpyro.distributions.continuous.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def binomial_logits(self,logits, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Binomial Logits 
        The BinomialLogits distribution represents a binomial distribution parameterized by logits.
        It is useful when the probability of success is not directly known but is instead
        expressed as logits, which are the natural logarithm of the odds ratio.
        
        $$
        p = \mathrm{logit}^{-1}(\alpha) = \frac{1}{1 + e^{-\alpha}}.
        $$
        
        #### Args:
            logits (jnp.ndarray): Log-odds of each success.
            total_count (int): Number of trials.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI BinomialLogits distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI BinomialLogits object (for advanced use cases).

        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial_logits(logits=jnp.zeros(10), total_count=5, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#binomialllogits
        """

           
        d = numpyro.distributions.discrete.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def binomial_probs(self,probs, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Binomial Probs
        
        Samples from a Binomial distribution with specified probabilities for each trial.
        
        The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials, where each trial has the same probability of success.
        
        For $k = 0,1,2,\dots,N$:
        $$   
        \Pr(X = k) = \binom{N}{k} ; p^k ; (1 - p)^{,N - k}
        $$

        where $\binom{N}{k} = \frac{N!}{k!(N-k)!}$.
        
        #### Args:

        - *probs* (jnp.ndarray): The probability of success for each trial. Must be between 0 and 1.
        
        - *total_count* (int): The number of trials in each sequence.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
        
        - When `sample=False`: A BI BinomialProbs distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI BinomialLogits object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial_probs(probs=0.5, total_count=10, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#binomialprobs
        """

           
        d = numpyro.distributions.discrete.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def car(self,loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):

        r"""### Conditional Autoregressive (CAR) 
        The CAR distribution models a vector of variables where each variable is a linear
        combination of its neighbors in a graph. The CAR model captures spatial dependence in areal data by modeling each observation as conditionally dependent on its neighbors.It specifies a joint distribution of a vector of random variables $\mathbf{y} = (y_1, y_2, \dots, y_N)$ based on their conditional distributions, where each $y_i$ is conditionally independent of all other variables given its neighbors.

        * **Application**: Widely used in disease mapping, environmental modeling, and spatial econometrics to account for spatial autocorrelation.
        
        * **Conditional Distribution**: Each $y_i$ given its neighbors $\mathbf{y}_{-i}$ follows a normal distribution:

        $$
        y_i | \mathbf{y}*{-i} \sim \mathcal{N}\left( \frac{\sum*{j \in \mathcal{N}(i)} w_{ij} y_j}{\sum_{j \in \mathcal{N}(i)} w_{ij}}, \sigma^2 \right)
        $$

        where:

        * $\mathcal{N}(i)$ denotes the set of neighbors of unit $i$,
        * $w_{ij}$ are the weights representing the strength of the spatial relationship between units $i$ and $j$,
        * $\sigma^2$ is the variance term.
              where $\mu_i$ is a function of the values of the neighbors of site $i$
              and $\Sigma_i$ is the variance of site $i$.
        
        
        #### Args:
        - *loc* (Union[float, Array]): Mean of the distribution.

        - *correlation* (Union[float, Array]): Correlation between variables.

        - *conditional_precision* (Union[float, Array]): Precision of the distribution.

        - *adj_matrix* (Union[Array, scipy.sparse.spmatrix]): Adjacency matrix defining the graph.
            is_sparse (bool): Whether the adjacency matrix is sparse. Defaults to False.

        - *validate_args/ (bool): Whether to validate arguments. Defaults to None.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        #### Returns:
        
        - When `sample=False`: A BI CAR distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI CAR object (for advanced use cases).

        #### References:
        [1]: https://www.pymc.io/projects/examples/en/latest/spatial/conditional_autoregressive_priors.html?utm_source=chatgpt.com "Conditional Autoregressive (CAR) Models for Spatial Data"
        [2]: https://mc-stan.org/learn-stan/case-studies/icar_stan.html?utm_source=chatgpt.com "Intrinsic Auto-Regressive Models for Areal Data - Stan"
        [3]: https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118445112.stat08048?utm_source=chatgpt.com "Conditional Autoregressive (CAR) Model - Schmidt"

        """

        
           
        d = numpyro.distributions.continuous.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def categorical_logits(self,logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Categorical Logits 
        Samples from a Categorical distribution with logits. This distribution represents a discrete probability distribution over a finite set of outcomes, where the probabilities are determined by the logits. The probability of each outcome is given by the softmax function applied to the logits.
        
        Given logits $ \boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_K) $, the probability of each category $ i $ is:

        $$
        p_i = \frac{e^{\gamma_i}}{\sum_{j=1}^{K} e^{\gamma_j}}
        $$

        where $ p_i  is the probability of category $ i , and the sum in the denominator ensures that the probabilities sum to 1.
        
        #### Args:
        - *logits* (jnp.ndarray): Log-odds of each category.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI CategoricalLogits distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI CategoricalLogits object (for advanced use cases).
        
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical_logits(logits=jnp.zeros(5), sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#categoricallogits
        """

           
        d = numpyro.distributions.discrete.CategoricalLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def categorical_probs(self,probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Categorical Probs distribution.
        
        Samples from a Categorical distribution.
        
        The Categorical distribution is a discrete probability distribution that
        represents the probability of each outcome from a finite set of possibilities.
        It is often used to model the outcome of a random process with a fixed number
        of possible outcomes, such as the roll of a die or the selection of an item
        from a list.
        
        $$   
        P(x) = \frac{probs_i}{\sum_{k=1}^{K} probs_k}
        $$

        #### Args:
        - *probs* (jnp.ndarray): Probabilities for each category. Must sum to 1.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
        - When `sample=False`: A BI CategoricalProbs distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI CategoricalProbs object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical_probs(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#categoricalprobs
        """

           
        d = numpyro.distributions.discrete.CategoricalProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def cauchy(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Cauchy 
        Samples from a Cauchy
        
        The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
        that arises frequently in various fields, including physics and statistics. It is characterized by its
        heavy tails, which extend indefinitely. This means it has a higher probability of extreme values compared to the normal distribution.
        
        * **Parameters**:
            * **Location parameter** ($\mu$): Determines the peak of the distribution.
            * **Scale parameter** ($\gamma$): Describes the half-width at half-maximum (HWHM) of the peak.
        $$
        f(x; \mu, \gamma) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x - \mu}{\gamma}\right)^2\right]}
        $$

        where:

        * $x$ is the random variable,
        * $\mu$ is the location parameter,
        * $\gamma$ is the scale parameter.
        
        #### Args:
        - *loc* (jnp.ndarray or float, optional): Location parameter.  Defaults to 0.0.

        - *sample* (jnp.ndarray or float, optional): Scale parameter. Must be positive. Defaults to 1.0.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                Defaults to None.
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to False.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

      - When `sample=False`: A BI Cauchy distribution object (for model building).

      - When `sample=True`: A JAX array of samples drawn from the Cauchy distribution (for direct sampling).

      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.cauchy(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#cauchy
        """

           
        d = numpyro.distributions.continuous.Cauchy(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def chi2(self,df, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Chi-squared
        
        The Chi-squared distribution is a continuous probability distribution that arises
        frequently in hypothesis testing, particularly in ANOVA and chi-squared tests.
        It is defined by a single positive parameter, degrees of freedom (df), the number of independent standard normal variables squared and summed, which
        determines the shape of the distribution.
        
        $$   
        f(x; k) = \begin{cases}
        \frac{x^{\frac{k}{2} - 1} e^{-x/2}}{2^{\frac{k}{2}} \Gamma\left(\frac{k}{2}\right)}, & x > 0 \
        0, & \text{otherwise}
        \end{cases}
        $$

        where:
            * ( x ) is the random variable,
            * ( k ) is the degrees of freedom,
            * ( \Gamma(\cdot) ) is the gamma function.
        
        #### Args:
            df (jnp.ndarray): Degrees of freedom. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI Chi2 distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI Chi2 object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu)
            m.dist.chi2(df=3.0, sample = True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#chi2
        """

           
        d = numpyro.distributions.continuous.Chi2(df=df, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def circulant_normal(self,loc: jax.Array, covariance_row: jax.Array = None, covariance_rfft: jax.Array = None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Circulant Normal Multivariate normal 

        Circulant Normal Multivariate normal distribution with covariance matrix $\mathbf{C}` that is
        positive-definite and circulant [1], i.e., has periodic boundary conditions. The
        density of a sample $\mathbf{x}\in\mathbb{R}^n` is the standard multivariate
        normal density
        
        $$
            p\left(\mathbf{x}\mid\boldsymbol{\mu},\mathbf{C}\right) =
            \frac{\left(\mathrm{det}\,\mathbf{C}\right)^{-1/2}}{\left(2\pi\right)^{n / 2}}
            \exp\left(-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\intercal
            \mathbf{C}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}\right)\right),
        $$

        where $\mathrm{det}` denotes the determinant and $^\intercal` the
        transpose. Circulant matrices can be diagnolized efficiently using the discrete
        Fourier transform [1], allowing the log likelihood to be evaluated in
        $n \log n` time for $n` observations [2].
        
        - *loc:* Mean of the distribution $\boldsymbol{\mu}`.

        - *covariance_row:* First row of the circulant covariance matrix
            $\boldsymbol{C}`. Because of periodic boundary conditions, the covariance
            matrix is fully determined by its first row (see
            :func:`jax.scipy.linalg.toeplitz` for further details).
            
        - *covariance_rfft:* Real part of the real fast Fourier transform of
            :code:`covariance_row`, the first row of the circulant covariance matrix
            $\boldsymbol{C}`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        References:
        
        1. Wikipedia. (n.d.). Circulant matrix. Retrieved March 6, 2025, from
           https://en.wikipedia.org/wiki/Circulant_matrix

        2. Wood, A. T. A., & Chan, G. (1994). Simulation of Stationary Gaussian Processes in
           $\left[0, 1\right]^d`. *Journal of Computational and Graphical Statistics*,
           3(4), 409--432. https://doi.org/10.1080/10618600.1994.10474655
        
        #### Args:

        - *loc* : jnp.ndarray
                Mean of the distribution $\boldsymbol{\mu}`.

        - *covariance_row* : jnp.ndarray, optional. First row of the circulant covariance matrix $\mathbf{C}`.
                Defaults to None.

        - *covariance_rfft* : jnp.ndarray, optional Real part of the real fast Fourier transform of :code:`covariance_row`. Defaults to None.

        #### Returns:

        - When `sample=False`: A BI Circulant Normal Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI Circulant Normal Distribution object (for advanced use cases).
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#circulantnormal

        #### References:
        Wikipedia. (n.d.). Circulant matrix. Retrieved March 6, 2025, from https://en.wikipedia.org/wiki/Circulant_matrix

        Wood, A. T. A., & Chan, G. (1994). Simulation of Stationary Gaussian Processes in $[0,1]^d$. Journal of Computational and Graphical Statistics, 3(4), 409–432. https://doi.org/10.1080/10618600.1994.10474655
        """

           
        d = numpyro.distributions.continuous.CirculantNormal(loc=loc, covariance_row=covariance_row, covariance_rfft=covariance_rfft, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def delta(self,v=0.0, log_density=0.0, event_dim=0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Delta
        
        The Delta distribution, also known as a point mass distribution, assigns probability 1 to a single point and 0 elsewhere. It's useful for representing deterministic variables or as a building block for more complex distributions.
        
        $$   
        F(x) =
        \begin{cases}
        0, & x < a, \
        1, & x \ge a.
        \end{cases}
        $$
        
        #### Args:
            v (jnp.ndarray): The location of the point mass.
            log_density (float, optional): The log probability density of the point mass. This is primarily for creating distributions that are non-normalized or for specific advanced use cases. For a standard delta distribution, this should be 0. Defaults to 0.0.    
            event_dim (int, optional): The number of rightmost dimensions of `v` to interpret as event dimensions. Defaults to 0.
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
        
        - When `sample=False`: A BI Circulant Delta Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI Circulant Delta Distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.delta(v=0.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#delta
        """

           
        d = numpyro.distributions.distribution.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def dirichlet(self,concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Dirichlet
        
        Samples from a Dirichlet distribution.
        
        The Dirichlet distribution is a multivariate generalization of the Beta distribution.
        It is a probability distribution over a simplex, which is a set of vectors where each element is non-negative and sums to one. It is often used as a prior distribution for categorical distributions.

        For (\boldsymbol X = (X_1, \dots, X_K) \sim \mathrm{Dir}(\boldsymbol\alpha)), the density is
        
        
        $$
        f(\boldsymbol x; \boldsymbol\alpha) = \frac{1}{B(\boldsymbol\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}
        $$

        for (\boldsymbol x) in the simplex, and 0 otherwise. Here (B(\boldsymbol\alpha)) is the multivariate Beta function (the normalization constant):
        
        $$
        B(\boldsymbol\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^K \alpha_i\right)}.
        $$
        
        #### Args:
            concentration (jnp.ndarray): The concentration parameter(s) of the Dirichlet distribution.
                Must be a positive array.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
        
        #### Returns:

        - When `sample=False`: A BI Dirichlet Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI Dirichlet Distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.dirichlet(concentration=jnp.array([1.0, 1.0, 1.0]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#dirichlet

        #### References
        https://distribution-explorer.github.io/multivariate_continuous/dirichlet.html
        https://en.wikipedia.org/wiki/Dirichlet_distribution


        """

           
        d = numpyro.distributions.continuous.Dirichlet(concentration=concentration, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def dirichlet_multinomial(self,concentration, total_count=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0, create_obj=False):
        r"""### Dirichlet-Multinomial
    
        Creates a Dirichlet-Multinomial compound distribution, which is a Multinomial
        distribution with a Dirichlet prior on its probabilities. It is often used in
        Bayesian statistics to model count data where the proportions of categories are
        uncertain.

        The **Dirichlet-Multinomial** distribution describes a model in which you first draw a probability vector
        $$
        \mathbf{p} = (p_1, \dots, p_K) \sim \mathrm{Dirichlet}(\boldsymbol\alpha)
        $$
        and then you draw counts
        $$
          \mathbf{x} = (x_1, \dots, x_K) \sim \mathrm{Multinomial}(n, \mathbf{p}).
        $$
        
        Integrating out $\mathbf{p}$ gives you the marginal distribution of $\mathbf{x}). 
        * It generalizes the **Beta-Binomial** model to multiple categories (i.e. (K > 2)). 
        * A key feature is that it introduces **overdispersion** relative to a standard multinomial: counts are more “clumped” because the underlying probabilities vary according to the Dirichlet prior.


    
        #### Args:
            concentration (jnp.ndarray): The concentration parameter (alpha) for the Dirichlet prior. Values must be positive. The last dimension is interpreted as the number of categories.
            total_count (int, jnp.ndarray, optional): The total number of trials (n). This must be a non-negative integer. Defaults to 1.
            validate_args (bool, optional): Whether to enable validation of distribution parameters. Defaults to `None`.
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.

        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. If provided, events with a `True` mask will be conditioned on `obs`, while the remaining events will be treated as latent variables. Defaults to `None`.
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. This argument has no effect when `sample=False`. Defaults to 0.

        - *shape* (tuple, optional): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.

        - *event* (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).

        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to `False`.
        

        #### Returns:

        - When `sample=False`: A BI dirichlet_multinomial Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI dirichlet_multinomial Distribution object (for advanced use cases).

        #### Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')
        
            # Direct sampling
            # Sample a single vector of counts for 10 trials from 3 categories
            counts = m.dist.dirichlet_multinomial(concentration=jnp.array([1.0, 1.0, 1.0]),total_count=10,sample=True)
        
            # Usage within a model
            def my_model(obs_data=None):
                # Define a prior on the concentration parameter
                alpha = m.dist.half_cauchy(scale=jnp.ones(5), name='alpha', shape=(5,))

                # Model observed counts
                with m.plate('data', len(obs_data)):
                    y = m.dist.dirichlet_multinomial(
                        concentration=alpha,
                        total_count=100,
                        name='y',
                        obs=obs_data
                    )
    
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#dirichletmultinomial

        #### References:
        https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html

        """

        d = numpyro.distributions.conjugate.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def discrete_uniform(self,low=0, high=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Discrete Uniform 
        Samples from a Discrete Uniform distribution.
        
        The Discrete Uniform distribution defines a uniform distribution over a range of integers.
        It is characterized by a lower bound (`low`) and an upper bound (`high`), inclusive.
        
        $$   
        P(X = k) = \frac{1}{high - low + 1}, \quad k \in \{low, low+1, ..., high\}
        $$

        Otherwise (if $k$ is outside the range), $ P(X = k) = 0 $.

        
        #### Args:
            low (jnp.ndarray): The lower bound of the uniform range, inclusive.
            high (jnp.ndarray): The upper bound of the uniform range, inclusive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

        - When `sample=False`: A BI DiscreteUniform Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI dirichlet_multinomial Distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.discrete_uniform(low=0, high=5, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#discreteuniform

        #### References:

        """

           
        d = numpyro.distributions.discrete.DiscreteUniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def doubly_truncated_power_law(self,alpha, low, high, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        r"""### Doubly Truncated Power Law 

        This distribution represents a continuous power law with a finite support bounded
        between `low` and `high`, and with an exponent `alpha`. It is normalized over the
        interval `[low, high]` to ensure the area under the density function is 1.

        Key parameters:

          * (\alpha > 0): shape (exponent) of the power law
          * (x_{\min}): lower cutoff (minimum possible value)
          * (x_{\max}): upper cutoff (maximum possible value)

        The probability density function (PDF) is defined as:

        $$
        f(x) = \begin{cases}
        \displaystyle \frac{(1-\alpha), x^{-\alpha}}{x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha}}, & \alpha \neq 1, ; x_{\min} \le x         \le x_{\max},\
        \displaystyle \frac{1}{x, \ln\left(x_{\max}/x_{\min}\right)}, & \alpha = 1, ; x_{\min} \le x \le x_{\max},\
        0, & \text{otherwise}.
        \end{cases}
        $$

        where the normalization constant $Z(\alpha,a,b)$ is given by

        $$
        Z(\alpha,a,b) =
        \begin{cases}
        \log(b) - \log(a), & \text{if } \alpha = -1, \\
        \dfrac{b^{1+\alpha} - a^{1+\alpha}}{1+\alpha}, & \text{otherwise}.
        \end{cases}
        $$

        This distribution is useful for modeling data that follows a power-law behavior
        but is naturally bounded due to measurement or theoretical constraints (e.g.,
        finite-size systems).


        #### Args:
            alpha (float or array-like): Power-law exponent.

            low (float or array-like): Lower bound of the distribution (must be $≥$  0).

            high (float or array-like): Upper bound of the distribution (must be $>$ 0).

        - *shape* (tuple, optional): The shape of the output tensor. Defaults to None.

            validate_args (bool, optional): Whether to validate the arguments. Defaults to True.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        #### Returns:

        - When `sample=False`: A BI doubly_truncated_power_law Distribution distribution object (for  model building).

        - When `sample=True`: A JAX array of samples drawn from the BernoulliLogits distribution (for direct sampling).

        - When `create_obj=True`: The raw BI doubly_truncated_power_law Distribution object (for advanced use cases).

        #### Example Usage:
            from BI import bi
            m = bi('cpu')        
            m.dist.doubly_truncated_power_law(low=0.1, high=10.0, alpha=2.0, sample=True)

        #### References:
       https://reference.wolframcloud.com/language/ref/TruncatedDistribution.html
        """

        d = numpyro.distributions.truncated.DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def euler_maruyama(self,t, sde_fn, init_dist, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,   shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""###  Euler–Maruyama
         
        Euler–Maruyama methode is a method for the approximate numerical solution
        of a stochastic differential equation (SDE). It simulates the solution
        to an SDE by iteratively applying the Euler method to each time step,
        incorporating a random perturbation to account for the diffusion term.
        
        $$
           dX_t = f(X_t, t) dt + g(X_t, t) dW_t
        $$
        
        where:
        - $X_t` is the state of the system at time $t`.
        - $f(X_t, t)` is the drift coefficient.
        - $g(X_t, t)` is the diffusion coefficient.
        - $dW_t` is a Wiener process (Brownian motion).

        
        #### Args:
            t (jnp.ndarray): Discretized time steps.

            sde_fn (callable): A function that takes the current state and time as input and returns the drift and diffusion coefficients.
            init_dist (Distribution): The initial distribution of the system.

        - *shape* (tuple, optional): The shape of the output tensor. Defaults to None.

            sample_shape (tuple, optional): The shape of the samples to draw. Defaults to None.

            validate_args (bool, optional): Whether to validate the arguments. Defaults to True.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            jnp.ndarray: Samples drawn from the Euler–Maruyama distribution.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.euler_maruyama(t=jnp.array([0.0, 0.1, 0.2]), sde_fn=lambda x, t: (x, 1.0), init_dist=m.dist.normal(0.0, 1.0, create_obj=True), sample = True)
        """

           
        d = numpyro.distributions.continuous.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def exponential(self,rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Exponential
        
        The Exponential distribution is a continuous probability distribution that models the time until an event occurs in a Poisson process, where events occur continuously and independently at a constant average rate. It is often used to model the duration of events, such as the time until a machine fails or the length of a phone call.
        
        For ( x \ge 0 )
        $$   
        F(x; \lambda) = 1 - e^{-\lambda x}
        $$

        Zhere ( \lambda > 0 ) is the rate parameter (the mean is ( 1/\lambda )).
        
        #### Args:
            rate (jnp.ndarray): The rate parameter, $\lambda`. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Exponential distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Exponential distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.exponential(rate=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#exponential

        #### References
            https://reference.wolfram.com/language/ref/ExponentialDistribution.html
            https://en.wikipedia.org/wiki/Exponential_distribution
        """

           
        d = numpyro.distributions.continuous.Exponential(rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def folded_distribution(self,base_dist, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Folded

        Samples from a Folded distribution, which is the absolute value of a base univariate distribution.
        This distribution reflects the base distribution across the origin, effectively taking the absolute value of each sample.
    
        $$
           p(x) = \sum_{k=-\infty}^{\infty} p(x - 2k)
        $$

        #### Args:
        - *loc* (float, optional): Location parameter of the base distribution. Defaults to 0.0.
        - *sample* (float, optional): Scale parameter of the base distribution. Defaults to 1.0.
    
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
    
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
    
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
    
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI FoldedDistribution distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the FoldedDistribution distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
    
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.folded_distribution(m.dist.normal(loc=0.0, scale=1.0, create_obj = True), sample=True)
    
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#foldeddistribution
        """

           
        d = numpyro.distributions.distribution.FoldedDistribution(base_dist=base_dist, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gamma(self,concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gamma 
        Samples from a Gamma distribution.
        
        The Gamma distribution is a continuous probability distribution frequently used in Bayesian statistics, particularly as a prior distribution for variance parameters. It is defined by two positive shape parameters: concentration ((k)) and rate ((\lambda)).

        There are two common parameterizations:
        
        1. **Shape–scale**: parameters ((\alpha, \theta)), with density
           [
           f(x) = \frac{1}{\Gamma(\alpha) \theta^\alpha} x^{\alpha - 1} e^{-x/\theta}, \quad x \ge 0
           ]
        
        2. **Shape–rate**: parameters ((\alpha, \lambda)) where (\lambda = 1/\theta), with density
           [
           f(x) = \frac{\lambda^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\lambda x}, \quad x \ge 0
           ]
        
        #### Args:
            concentration (jnp.ndarray): The shape parameter of the Gamma distribution (k > 0).
        
            rate (jnp.ndarray): The rate parameter of the Gamma distribution (theta > 0).
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            Gamma: A BI Gamma distribution object (for model building).
        
            jnp.ndarray: A JAX array of samples drawn from the Gamma distribution (for direct sampling).
        
            Gamma: The raw BI distribution object (for advanced use cases).
        
        
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gamma(concentration=2.0, rate=0.5, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#gamma

        #### References
        https://reference.wolfram.com/language/ref/GammaDistribution.html
        https://en.wikipedia.org/wiki/Gamma_distribution

        """

           
        d = numpyro.distributions.continuous.Gamma(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gamma_poisson(self,concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gamma Poisson 
        The Gamma-Poisson distribution, also known as the Negative Binomial distribution, models overdispersed count data. It arises from a hierarchical process where the rate parameter of a Poisson distribution is itself a random variable following a Gamma distribution. This structure allows the model to capture variability in count data that exceeds what is predicted by the Poisson distribution, making it suitable for applications like modeling RNA-sequencing data and microbial count [1].
        
        $$
        f(x) = \frac{\Gamma(x + \beta) \alpha^x}{\Gamma(\beta)(1 + \alpha)^{\beta + x} x!}, \quad x = 0, 1, 2, \ldots
        $$
        Where $\Gamma$ denotes the gamma function.

        This CDF expresses the probability that the count variable ( X ) takes a value less than or equal to ( x ), incorporating the overdispersion introduced by the Gamma distribution.
        
        #### Args:
            concentration (jnp.ndarray): Shape parameter (alpha) of the Gamma distribution.
            rate (jnp.ndarray): Rate parameter (beta) for the Gamma distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI GammaPoisson distribution object (for model
                building).
        
      - When `sample=True`: A JAX array of samples drawn from the GammaPoisson
                distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced
                use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gamma_poisson(concentration=1.0, rate=2.0, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#gammapoisson

        #### References
        [1] https://www.statisticshowto.com/gamma-poisson-distribution
        https://en.wikipedia.org/wiki/Negative_binomial_distribution
        """

           
        d = numpyro.distributions.conjugate.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gaussian_copula(self,marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""###  Gaussian Copula

        A distribution that links the `batch_shape[:-1]` of a marginal distribution with a multivariate Gaussian copula,
        modelling the correlation between the axes. A copula is a multivariate distribution over the uniform distribution
        on [0, 1]. The Gaussian copula links the marginal distributions through a multivariate normal distribution.
        

        
        #### Args:
            marginal_dist (Distribution): Distribution whose last batch axis is to be coupled.

            correlation_matrix (array_like, optional): Correlation matrix of the coupling multivariate normal distribution. Defaults to None.

            correlation_cholesky (array_like, optional): Correlation Cholesky factor of the coupling multivariate normal distribution. Defaults to None.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.

        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).

        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.

        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `Mi  xtureSameFamily`. Defaults to False.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI GaussianCopula distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            BI distribution object: When `create_obj=True` (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_copula(
                marginal_dist = m.dist.beta(2.0, 5.0, create_obj = True), 
                correlation_matrix = jnp.array([[1.0, 0.7],[0.7, 1.0]]), 
                sample = True
            )
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussiancopula
        """

           
        d = numpyro.distributions.copula.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gaussian_copula_beta(self,concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gaussian Copula Beta 
    
        This distribution combines a Gaussian copula with a Beta distribution.
        The Gaussian copula models the dependence structure between random variables,
        while the Beta distribution defines the marginal distributions of each variable.

    
        #### Args:
            concentration1 (jnp.ndarray): The first shape parameter of the Beta distribution.

            concentration0 (jnp.ndarray): The second shape parameter of the Beta distribution.

            correlation_matrix (array_like, optional): Correlation matrix of the coupling multivariate normal distribution. Defaults to None.
            
            correlation_cholesky (jnp.ndarray): The Cholesky decomposition of the correlation matrix.
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
                of the given shape.
    
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
    
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
    
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            GaussianCopulaBeta: A BI GaussianCopulaBeta distribution object (for model building).
            jnp.ndarray: A JAX array of samples drawn from the GaussianCopulaBeta distribution (for direct sampling).
            Distribution: The raw BI distribution object (for advanced use cases).
    
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_copula_beta(
                concentration1 = jnp.array([2.0, 3.0]), 
                concentration0 = jnp.array([5.0, 3.0]),
                correlation_cholesky = jnp.linalg.cholesky(jnp.array([[1.0, 0.7],[0.7, 1.0]])), 
                sample = True
            )
    
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussiancopulabetadistribution
        """

           
        d = numpyro.distributions.copula.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gaussian_random_walk(self,scale=1.0, num_steps=1, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0, create_obj=False):
        r"""### Gaussian Random Walk 
    
        Creates a distribution over a Gaussian random walk of a specified number of steps.
        This is a discrete-time stochastic process where the value at each step is the
        previous value plus a Gaussian-distributed increment. The distribution is over
        the entire path.

    
        #### Args:
        - *sample* (float, jnp.ndarray, optional): The standard deviation ($\sigma`) of the Gaussian increments. Must be positive. Defaults to 1.0.
            num_steps (int, optional): The number of steps in the random walk, which determines the event shape of the distribution. Must be positive. Defaults to 1.
            validate_args (bool, optional): Whether to enable validation of distribution parameters. Defaults to `None`.
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. If provided, events with a `True` mask will be conditioned on `obs`, while the remaining events will be treated as latent variables. Defaults to `None`.
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. This argument has no effect when `sample=False`. Defaults to 0.
        - *shape* (tuple, optional): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        - *event* (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to `False`.
        
        #### Returns:
            BI.primitives.Messenger: A BI sample site object when used in a model context (`sample=False`).
            jnp.ndarray: A JAX array of samples drawn from the GaussianRandomWalk distribution (for direct sampling, `sample=True`).
            numpyro.distributions.Distribution: The raw BI distribution object (if `create_obj=True`).
    
        #### Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')
            
            # Direct sampling of a random walk with 100 steps
            path = m.dist.gaussian_random_walk(scale=0.5, num_steps=100, sample=True)
            
            # Usage within a model for a latent time series
            def my_model(data=None):
                # Prior on the volatility of the random walk
                volatility = m.dist.half_cauchy(scale=1.0, name='volatility')
                
                # The latent random walk
                latent_process = m.dist.gaussian_random_walk(
                    scale=volatility,
                    num_steps=len(data) if data is not None else 10,
                    name='latent_process'
                )
                
                # Observation model
                # Assumes the observed data is the latent process plus some noise
                obs_noise = m.dist.half_cauchy(scale=1.0, name='obs_noise')
                with m.plate('time', len(data) if data is not None else 10):
                    return m.dist.normal(loc=latent_process, scale=obs_noise, obs=data, name='obs')

        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussianrandomwalk
        """
        d = numpyro.distributions.continuous.GaussianRandomWalk(scale=scale,    num_steps=num_steps, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gaussian_state_space(self,num_steps, transition_matrix, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gaussian State Space 
        Samples from a Gaussian state space model.


        
        #### Args:
            num_steps (int): Number of steps.

            transition_matrix (jnp.ndarray): State space transition matrix $\mathbf{A}`.

            covariance_matrix (jnp.ndarray, optional): Covariance of the innovation noise
                $\boldsymbol{\epsilon}`. Defaults to None.

            precision_matrix (jnp.ndarray, optional): Precision matrix of the innovation noise
                $\boldsymbol{\epsilon}`. Defaults to None.

            scale_tril (jnp.ndarray, optional): Scale matrix of the innovation noise
                $\boldsymbol{\epsilon}`. Defaults to None.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.

        - *event* (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).

        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.

        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. Defaults to False.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI GaussianStateSpace distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the GaussianStateSpace distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gaussian_state_space(num_steps=5, transition_matrix=jnp.array([[0.5]]), covariance_matrix =  jnp.array([[1.0, 0.6],[0.6, 1.0]]), sample=True)

        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gaussianstate
        """

           
        d = numpyro.distributions.continuous.GaussianStateSpace(num_steps=num_steps, transition_matrix=transition_matrix, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def geometric_logits(self,logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### GeometricLogits 
        Samples from a GeometricLogits distribution, which models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is parameterized by logits, which are transformed into probabilities using the sigmoid function.
        

        $$
        P(X = k) = (1 - p)^k , p,
        $$
            

        where:
        
        *   X is the number of failures before the first success.
        *   k is the number of failures.
        *   p is the probability of success on each trial (derived from the logits).

        #### Args:
            logits (jnp.ndarray): Log-odds parameterization of the probability of success.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI GeometricLogits distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the GeometricLogits distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric_logits(logits=jnp.zeros(10), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#geometriclogits

        #### References:
             https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometric
        """

           
        d = numpyro.distributions.discrete.GeometricLogits(logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def geometric_probs(self,probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### GeometricProbs
        
        Samples from a Geometric
        
        The Geometric distribution models the number of trials until the first success in a sequence of independent Bernoulli trials, where each trial has the same probability of success.
        
        $$
           P(X = k) = (1 - p)^k p
        $$
        
        #### Args:
        - *probs* (jnp.ndarray): Probability of success on each trial. Must be between 0 and 1.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI GeometricProbs distribution object (for model building).
            JAX array of samples drawn from the GeometricProbs distribution (for direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric_probs(probs=0.5, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#geometricprobs
        """

           
        d = numpyro.distributions.discrete.GeometricProbs(probs=probs, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gompertz(self,concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gompertz
        
        The Gompertz distribution is a distribution with support on the positive real line that is closely
        related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
        entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.
        
        The probability density function (PDF) is:


        $$
        F(x) = 1 - \exp!\bigl[-a (e^x - 1)\bigr], \quad x \ge 0.
        $$


        
        #### Args:
            concentration (jnp.ndarray): The concentration parameter. Must be positive.
            rate (jnp.ndarray): The rate parameter. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Gompertz distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            BI distribution object: When `create_obj=True` (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gompertz(concentration=1.0, rate=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gompertz
        #### References:
           https://en.wikipedia.org/wiki/Gompertz_distribution
        """

           
        d = numpyro.distributions.continuous.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def gumbel(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Gumbel
        
        Samples from a Gumbel (or Extreme Value) distribution.
        
        The Gumbel distribution is a continuous probability distribution named after German mathematician Carl Gumbel.
        It is often used to model the distribution of maximum values in a sequence of independent random variables.


        $$
        F(x;\mu,\beta) = \exp\bigl( -\exp\bigl( -\tfrac{x - \mu}{\beta} \bigr) \bigr), \quad x \in \mathbb{R}.
        $$


        #### Args:
        - *loc* (jnp.ndarray or float, optional): Location parameter. Defaults to 0.0.
        - *sample* (jnp.ndarray or float, optional): Scale parameter. Must be positive. Defaults to 1.0.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                Defaults to 1.
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. Defaults to None.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to False.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Gumbel distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Gumbel distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.gumbel(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#gumbel
        #### References:
            https://en.wikipedia.org/wiki/Gumbel_distribution
        """

           
        d = numpyro.distributions.continuous.Gumbel(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def half_cauchy(self,scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### HalfCauchy 

        The HalfCauchy distribution is a probability distribution that is half of the Cauchy distribution. It is defined on the positive real numbers and is often used in situations where only positive values are relevant.
        
        $$   
        f(x) = \frac{1}{2} \cdot \frac{1}{\pi \cdot \frac{1}{scale} \cdot (x^2 + \frac{1}{scale^2})}
        $$
        
        #### Args:

        - *sample* (jnp.ndarray): The scale parameter of the Cauchy distribution. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI HalfCauchy distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the HalfCauchy distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.half_cauchy(scale=1.0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#halfcauchy
        """

           
        d = numpyro.distributions.continuous.HalfCauchy(scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def half_normal(self,scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### HalfNormal
        
        Samples from a HalfNormal distribution.
        
        The HalfNormal distribution is a distribution of the absolute value of a normal random variable.
        It is defined by a location parameter (implicitly 0) and a scale parameter.

        $$
        f(x; \sigma) = \frac{2}{\pi,\sigma} ; \frac{1}{1 + \bigl(x/\sigma\bigr)^2}, \qquad x \ge 0.
        $$

        #### Args:
        - *sample* (float, array): The scale parameter of the distribution. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI HalfNormal distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the HalfNormal distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.half_normal(scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#halfnormal

        #### References:
            https://distribution-explorer.github.io/continuous/halfcauchy.html?utm_source=chatgpt.com
        """

           
        d = numpyro.distributions.continuous.HalfNormal(scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def improper_uniform(self,support, batch_shape, event_shape, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Improper Uniform 
        A helper distribution with zero :meth:`log_prob` over the `support` domain.

        #### Args:
            support (numpyro.distributions.constraints.Constraint): The support of this distribution.

            batch_shape (tuple): Batch shape of this distribution. It is usually safe to
                set `batch_shape=()`.

            event_shape (tuple): Event shape of this distribution.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).

        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.

        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI ImproperUniform distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the ImproperUniform distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import sample
            from numpyro.distributions import ImproperUniform, Normal, constraints
        
            def model():
                x = sample('x', ImproperUniform(constraints.ordered_vector, (), event_shape=(10,)))
        
            #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#improperuniform
        """

           
        d = numpyro.distributions.distribution.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def inverse_gamma(self,concentration, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Inverse Gamma 
        The InverseGamma distribution is a two-parameter family of continuous probability
        distributions. It is defined by its shape $\alpha$ and rate $\beta$ parameters. It is often used as a prior distribution for precision parameters (inverse variance) in Bayesian statistics.
        
        $$
        f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-(\alpha + 1)} \exp\left(-\frac{\beta}{x}\right), \quad x > 0
        $$

        where:
        * (\Gamma(\alpha)) is the Gamma function.
        
        #### Args:
            concentration (jnp.ndarray): The shape parameter (\\alpha) of the InverseGamma distribution. Must be positive.
            rate (jnp.ndarray): The rate parameter (\\beta) of the InverseGamma distribution. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI InverseGamma distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the InverseGamma distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.inverse_gamma(concentration=2.0, rate=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#inversegamma
        """

           
        d = numpyro.distributions.continuous.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def kumaraswamy(self,concentration1, concentration0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Kumaraswamy
        
        The Kumaraswamy distribution is a continuous probability distribution defined on the interval [0, 1].
        It is a flexible distribution that can take on various shapes depending on its parameters.
        
        $$
        $$
        f(x; a, b) = a b x^{a-1} (1 - x^a)^{b-1}, \quad x \in (0, 1)
        $$

        where:
        * ( a ) and ( b ) are shape parameters.
        * ( x ) is the random variable.
        
        #### Args:
            concentration1 (jnp.ndarray): The first shape parameter. Must be positive.
            concentration0 (jnp.ndarray): The second shape parameter. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Kumaraswamy distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Kumaraswamy distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.kumaraswamy(concentration1=2.0, concentration0=3.0, sample=True)
        
        #### Wrapper of: 
            https://num.pyro.ai/en/stable/distributions.html#kumaraswamy
        #### Rerferences:
            https://en.wikipedia.org/wiki/Kumaraswamy_distribution

        """

           
        d = numpyro.distributions.continuous.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def lkj(self,dimension, concentration=1.0, sample_method='onion', validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Lewandowski Kurowicka Joe (LKJ)  
        
        The LKJ distribution is controlled by the concentration parameter $\eta$ to make the probability
        of the correlation matrix $M` proportional to $\det(M)^{\eta - 1}$. When $\eta = 1$,
        the distribution is uniform over correlation matrices.  When $\eta > 1$, the distribution favors
        samples with large determinants. When $\eta < 1$, the distribution favors samples with small
        determinants.
        
        $$
           P(M) \propto |\det(M)|^{\eta - 1}
        $$
        
        #### Args:
            dimension (int): The dimension of the correlation matrices.
        
            concentration (ndarray): The concentration/shape parameter of the distribution (often referred to as eta). Must be positive.

            sample_method (str): Either “cvine” or “onion”. Both methods are proposed in [1] and offer the same distribution over correlation matrices. But they are different in how to generate samples. Defaults to “onion”.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI LKJ distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the LKJ distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.lkj(dimension=2, concentration=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#lkj
        #### References:
            https://en.wikipedia.org/wiki/Kumaraswamy_distribution
        """

           
        d = numpyro.distributions.continuous.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def lkj_cholesky(self,dimension, concentration=1.0, sample_method='onion', validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### LKJ Cholesky 

        The LKJ (Leonard-Kjærgaard-Jørgensen) Cholesky distribution is a family of distributions
        on symmetric matrices, often used as a prior for the Cholesky decomposition of a
        symmetric matrix. It is particularly useful in Bayesian inference for models with
        covariance structure. Given a lower triangular matrix (L) with unit diagonal entries, the PDF is:
        $$
        p(L | \eta) \propto \prod_{k=2}^d L_{kk}^{d-k+2\eta-2}
        $$

        where:
            * (L_{kk}) is the diagonal element of (L).
            * (\eta) is the shape parameter.
            * (d) is the dimension of the matrix.

        * (\eta = 1): Uniform prior over correlation matrices.
        * (\eta > 1): Prior favors correlation matrices close to the identity matrix (i.e., weak correlations).
        * (\eta < 1): Prior allows stronger correlations.

        #### Args:
            dimension (int): The dimension of the correlation matrices.

            concentration (float): A parameter controlling the concentration of the distribution
                around the identity matrix. Higher values indicate greater concentration.
                Must be greater than 1.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).

        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            Attributes:
                concentration (float): The concentration parameter.
            
        #### References:
            https://mc-stan.org/docs/2_21/functions-reference/cholesky-lkj-correlation-distribution.html
        """
        
        d = numpyro.distributions.continuous.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def laplace(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Laplace 
        Samples from a Laplace distribution, also known as the double exponential distribution.
        The Laplace distribution is defined by its location parameter (loc) and scale parameter (scale).
        
        $$
        f(x \mid \mu, b) = \frac{1}{2b} \exp\left( -\frac{|x - \mu|}{b} \right)
        $$

        Where:
        * ($\mu$) is the **location parameter**, indicating the peak of the distribution.
        * (b > 0) is the **scale parameter**, controlling the spread of the distribution.
        
        #### Args:
        - *loc* (jnp.ndarray): Location parameter of the Laplace distribution.
        
        - *sample* (jnp.ndarray): Scale parameter of the Laplace distribution. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a
                sample site. This is essential for building complex distributions like `MixtureSameFamily`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Laplace distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Laplace distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.laplace(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#laplace
        """

           
        d = numpyro.distributions.continuous.Laplace(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def left_truncated_distribution(self,base_dist, low=0.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Left Truncated 

        Samples from a left-truncated distribution.
        
        A left-truncated distribution is a probability distribution
        obtained by restricting the support of another distribution
        to values greater than a specified lower bound. This is useful
        when dealing with data that is known to be greater than a certain value. All the “mass” below (or equal to) (a) is **excluded** (not just unobserved, but removed from the sample/analysis).
        
        Let (X) be a random variable with PDF (f_X(x)) and CDF (F_X(x)). Choose a truncation point (a). Then define a new random variable (Y = X \mid (X > a)). The PDF and CDF of (Y) are:

        * **PDF** for ( y > a ):
        $$
        f_Y(y) = \frac{f_X(y)}{1 - F_X(a)} .
        $$
        (Here (1 - F_X(a) = P(X > a)) is the normalising constant) ([Wikipédia][1])

        * **CDF** for (y > a):
        $$
        F_Y(y) = P(Y \le y) = \frac{F_X(y) - F_X(a)}{1 - F_X(a)} .
        $$

        #### Args:
            base_dist: The base distribution to truncate. Must be univariate and have real support.
            low: The lower truncation bound. Values less than this are excluded from the distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI LeftTruncatedDistribution distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the LeftTruncatedDistribution distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        

        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#lefttruncateddistribution

        #### References:
        https://en.wikipedia.org/wiki/Truncated_distribution
        https://encyclopediaofmath.org/wiki/Truncated_distribution

        """

           
        d = numpyro.distributions.truncated.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def levy(self,loc, scale, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Levy 
        
        Samples from a Levy distribution.
        The Lévy distribution is a continuous probability distribution on the positive real line (or shifted positive line) that is *heavy-tailed*, *skewed*, and arises naturally in connection with stable distributions (specifically the case with stability index $\alpha = \tfrac12$. 
        It is often used in contexts such as hitting‐time problems for Brownian motion, physics (e.g., van der Waals line‐shapes), and modelling very heavy‐tailed phenomena. Let (X) be a Lévy‐distributed random variable with **location** parameter $\mu$ and **scale** parameter (c > 0). The support is $x \ge \mu$.
        
        **PDF**
        $$
        f(x; \mu, c) = \sqrt{\frac{c}{2\pi}} ; \frac{1}{(x - \mu)^{3/2}} ; \exp!\Bigl( -\frac{c}{2 (x - \mu)} \Bigr),
        \quad x \ge \mu.
        $$

        **CDF**
        $$
        F(x; \mu, c) = \Pr(X \le x)
        = \operatorname{erfc}!\Bigl(\sqrt{\tfrac{c}{2 (x - \mu)}}\Bigr)
        = 2 - 2,\Phi!\Bigl(\sqrt{\tfrac{c}{(x - \mu)}}\Bigr),
        \quad x \ge \mu.
        $$
        Here $\operatorname{erfc}$ is the complementary error function and $\Phi$ is the standard normal CDF.  

        In the *standard* case with $\mu = 0$ and $c = 1$, one often simply writes $X \sim \text{Lévy}(0,1)$.


        #### Args:
        - *loc* (jnp.ndarray): Location parameter.
        - *sample* (jnp.ndarray): Scale parameter.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.

        - *event* (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).

        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.

        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Levy distribution object: When `sample=False` (for model building).
            JAX array: When `sample=True` (for direct sampling).
            BI distribution object: When `create_obj=True` (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.levy(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#levy
        """

           
        d = numpyro.distributions.continuous.Levy(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def log_normal(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Log-Normal
        
        The Log-Normal distribution is a probability distribution defined for positive real-valued random variables, parameterized by a location parameter (loc : $\mu$) and a scale parameter (scale).  It arises when the logarithm of a random variable is normally distributed.
        * A random variable (X) is **log-normal** if its natural logarithm (\ln X) is (approximately) normally distributed.  Equivalently, one can write: 
        $$
        X = \exp(Y), \quad \text{where } Y \sim \mathcal{N}(\mu, \sigma^2).
        $$

        Because (X) is the exponential of a normal, it is always positive (support (x>0)).
        It is useful in contexts where growth, multiplicative effects, or compounding factors dominate (e.g. stock prices, income, sizes of biological organisms). 

        ## Parameters

        * $\mu$: the mean of $\ln X$ (i.e. the location parameter in log-space)
        * $\sigma > 0$: the standard deviation of $\ln X$ (i.e. the “scale” in log-space)

        Sometimes another parameterization uses a threshold (shift) $\theta$, i.e. $X = \theta + \exp(Y)$.
        For the two-parameter (no shift) case:

        * **PDF**
        $$
        f(x; \mu, \sigma) = \frac{1}{x,\sigma\sqrt{2\pi}} ; \exp!\Bigl( -\frac{(\ln x - \mu)^2}{2\sigma^2} \Bigr), \quad x > 0.
        $$

        * **CDF**
        $$
        F(x; \mu, \sigma) = \Pr(X \le x) = \Phi!\bigl( \frac{\ln x - \mu}{\sigma} \bigr),
        $$
        where $\Phi$ is the standard normal CDF. 

        ---
        
        #### Args:
        - *loc* (float): Location parameter.
        - *sample* (float): Scale parameter.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
                of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI LogNormal distribution object (for model building).
            JAX array of samples drawn from the LogNormal distribution (for direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.log_normal(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#lognormal

        #### References:    
        https://en.wikipedia.org/wiki/Log-normal_distribution

        """

           
        d = numpyro.distributions.continuous.LogNormal(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def log_uniform(self,low, high, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Log-Uniform
        
        Samples from a Log Uniform distribution.
        
        The Log Uniform distribution is defined over the positive real numbers and is the result of applying an exponential transformation to a uniform distribution over the interval [low, high]. It is often used when modeling parameters that must be positive.

        A random variable $X$ is **log-uniform** on $[a, b]$, with $0 < a < b$, if $\ln X$ is uniformly distributed on $[\ln a, \ln b]$. 
        Equivalently, the density of $X$ is proportional to $1/x$ over that interval. This distribution is sometimes called the *reciprocal distribution*.
        It is useful in modeling scales spanning several orders of magnitude, where you want every decade (or log-interval) to have equal weight.

        For $x \in [a, b]$:

        * **PDF**
        $$
        f(x) = \frac{1}{x , [\ln(b) - \ln(a)]}, \quad a \le x \le b.
        $$
        Outside $[a, b]$, $f(x) = 0$.

        * **CDF**
        $$
        F(x) = \Pr(X \le x) = \frac{\ln(x) - \ln(a)}{\ln(b) - \ln(a)}, \quad a \le x \le b.
        $$
        (For $x < a$, $F(x) = 0$; for $x > b$, $F(x) = 1$.

        #### Args:
            low (jnp.ndarray): The lower bound of the uniform distribution's log-space. Must be positive.
        
            high (jnp.ndarray): The upper bound of the uniform distribution's log-space. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI LogUniform distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the LogUniform distribution (for direct sampling) when `sample=True`.
        
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.log_uniform(low=0.1, high=10.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#loguniform

        #### References:
            https://en.wikipedia.org/wiki/Reciprocal_distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html
            https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_loguniform.html

        """

           
        d = numpyro.distributions.continuous.LogUniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def logistic(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Logistic 
        Samples from a Logistic distribution.
        
        The Logistic distribution is a continuous probability distribution defined by two parameters: location and scale. It is often used to model growth processes and is closely related to the normal distribution.Its CDF is the logistic (sigmoid) function, which makes it appealing in modeling probabilities, logistic regression, and various growth models. It resembles the normal distribution in shape (bell‐shaped, symmetric) but has **heavier tails** (i.e. more probability in the extremes) and simpler closed‐form expressions for the CDF.

        * **Location parameter**: $\mu \in \mathbb{R}$\ — the center or “mean” of the distribution.
        * **Scale parameter**: $\s > 0$\ — controls the spread (similar role to standard deviation).
        * **Support**: $\ x \in (-\infty, +\infty) $\.

        Let (X \sim \mathrm{Logistic}(\mu, s)). Then:
        
        $$
        F(x; \mu, s) = \frac{1}{1 + \exp\bigl(-\frac{x - \mu}{s}\bigr)}.
        $$

        Equivalently:
        $$
        F(x) = \frac12 + \frac12 \tanh!\bigl(\frac{x - \mu}{2s}\bigr).
        $$
        
        #### Args:
        - *loc* (jnp.ndarray or float): The location parameter, specifying the median of the distribution. Defaults to 0.0.
        
        - *sample* (jnp.ndarray or float): The scale parameter, which determines the spread of the distribution. Must be positive. Defaults to 1.0.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Logistic distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Logistic distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.logistic(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#logistic

        #### Reference: 
             https://en.wikipedia.org/wiki/Logistic_distribution
        """

           
        d = numpyro.distributions.continuous.Logistic(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def low_rank_multivariate_normal(self,loc, cov_factor, cov_diag, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Low Rank Multivariate Normal 
        The *Low-Rank Multivariate Normal* (LRMVN) distribution is a parameterization of the multivariate normal distribution where the covariance matrix is expressed as a low-rank plus diagonal decomposition:
        $$
        \Sigma = F F^\top + D
        $$
        where $F$ is a low-rank matrix (capturing correlations) and $D$ is a diagonal matrix (capturing independent noise). This representation is often used in probabilistic modeling and variational inference to efficiently handle high-dimensional Gaussian distributions with structured covariance.
        


        For a (d)-dimensional random vector ( x \in \mathbb{R}^d ),
        $$
        p(x \mid \mu, F, D) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
        \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right),
        $$
        where ( \Sigma = F F^\top + D ).

        
        Parameters:
        - *loc* (jnp.ndarray): Mean vector.

            cov_factor (jnp.ndarray): Matrix used to construct the covariance matrix.

            cov_diag (jnp.ndarray): Diagonal elements of the covariance matrix.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            event_size = 100  # Our distribution has 100 dimensions
            rank = 5   
            m.dist.low_rank_multivariate_normal(
            - *loc*=m.dist.normal(0,1, shape = (event_size,), sample=True)*2, 
                cov_factor=m.dist.normal(0,1, shape = (event_size, rank), sample=True),
                cov_diag=jnp.exp(m.dist.normal(0,1, shape = (event_size,), sample=True)) * 0.1, sample=True
            )
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#lowrankmultivariatenormal

        #### Reference:
        
        * [Multivariate normal distribution – Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

        * [TensorFlow Probability: LowRankMultivariateNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRank)
        """

        d = numpyro.distributions.continuous.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def lower_truncated_power_law(self,alpha, low, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Lower Truncated Power Law
        The *Lower-Truncated Power-Law* distribution (also known as the *Pareto Type I* or *power-law with a lower bound*) models quantities that follow a heavy-tailed power-law behavior but are bounded below by a minimum value ( x_{\min} ). It is commonly used to describe phenomena such as wealth distributions, city sizes, and biological scaling laws.
        
        Lower truncated power law distribution with $\alpha` index.
        
        The probability density function (PDF) is given by:
        
        For ( x \ge x_{\min} > 0 ):
        $$
        f(x \mid \alpha, x_{\min}) = \frac{\alpha - 1}{x_{\min}} \left( \frac{x_{\min}}{x} \right)^{\alpha}
        $$

        where $ \alpha > 1 $ is the shape (scaling) parameter.

        
        #### Args:
        - `alpha` (jnp.ndarray): index of the power law distribution. Must be less than -1.

        - `low` (jnp.ndarray): lower bound of the distribution. Must be greater than 0.
        
        - `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - `event` (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - `create_obj` (bool): If True, returns the raw BI distribution object instead of creating a sample site.This is essential for building complex distributions like `MixtureSameFamily`.
                
        - `sample` (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - `seed` (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - `obs` (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - `name` (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI LowerTruncatedPowerLaw distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the LowerTruncatedPowerLaw distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.lower_truncated_power_law(alpha=-2.0, low=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#lowertruncatedpowerlaw

        #### Reference:


        * [Power-law distribution – Wikipedia](https://en.wikipedia.org/wiki/Power_law#Continuous_power-law_distribution)

        * [Pareto distribution – Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution)

        * [Truncated Pareto distribution](https://arxiv.org/abs/0706.1062)
        """

           
        d = numpyro.distributions.truncated.LowerTruncatedPowerLaw(alpha=alpha, low=low, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def matrix_normal(self,loc, scale_tril_row, scale_tril_column, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Matrix Normal 
        Samples from a Matrix Normal distribution, which is a multivariate normal distribution over matrices.
        The distribution is characterized by a location matrix and two lower triangular matrices that define the correlation structure.
        The distribution is related to the multivariate normal distribution in the following way.
        If $X ~ MN(loc,U,V)$ then $vec(X) ~ MVN(vec(loc), kron(V,U) )$.
        
        Let (X) be (n \times p), with mean matrix $$M\in\mathbb R^{n\times p}$$, row-covariance matrix $U\in\mathbb R^{n\times n}$ (positive-definite), and column-covariance matrix $V\in\mathbb R^{p\times p}$(positive-definite). Then the PDF is:

        $$
        f(X;M,U,V) = \frac{1}{(2\pi)^{\tfrac{np}{2}},|V|^{\tfrac n2},|U|^{\tfrac p2}},
        \exp!\Big[-\tfrac12,\mathrm{tr}!\big( V^{-1}(X-M)^\top,U^{-1}(X-M)\big)\Big].
        $$

        Also the equivalence to vec-form is given:
        $$
        \mathrm{vec}(X) \sim \mathcal N_{np}\big(\mathrm{vec}(M),,V\otimes U\big). ; ; \text{([The Book of Statistical Proofs][3])}
        $$

        
        #### Args:
        - `loc` (array_like): Location of the distribution.
        
            scale_tril_row (array_like): Lower cholesky of rows correlation matrix.
        
            scale_tril_column (array_like): Lower cholesky of columns correlation matrix.
        
        - `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - `event` (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - `create_obj` (bool): If True, returns the raw BI distribution object instead of creating a sample site.This is essential for building complex distributions like `MixtureSameFamily`.
                
        - `sample` (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - `seed` (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - `obs` (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - `name` (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI MatrixNormal distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the MatrixNormal distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')            
            n_rows, n_cols = 3, 4

        - *loc* = jnp.zeros((n_rows, n_cols))
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
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase

        #### Reference: 
        * [1]: https://en.wikipedia.org/wiki/Matrix_normal_distribution "Matrix normal distribution"
        * [2]: https://statproofbook.github.io/P/matn-pdf.html "Probability density function of the matrix-normal distribution | The Book of Statistical Proofs"
        """

           
        d = numpyro.distributions.continuous.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def mixture_general(self,mixing_distribution, component_distributions, support=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Mixture General 
        A mixture distribution is a probability distribution constructed by selecting one of several component distributions according to specified weights, and then drawing a sample from the chosen component. It allows modelling of heterogeneous populations and multimodal data.
        
        Let the mixture have (K) component distributions with densities (f_i(x)) for (i = 1,\dots,K). Let weights       (w_i\ge0) satisfy (\sum_{i=1}^K w_i = 1). Then the PDF is
        $$
        f(x) = \sum_{i=1}^K w_i ; f_i(x).
        $$

        
        **Parameters:**
        
        *   **mixing_distribution**:  A `Categorical` distribution representing the mixing weights.

        *   **component_distributions**: A list of distributions representing the components of the mixture.
                
        *   **sample (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        * `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.

        *   **seed (int, optional)**:  An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        *   **obs (jnp.ndarray, optional)**:  The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        *   **name (str, optional)**:  The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        **#### Returns:**
        
        *   **When `sample=False`**: A BI MixtureGeneral distribution object (for model building).
        *   **When `sample=True`**: A JAX array of samples drawn from the MixtureGeneral distribution (for direct sampling).
        *   **When `create_obj=True`**: The raw BI distribution object (for advanced use cases).
        
        **#### Example Usage:**      

            from BI import bi
            m = bi('cpu')
            m.dist.mixture_general(
                mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True),     
                component_distributions=[m.dist.normal(loc=0.0, scale=1.0, create_obj=True),m.dist.normal  (loc=0.0, scale=1.0,  create_obj=True)],
                sample = True
            )           

        #### Wrapper of: [https://num.pyro.ai/en/stable/distributions.html#mixturegeneral](https://num.pyro.ai/en/stable/distributions.html#mixturegeneral)

        #### Reference: \
        * [1]: https://preliz.readthedocs.io/en/latest/distributions/gallery/mixture.html "Mixture Distribution — PreliZ 0.22.0 documentation"
        * [2]: https://reliability.readthedocs.io/en/latest/Mixture%20models.html "Mixture models — reliability 0.9.0 documentation"
        * [3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Mixture.html "Mixture — SciPy v1.16.2 Manual"
        """
        
           
        d = numpyro.distributions.mixtures.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def mixture_same_family(self,mixing_distribution, component_distribution, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""###  Finite mixture of component distributions from the same family.
        
        A *Mixture (Same-Family)* distribution is a finite mixture in which **all components come from the same parametric family** (for example, all Normal distributions but with different parameters), and are combined via mixing weights. The class is typically denoted as: 
        
        Let the mixture have (K) components. Let weights (w_i\ge0), (\sum_{i=1}^K w_i = 1). Let the component family have density (f(x \mid \theta_i)) for each component (i). Then the mixture’s PDF is
        $$
        f_X(x) = \sum_{i=1}^K w_i ; f(x \mid \theta_i).
        $$

        where each $f(x \mid \theta_i)$ is from the same family with parameter $\theta_i$. 
        
        #### Args:
            *   **Distribution Args**:
                *   **mixing_distribution**:  A `Categorical` distribution representing the mixing weights.

                *   **component_distributions**: A list of distributions representing the components of the mixture.
        
            *   **Sampling / Modeling Args**:
                * `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.

                *   `event` (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
                *   `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.

                *   `create_obj` (bool): If True, returns the raw BI distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.

                *   **sample (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
                *   **seed (int, optional)**:  An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

                *   **obs (jnp.ndarray, optional)**:  The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
                *   **name (str, optional)**:  The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            
        #### Returns:
            *   **When `sample=False`**: A BI MixtureSameFamily distribution object (for model building).
            *   **When `sample=True`**: A JAX array of samples drawn from the MixtureSameFamily distribution (for direct sampling).
            *   **When `create_obj=True`**: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.mixture_same_family(
                mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True), 
                component_distribution=m.dist.normal(loc=0.0, scale=1.0, shape = (2,), create_obj=True),
                sample = True
            )
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#mixture-, to_jax = True)same-family

        #### Reference: 
        * [Mixture distributions ](https://cseweb.ucsd.edu/~elkan/250Bwinter2011/mixturemodels.pdf)
        """

           
        d = numpyro.distributions.mixtures.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    
    def multinomial_logits(self,logits, total_count=1, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Multinomial Logits 
        A *multinomial logits* distribution refers to a categorical (or more generally multinomial) distribution over (K) classes whose probabilities are given via the softmax of a vector of logits. That is, given a vector of real-valued logits $\ell = (\ell_1, …, \ell_K)$, the class probabilities are:
        $$
        p_k = \frac{\exp(\ell_k)}{\sum_{j=1}^K \exp(\ell_j)}.
        $$
        Then a single draw from the distribution yields one of the (K) classes (or for a multinomial count version, counts over the classes) with those probabilities.

        **Categorical version (one draw):**
        Let (X \in {1,2,\dots,K}) be the class. Then
        $$
        \Pr(X=k \mid \ell) ;=; p_k = \frac{\exp(\ell_k)}{\sum_{j=1}^K \exp(\ell_j)}.
        $$
        This directly uses the logits $\ell$. 

        **Multinomial version (n draws):**
        If you draw (n) independent draws (or equivalently count vector) with probabilities (p = (p_1,…,p_K)), then for a count vector (x = (x_1,…,x_K)) with (\sum_k x_k = n),
        $$
        \Pr(X = x \mid n, p) = \frac{n!}{x_1! \cdots x_K!} ; \prod_{k=1}^K p_k^{x_k}
        $$
        and $p_k$ is given by the softmax of logits. 
        #### Args:
            logits (jnp.ndarray): Logits for each outcome. Must be at least one-dimensional.
        
            total_count (jnp.ndarray): The total number of trials.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI MultinomialLogits distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the MultinomialLogits distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial_logits(logits=jnp.array([1.0, 0.5], dtype=jnp.float32), total_count=jnp.array(5, dtype=jnp.int32), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#multinomiallogits
        
        #### Reference:
        *   https://en.wikipedia.org/wiki/Multinomial_distribution
        """

           
        d = numpyro.distributions.discrete.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def multinomial_probs(self,probs, total_count=1, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Multinomial Probs 
        Samples from a Multinomial distribution.
        
        The Multinomial distribution models the number of times each of several discrete outcomes occurs in a fixed number of trials.  Each trial independently results in one of several outcomes, and each outcome has a probability of occurring.

        
        #### Args:
        - *probs* (jnp.ndarray): Vector of probabilities for each outcome. Must sum to 1.
            total_count (jnp.ndarray): The number of trials.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI MultinomialProbs distribution object (for model building).
            JAX array of samples drawn from the MultinomialProbs distribution (for direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial_probs(probs=jnp.array([0.2, 0.3, 0.5]), total_count=10, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#multinomialprobs
        """

           
        d = numpyro.distributions.discrete.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def multivariate_normal(self,loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Multivariate Normal
        
        The Multivariate Normal distribution, also known as the Gaussian distribution in multiple dimensions,
        is a probability distribution that arises frequently in statistics and machine learning. It is
        defined by its mean vector and covariance matrix, which describe the central tendency and
        spread of the distribution, respectively.
        
        $$
           p(x) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)
        $$
            

        where:
        - $x$ is a $n$-dimensional vector of random variables.
        - $\mu$ is the mean vector.
        - $\Sigma$ is the covariance matrix.

        
        #### Args:
        - *loc* (tuple): The mean vector of the distribution.
        
            covariance_matrix (jnp.ndarray, optional): The covariance matrix of the distribution. Must be positive definite.
        
            precision_matrix (jnp.ndarray, optional): The precision matrix (inverse of the covariance matrix) of the distribution. Must be positive definite.
        
            scale_tril (jnp.ndarray, optional): The lower triangular Cholesky decomposition of the covariance matrix.
        
        - `shape` (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI MultivariateNormal distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the MultivariateNormal distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:

        from BI import bi
        m = bi('cpu')
        m.dist.multivariate_normal(
        - *loc*=jnp.array([1.0, 0.0, -2.0]), 
            covariance_matrix=jnp.array([[ 2.0,  0.7, -0.3],
                                        [ 0.7,  1.0,  0.5],
                                        [-0.3,  0.5,  1.5]]), 
            sample=True
        )

        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#multivariate-normal

        #### Reference:

        * [Multivariate normal distribution – Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
        """

           
        d = numpyro.distributions.continuous.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def multivariate_student_t(self,df, loc=0.0, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Multivariate Student's t 
        The Multivariate Student's t distribution is a generalization of the Student's t
        distribution to multiple dimensions. It is a heavy-tailed distribution that is
        often used to model data that is not normally distributed.
        
        The PDF of the multivariate Student's t-distribution for a random vector ( \mathbf{x} \in \mathbb{R}^d ) is given by:

        $$
        f(\mathbf{x}) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \nu^{d/2} \pi^{d/2} | \Sigma|^{1/2}} \left(1 + \frac{1}{\nu} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)^{-(\nu + d)/2}
        $$

        where:

        * $ \Gamma(\cdot) $ is the Gamma function.
        * $ \mu $ is the mean vector.
        * $ \Sigma $ is the scale (covariance) matrix.
        * $ \nu $ is the degrees of freedom.
        * $ d $ is the dimensionality of $ \mathbf{x} $.
        
        #### Args:

        - *df* (jnp.ndarray): Degrees of freedom, must be positive.

        - *loc* (jnp.ndarray): Location vector, representing the mean of the distribution.
            scale_tril (jnp.ndarray): Lower triangular matrix defining the scale.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI MultivariateStudentT distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the MultivariateStudentT distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')
            m.dist.multivariate_student_t(
                df = 2,
            - *loc*=jnp.array([1.0, 0.0, -2.0]), 
                scale_tril=jnp.linalg.cholesky(
                    jnp.array([[ 2.0,  0.7, -0.3],
                                [ 0.7,  1.0,  0.5],
                                [-0.3,  0.5,  1.5]])), 
                sample=True
            )   

        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#multivariatestudentt

        #### Reference:

        * [Multivariate Student-t distribution – Wikipedia](https://en.wikipedia.org/wiki/Multivariate_Student%27s_t-distribution)
        """

           
        d = numpyro.distributions.continuous.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def negative_binomial2(self,mean, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### 
            Samples from a Negative Binomial 
        
            This distribution is parameterized as a GammaPoisson with a modified rate.
            It represents the number of events occurring in a fixed amount of time or trials,
            where each event has a probability of success.
        
            $$
            P(k) = \frac{\Gamma(k + \alpha)}{\Gamma(k + 1) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^k \left(1 - \frac{\beta}{\alpha + \beta}\right)^k
            $$

            #### Args:
                mean (jnp.ndarray or float): The mean of the distribution.  This is equivalent to the `mu` parameter.
        
                concentration (jnp.ndarray or float): The concentration parameter. This is equivalent to the `alpha` parameter.
        
            - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
            - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
            - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

            - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
            - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            #### Returns:
                BI NegativeBinomial distribution object: When `sample=False` (for model building).
                jnp.ndarray: A JAX array of samples drawn from the NegativeBinomial distribution (for direct sampling).
                BI NegativeBinomial distribution object: When `create_obj=True` (for advanced use cases).
        
            #### Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.negative_binomial2(mean=2.0, concentration=3.0, sample=True)
        
            #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomial
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def negative_binomial_logits(self,total_count, logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Negative Binomial Logits
        
        Samples from a Negative Binomial Logits distribution.
        
        The Negative Binomial Logits distribution is a generalization of the Negative Binomial distribution where the parameter 'r' (number of successes) is expressed as a function of a logit parameter. This allows for more flexible modeling of count data.
        
        $$
        P(k) = \frac{e^{-n \cdot \text{softplus}(x)} \cdot \text{softplus}(-x)^k}{k!}
        $$
        
        #### Args:
            total_count (jnp.ndarray): The parameter controlling the shape of the distribution.  Represents the total number of trials.
        
            logits (jnp.ndarray): The log-odds parameter.  Related to the probability of success.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            Negative Binomial Logits: A BI Negative Binomial Logits distribution object (for model building).
        
            jnp.ndarray: A JAX array of samples drawn from the Negative Binomial Logits distribution (for direct sampling).
        
            Negative Binomial Logits: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial_logits(total_count=5.0, logits=0.0, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#Negative Binomial Logits
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def negative_binomial_probs(self,total_count, probs, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Negative Binomial with probabilities.
        
        The Negative Binomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is characterized by two parameters: 'concentration' (r) and 'rate' (p).  In this implementation, the 'concentration' parameter is derived from 'total_count' and the 'rate' parameter is derived from 'probs'.
        
        $$
        P(k) = \binom{k+r-1}{r-1} p^r (1-p)^k
        $$
        
        #### Args:
            total_count (jnp.ndarray):  A numeric vector, matrix, or array representing the parameter.

        - *probs* (jnp.ndarray): A numeric vector representing event probabilities. Must sum to 1.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample  site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI NegativeBinomialProbs distribution object (for model building).
            JAX array of samples drawn from the NegativeBinomialProbs distribution (for direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial_probs(total_count=10.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomialprobs
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def normal(self,loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Normal 
        Samples from a Normal (Gaussian) distribution.
        
        The Normal distribution is characterized by its mean (loc) and standard deviation (scale).
        It's a continuous probability distribution that arises frequently in statistics and
        probability theory.
        
        $$   p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
        $$
        
        #### Args:
        - *loc* (jnp.ndarray): The mean of the distribution.
        
        - *sample* (jnp.ndarray): The standard deviation of the distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Normal distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the Normal distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.normal(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#normal
        """

           
        d = numpyro.distributions.continuous.Normal(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def ordered_logistic(self,predictor, cutpoints, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Ordered Logistic
        
        A categorical distribution with ordered outcomes. This distribution represents the probability of an event falling into one of several ordered categories, based on a predictor variable and a set of cutpoints. The probability of an event falling into a particular category is determined by the number of categories above it.
        
        $$   P(Y = k) = \begin{cases}
                1 & \text{if } k = 0 \\
                \frac{1}{k} & \text{if } k > 0
            \end{cases}
        $$
        
        #### Args:

        - predictor (jnp.ndarray): Prediction in real domain; typically this is output of a linear model.
        
        - cutpoints (jnp.ndarray): Positions in real domain to separate categories.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI OrderedLogistic distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the OrderedLogistic distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.ordered_logistic(predictor=jnp.array([0.2, 0.5, 0.8]), cutpoints=jnp.array([-1.0, 0.0, 1.0]), sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#orderedlogistic
        """

           
        d = numpyro.distributions.discrete.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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


    def pareto(self,scale, alpha, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Pareto 
        Samples from a Pareto distribution.
        
        The Pareto distribution is a power-law probability distribution that is often
        used to model income, wealth, and the size of cities. It is defined by two
        parameters: alpha (shape) and scale.
        
        $$
           f(x) = \frac{\alpha \cdot \text{scale}^{\alpha}}{x^{\alpha + 1}}
            \text{ for } x \geq \text{scale}
        $$
        
        #### Args:
        - *sample* (jnp.ndarray or float): Scale parameter of the Pareto distribution.
                Must be positive.
        
            alpha (jnp.ndarray or float): Shape parameter of the Pareto distribution.
                Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Pareto distribution object (for model
                building).
        
      - When `sample=True`: A JAX array of samples drawn from the Pareto
                distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced
                use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.pareto(scale=2.0, alpha=3.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#pareto
        """

           
        d = numpyro.distributions.continuous.Pareto(scale=scale, alpha=alpha, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def poisson(self,rate, is_sparse=False, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Poisson

        Creates a Poisson distribution, a discrete probability distribution that models the number of events occurring in a fixed interval of time or space if these events occur with a known average rate and independently of the time since the last event.
        
        $$  \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}
        $$
        
        #### Args:
        - *rate* (jnp.ndarray): The rate parameter, representing the average number of events.
            is_sparse (bool, optional): Indicates whether the `rate` parameter is sparse. If `True`, a specialized sparse sampling implementation is used, which can be more efficient for models with many zero-rate components (e.g., zero-inflated models). Defaults to `False`.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
        - When `sample=False`: A BI Poisson distribution object (for model building).
        - When `sample=True`: A JAX array of samples drawn from the Poisson distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.poisson(rate=2.0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#poisson
        """

           
        d = numpyro.distributions.discrete.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def projected_normal(self,concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Projected Normal 
        
        This distribution over directional data is qualitatively similar to the von
        Mises and von Mises-Fisher distributions, but permits tractable variational
        inference via reparametrized gradients.
        
        $$   p(x) = \frac{1}{Z} \exp\left(-\frac{1}{2\sigma^2} ||x - \mu||^2\right)
        $$
        
        #### Args:
            concentration (jnp.ndarray): The concentration parameter, representing the
                direction towards which the samples are concentrated.  Must be a
                JAX array with at least one dimension.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI ProjectedNormal distribution object (for
                model building).
        
      - When `sample=True`: A JAX array of samples drawn from the
                ProjectedNormal distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for
                advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.projected_normal(concentration=jnp.array([1.0, 3.0, 2.0]), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#projectednormal
        """

           
        d = numpyro.distributions.directional.ProjectedNormal(concentration=concentration, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def relaxed_bernoulli_logits(self,temperature, logits, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Relaxed Bernoulli Logits
        
            Represents a relaxed version of the Bernoulli distribution, parameterized by logits and a temperature.
            The temperature parameter controls the sharpness of the distribution.  The distribution is defined
            by transforming the output of a Logistic distribution through a sigmoid function.
        
            $$
                P(x) = \sigma\left(\frac{x}{\text{temperature}}\right)
            $$

            #### Args:
                temperature (jnp.ndarray): The temperature parameter, must be positive.
                logits (jnp.ndarray): The logits parameter.
        
            - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                    this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
                    (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.
                
            - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
            - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

            - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
            - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            #### Returns:
                RelaxedBernoulliLogits: A BI RelaxedBernoulliLogits distribution object (for model building).
                jnp.ndarray: A JAX array of samples drawn from the RelaxedBernoulliLogits distribution (for direct sampling).
                RelaxedBernoulliLogits: The raw BI distribution object (for advanced use cases).
        
            #### Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.relaxed_bernoulli_logits(temperature=1.0, logits=0.0, sample=True)
        
            #### Wrapper of:
                https://num.pyro.ai/en/stable/distributions.html#relaxed-bernoulli-logits
        """

           
        d = numpyro.distributions.continuous.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def right_truncated_distribution(self,base_dist, high=0.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Right Truncated
        
        Samples from a right-truncated distribution.
        
        This distribution truncates the base distribution at a specified high value.  Values greater than `high` are discarded,
        effectively creating a distribution that is only supported up to that point. This is useful for modeling data
        where observations are only possible within a certain range.
        
        The probability density function (PDF) of the truncated distribution is:

        $$
           f_{\text{trunc}}(x) = \frac{f_{\text{base}}(x)}{F_{\text{base}}(\text{high})} \quad \text{for } x \le \text{high}
        $$

        where $f_{\text{base}}(x)` is the PDF of the base distribution and
        $F_{\text{base}}(\text{high})` is the cumulative distribution function (CDF)
        of the base distribution evaluated at `high`.
    
        
        where $f(x)` is the probability density function (PDF) of the base distribution and $P(X \le high)` is the
        cumulative distribution function (CDF) of the base distribution evaluated at `high`.
        $$
        
        #### Args:
            base_dist: The base distribution to truncate.  Must be a univariate distribution with real support.
        
            high (float, jnp.ndarray, optional): The upper truncation point. The support of the new distribution is $(-\infty, \text{high}]`. Defaults to 0.0.
            
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI RightTruncatedDistribution distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the RightTruncatedDistribution distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.right_truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#righttruncateddistribution
        """

           
        d = numpyro.distributions.truncated.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def sine_bivariate_vonmises(self,phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None, weighted_correlation=None,    validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0, create_obj=False):

        r"""### Sine Bivariate Von Mises

        A unimodal distribution for two dependent angles on the 2-torus
        ($S^1 \otimes S^1`), which is useful for modeling coupled angles like
        torsion angles in peptide chains. [1]

        The probability density function is given by:

        $$
           C^{-1}\exp(\kappa_1\cos(x_1-\mu_1) + \kappa_2\cos(x_2 -\mu_2) + \rho\sin(x_1 - \mu_1)\sin(x_2 - \mu_2))
        $$

        where the normalization constant $C` is:

        $$
           C = (2\pi)^2 \sum_{i=0}^{\infty} \binom{2i}{i} \left(\frac{\rho^2}{4\kappa_1\kappa_2}\right)^i I_i(\kappa_1)I_i(\kappa_2)
        $$

        Here, $I_i(\cdot)` is the modified Bessel function of the first kind,
        $\mu`'s are the locations, $\kappa`'s are the concentrations,
        and $\rho` represents the correlation between the angles $x_1` and $x_2`.
        

        #### Args:
        - *phi_loc* (jnp.ndarray): The location parameter for the first angle (phi).

        - *psi_loc* (jnp.ndarray): The location parameter for the second angle (psi).

        - *phi_concentration* (jnp.ndarray): The concentration parameter for the first angle (phi). Must be positive.

        - *psi_concentration* (jnp.ndarray): The concentration parameter for the second angle (psi). Must be positive.

        - *correlation* (jnp.ndarray, optional): The correlation parameter between the two angles. One of `correlation` or `weighted_correlation` must be specified.

        - *weighted_correlation* (jnp.ndarray, optional): An alternative correlation parameter. One of `correlation` or `weighted_correlation` must be specified.

        - *validate_args* (bool, optional): Whether to enable validation of distribution parameters. Defaults to `None`.

        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.

        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations. If provided, events with a `True` mask will be conditioned on `obs`, while the remaining events will be treated as latent variables. Defaults to `None`.

        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution,   bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.

        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *shape* (tuple, optional): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to  set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.

        - *event* (int, optional): The number of batch dimensions to reinterpret as event dimensions (used in model building).

        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for  building complex distributions like `MixtureSameFamily`. Defaults to `False`.

        #### Returns:
            BI.primitives.Messenger: A BI sample site object when used in a model context (`sample=False`).
            jnp.ndarray: A JAX array of samples drawn from the SineBivariateVonMises distribution (for direct sampling, `sample=True`).
            numpyro.distributions.Distribution: The raw BI distribution object (if `create_obj=True`).

        #### Example Usage:
            from BI import bi
            import jax.numpy as jnp
            m = bi('cpu')

            # Direct sampling
            samples = m.dist.sine_bivariate_vonmises(
                phi_loc=0.0,
                psi_loc=jnp.pi,
                phi_concentration=1.0,
                psi_concentration=1.0,
                correlation=0.5,
                sample=True,
            - *shape*=(10,)
            )

            # Usage within a model
            def my_model():
                angles = m.dist.sine_bivariate_vonmises(
                    phi_loc=0.0,
                    psi_loc=0.0,
                    phi_concentration=2.0,
                    psi_concentration=2.0,
                    weighted_correlation=0.9,
                    name='angles'
                )
                # ... rest of the model

        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#sinebivariatevonmises
        """

        d = numpyro.distributions.directional.SineBivariateVonMises(
            phi_loc=phi_loc,
            psi_loc=psi_loc,
            phi_concentration=phi_concentration,
            psi_concentration=psi_concentration,
            correlation=correlation,
            weighted_correlation=weighted_correlation,
            validate_args=validate_args
        )
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            return d.sample(seed_key, sample_shape=shape)
        else:
            if shape:
                d = d.expand(shape)
            if event > 0:
                d = d.to_event(event)
            if create_obj:
                return d
            else:
                infer_dict = {'obs_mask': mask} if mask is not None else None
                return numpyro.sample(name, d, obs=obs, infer=infer_dict)

    def sine_skewed(self,base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        
        r"""### Sine-skewing

        Sine-skewing [1] is a procedure for producing a distribution that breaks pointwise symmetry on a torus distribution. The new distribution is called the Sine Skewed X distribution, where X is the name of the (symmetric) base distribution. Torus distributions are distributions with support on products of circles (i.e., $\otimes S^1` where $S^1 = [-pi,pi)`).
        So, a 0-torus is a point, the 1-torus is a circle, and the 2-torus is commonly associated with the donut shape.
        
        .. note: This distribution is available in BI: [https://num.pyro.ai/en/stable/distributions.html#sineskewed](https://num.pyro.ai/en/stable/distributions.html#sineskewed)
        
        **Parameters:**
        
        *   **base\_dist:** Base density on a d-dimensional torus. Supported base distributions include: 1D :class:`~numpyro.distributions.VonMises`, :class:`~numnumpyro.distributions.SineBivariateVonMises`, 1D :class:`~numpyro.distributions.ProjectedNormal`, and :class:`~numpyro.distributions.Uniform` (-pi, pi).
        *   **skewness:** Skewness of the distribution.
                
        *   **sample (bool, optional):** A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        *   **seed (int, optional):** An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        *   **obs (jnp.ndarray, optional):** The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        *   **name (str, optional):** The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        **PDF:**
        
        The probability density function (PDF) of the Sine Skewed X distribution is not explicitly defined here, but it is derived from the base distribution and the skewness parameter.
        
        #### Example Usage:        
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
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def soft_laplace(self,loc, scale, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### SoftLaplace
        
        Samples from a SoftLaplace distribution.
        
        This distribution is a smooth approximation of a Laplace distribution,
        characterized by its log-convex density. It offers Laplace-like tails
        while being infinitely differentiable, making it suitable for HMC and
        Laplace approximation.
        
        $$
        f(x) = \log\!\left(\tfrac{2}{\pi}\right) - \log(\text{scale})
               - \log\!\left( e^{\tfrac{x - \text{loc}}{\text{scale}}} + e^{-\tfrac{x - \text{loc}}{\text{scale}}} \right)
        $$
        
        #### Args:
        - *loc*: Location parameter.
            scale: Scale parameter.
        $$
        
        #### Args:
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI SoftLaplace distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the SoftLaplace distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.soft_laplace(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#softlaplace
        """

           
        d = numpyro.distributions.continuous.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def student_t(self,df, loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Student's t
        
        The Student's t-distribution is a probability distribution that arises in hypothesis testing involving the mean of a normally distributed population when the population standard deviation is unknown. It is similar to the normal distribution, but has heavier tails, making it more robust to outliers.
        
        $$
           f(x) = \frac{1}{\Gamma(\nu/2) \sqrt{\nu \pi}} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}
        $$
        
        #### Args:
            df (jnp.ndarray): Degrees of freedom, must be positive.
        - *loc* (jnp.ndarray): Location parameter, defaults to 0.0.
        - *sample* (jnp.ndarray): Scale parameter, defaults to 1.0.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI StudentT distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the StudentT distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.student_t(df = 2, loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#studentt
        """

           
        d = numpyro.distributions.continuous.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def truncated_polya_gamma(self,batch_shape=(), validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Truncated PolyaGamma 
        Samples from a Truncated PolyaGamma distribution.
        
        This distribution is a truncated version of the PolyaGamma distribution,
        defined over the interval [0, truncation_point]. It is often used in
        Bayesian non-parametric models.
        
        $$   p(x) = \frac{1}{Z} \exp\left( \sum_{n=0}^{N} \left( \log(2n+1) - 1.5 \log(x) - \frac{(2n+1)^2}{4x} \right) \right)
        $$
        
        #### Args:
            batch_shape (tuple): The shape of the batch dimension.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions.
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Truncated PolyaGamma distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Truncated PolyaGamma distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_polya_gamma(batch_shape=(), sample=True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#truncatedpolygammadistribution
        """

           
        d = numpyro.distributions.truncated.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def two_sided_truncated_distribution(self,base_dist, low=0.0, high=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Two Sided Truncated 
        This distribution truncates a base distribution between two specified lower and upper bounds.
        
        $$
        f(x) =
        \begin{cases}
        \dfrac{p(x)}{P(\text{low} \le X \le \text{high})}, & \text{if } \text{low} \le x \le \text{high}, \\[6pt]
        0, & \text{otherwise}.
        \end{cases}
        $$

        where $p(x)$ is the probability density function of the base distribution.
        
        #### Args:
            base_dist: The base distribution to truncate.

            low: The lower bound for truncation.

            high: The upper bound for truncation.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI TwoSidedTruncatedDistribution distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the TwoSidedTruncatedDistribution distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution
        """
           
        d = numpyro.distributions.truncated.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def uniform(self,low=0.0, high=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Uniform 
        Samples from a Uniform distribution, which is a continuous probability distribution
        where all values within a given interval are equally likely.
        
        $$   f(x) = \frac{1}{b - a}, \text{ for } a \le x \le b
        $$
        
        #### Args:
            low (jnp.ndarray): The lower bound of the uniform interval.
        
            high (jnp.ndarray): The upper bound of the uniform interval.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling), this
                is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object
                instead of creating a sample site. This is essential for building
                complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Uniform distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the Uniform distribution (for direct
            sampling) when `sample=True`.
        
            The raw BI distribution object (for advanced use cases) when
            `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.uniform(low=0.0, high=1.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#uniform
        """

           
        d = numpyro.distributions.continuous.Uniform(low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def unit(self,log_factor, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Unit 
            The Unit distribution is a trivial, non-normalized distribution representing the unit type.
            It has a single value with no data, effectively a placeholder often used in probabilistic programming
            for situations where no actual data is involved.
        
            $$
                p(x) = 1
            $$
        
            #### Args:
                log_factor (jnp.ndarray): Log factor for the unit distribution. This parameter determines the
                - *shape* and batch size of the distribution.
        
            - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                    this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True`
                    (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample
                    site. This is essential for building complex distributions like `MixtureSameFamily`.
                
            - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
            - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

            - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
            - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            #### Returns:
                BI Unit distribution object: When `sample=False` (for model building).
                jnp.ndarray: A JAX array of samples drawn from the Unit distribution (for direct sampling).
                BI Unit distribution object: When `create_obj=True` (for advanced use cases).
        
            #### Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.unit(log_factor=jnp.ones(5), sample=True)
        
            #### Wrapper of:
                https://num.pyro.ai/en/stable/distributions.html#unit
        """

           
        d = numpyro.distributions.distribution.Unit(log_factor=log_factor, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def weibull(self,scale, concentration, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Weibull 
        Samples from a Weibull distribution.
        
        The Weibull distribution is a versatile distribution often used to model failure rates in engineering and reliability studies. It is characterized by its shape and scale parameters.
        
        $$
           f(x) = \frac{\beta}{\alpha} \left(\frac{x}{\alpha}\right)^{\beta - 1} e^{-\left(\frac{x}{\alpha}\right)^{\beta}} \text{ for } x \ge 0
        $$
               
        where $\alpha` is the scale parameter and $\beta` is the shape parameter.

        
        #### Args:
        - *sample* (jnp.ndarray): The scale parameter of the Weibull distribution. Must be positive.
            concentration (jnp.ndarray): The shape parameter of the Weibull distribution. Must be positive.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Weibull distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Weibull distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.weibull(scale=1.0, concentration=2.0, sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#weibull
        """

           
        d = numpyro.distributions.continuous.Weibull(scale=scale, concentration=concentration, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def wishart(self,concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Wishart 
        The Wishart distribution is a multivariate distribution used to model positive definite matrices,
        often representing covariance matrices. It's commonly used in Bayesian statistics and machine learning,
        particularly in models involving covariance estimation.
        
        $$   p(X) = \frac{1}{W^{p/2} \Gamma_p(concentration/2)} \left|X\right|^{-concentration/2} \exp\left(-\frac{1}{2} \text{tr}(X^{-1} X)\right)
        $$
        
        #### Args:
            concentration (jnp.ndarray): Positive concentration parameter analogous to the
                concentration of a :class:`Gamma` distribution. The concentration must be larger
                than the dimensionality of the scale matrix.
        
            scale_matrix (jnp.ndarray, optional): Scale matrix analogous to the inverse rate of a :class:`Gamma`
                distribution.
        
            rate_matrix (jnp.ndarray, optional): Rate matrix anaologous to the rate of a :class:`Gamma`
                distribution.
        
            scale_tril (jnp.ndarray, optional): Cholesky decomposition of the :code:`scale_matrix`.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample
                site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Wishart distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Wishart distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.wishart(concentration=5.0, scale_matrix=jnp.eye(2), sample=True)
        
        #### Wrapper of:
            https://num.pyro.ai/en/stable/distributions.html#wishart
        """

           
        d = numpyro.distributions.continuous.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def wishart_cholesky(self,concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Wishart Cholesky 

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
        
        - concentration: (Tensor) Positive concentration parameter analogous to the
            concentration of a :class:`Gamma` distribution. The concentration must be larger
            than the dimensionality of the scale matrix.

        - scale_matrix: (Tensor, optional) Scale matrix analogous to the inverse rate of a :class:`Gamma`
                distribution. If not provided, `rate_matrix` or `scale_tril` must be.

        - rate_matrix: (Tensor, optional) Rate matrix anaologous to the rate of a :class:`Gamma`
                distribution. If not provided, `scale_matrix` or `scale_tril` must be.

        - scale_tril: (Tensor, optional) Cholesky decomposition of the :code:`scale_matrix`.
            If not provided, `scale_matrix` or `rate_matrix` must be.
                
        - sample (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - seed (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - obs (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - name (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        """
        
        d = numpyro.distributions.continuous.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def zero_inflated_poisson(self,gate, rate=1.0, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### A Zero Inflated Poisson 
        
            This distribution combines two Poisson processes: one with a rate parameter and another that generates only zeros.
            The probability of observing a zero is determined by the 'gate' parameter, while the probability of observing a non-zero value is governed by the 'rate' parameter of the underlying Poisson distribution.
        
            $$
            P(X = k) = (1 - gate) * \frac{e^{-rate} rate^k}{k!} + gate
            $$
        
            #### Args:
                rate (jnp.ndarray): The rate parameter of the underlying Poisson distribution.
        
            - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
            - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
            - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
            - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
            - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
            - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

            - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
            - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
            #### Returns:
                BI ZeroInflatedPoisson distribution object (when `sample=False`).
                JAX array of samples drawn from the ZeroInflatedPoisson distribution (when `sample=True`).
                The raw BI distribution object (when `create_obj=True`).
        
            #### Example Usage:
                from BI import bi
                m = bi('cpu')
                m.dist.zero_inflated_poisson(gate = 0.3, rate=2.0, sample=True)
        
            #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflatedpoisson
        """

           
        d = numpyro.distributions.discrete.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def zero_sum_normal(self,scale, event_shape, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Zero Sum Normal 
        Samples from a ZeroSumNormal distribution, which is a Normal distribution where one or more axes are constrained to sum to zero.
        
        $$
        ZSN(\sigma) = N(0, \sigma^2 (I - \tfrac{1}{n}J)) \\
        \text{where} \ ~ J_{ij} = 1 \ ~ \text{and} \\
        n = \text{number of zero-sum axes}
        $$
        
        #### Args:
        - *sample* (array_like): Standard deviation of the underlying normal distribution before the zerosum constraint is enforced.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI ZeroSumNormal distribution object (for model building).
        
      - When `sample=True`: A JAX array of samples drawn from the ZeroSumNormal distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_sum_normal(scale=1.0, event_shape = (2,), sample = True)
        
        #### Wrapper of:
        https://num.pyro.ai/en/stable/distributions.html#zerosumnormal
        """

           
        d = numpyro.distributions.continuous.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def bernoulli(self,probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Bernoulli
        
        The Bernoulli distribution models a single trial with two possible outcomes: success or failure.
        It is parameterized by the probability of success, often denoted as 'p'.
        
        $$   P(X=1) = p \\
           P(X=0) = 1 - p
        $$
        
        #### Args:
        - *probs* (jnp.ndarray, optional):  Probability of success for each Bernoulli trial. Must be between 0 and 1.
            logits (jnp.ndarray, optional): Log-odds of success for each Bernoulli trial.  `probs = sigmoid(logits)`.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI Bernoulli distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Bernoulli distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.bernoulli(probs=0.7, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#bernoulli
        """

           
        d = numpyro.distributions.discrete.Bernoulli(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def binomial(self,total_count=1, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Binomial
        
        The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials.
        It represents the probability of obtaining exactly *k* successes in *n* trials, where each trial has a probability *p* of success.
        
        $$   P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
        $$
        
        #### Args:
        - *total_count* (int): The number of trials *n*.
        
        - *probs* (jnp.ndarray, optional): The probability of success *p* for each trial. Must be between 0 and 1.
            
        - *logits* (jnp.ndarray, optional): The log-odds of success for each trial.  `probs = jax.nn.sigmoid(logits)`.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            Binomial distribution object (for model building) when `sample=False`.
            JAX array of samples drawn from the Binomial distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.binomial(total_count=10, probs=0.5, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#binomial
        """

           
        d = numpyro.distributions.discrete.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def categorical(self,probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Categorical distribution.
        
        The Categorical distribution, also known as the multinomial distribution,
        describes the probability of different outcomes from a finite set of possibilities.
        It is commonly used to model discrete choices or classifications.
        
        $$   
        P(k) = \frac{e^{\log(p_k)}}{\sum_{j=1}^{K} e^{\log(p_j)}}
        $$

        where $p_k$ is the probability of outcome $k`, and the sum is over all possible outcomes.
        
        #### Args:
        - *probs* (jnp.ndarray): A 1D array of probabilities for each category. Must sum to 1.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                    (model building), this is used with `.expand(shape)` to set the
                    distribution's batch shape. When `sample=True` (direct sampling),
                    this is used as `sample_shape` to draw a raw JAX array of the
                    given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                    dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution
                    object instead of creating a sample site. This is essential for
                    building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

      - When `sample=False`: A BI Categorical distribution object (for model building).

      - When `sample=True`: A JAX array of samples drawn from the Categorical distribution (for direct sampling).

      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.categorical(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#categorical
        """

           
        d = numpyro.distributions.discrete.Categorical(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def geometric(self,probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Geometric distribution.
        
        The Geometric distribution models the number of failures before the first success in a sequence of Bernoulli trials.
        It is characterized by a single parameter, the probability of success on each trial.
        
        $$   P(X = k) = (1 - p)^k p
        $$
        
        #### Args:
        - *probs* (jnp.ndarray, optional): Probability of success on each trial. Must be between 0 and 1.
            logits (jnp.ndarray, optional): Log-odds of success on each trial.  `probs = jax.nn.sigmoid(logits)`.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Geometric distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.geometric(probs=0.5, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#geometric
        """

           
        d = numpyro.distributions.discrete.Geometric(probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def mixture(self,mixing_distribution, component_distributions, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=0,  shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### A marginalized finite mixture of component distributions.
        
        This distribution represents a mixture of component distributions, where the
        mixing weights are determined by a Categorical distribution. The resulting
        distribution can be either a MixtureGeneral (when component distributions
        are a list) or a MixtureSameFamily (when component distributions are a single
        distribution).
        
        $$   p(x) = \sum_{i=1}^{K} w_i p_i(x)
        $$
        
        #### Args:

         - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                    (model building), this is used with `.expand(shape)` to set the
                    distribution's batch shape. When `sample=True` (direct sampling),
                    this is used as `sample_shape` to draw a raw JAX array of the
                    given shape.
        
         - *event* (int): The number of batch dimensions to reinterpret as event
                    dimensions (used in model building).
        
         - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
         - *create_obj* (bool): If True, returns the raw BI distribution
                    object instead of creating a sample site. This is essential for
                    building complex distributions like `MixtureSameFamily`.
                
         - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
         - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

         - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
         - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:

      - When `sample=False`: A BI Mixture distribution object (for model
                building).
        
      - When `sample=True`: A JAX array of samples drawn from the Mixture
                distribution (for direct sampling).
        
      - When `create_obj=True`: The raw BI distribution object (for advanced
                use cases).
        
        #### Example Usage:
            from jax import random
            import BI as pyro
            m = pyro.distributions.Mixture(
                pyro.distributions.Categorical(torch.ones(2)),
                [pyro.distributions.Normal(0, 1), pyro.distributions.Normal(2, 1)]
            )
            samples = m.sample(sample_shape=(10,))
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#mixture
        """

           
        d = numpyro.distributions.mixtures.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def multinomial(self,total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Multinomial
        
        Samples from a Multinomial distribution, which models the probability of different outcomes in a sequence of independent trials, each with a fixed number of trials and a fixed set of possible outcomes.  It generalizes the binomial distribution to multiple categories.
        
        $$   P(X = x) = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}
        $$
        
        #### Args:
            total_count (int or jnp.ndarray): The number of trials.
        
        - *probs* (jnp.ndarray, optional): Event probabilities. Must sum to 1.
        
            logits (jnp.ndarray, optional): Event log probabilities.
        
            total_count_max (int, optional): An optional integer providing an upper bound on `total_count`. This is used for performance optimization with `lax.scan` when `total_count` is a dynamic JAX tracer, helping to avoid recompilation.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool, optional): Optional boolean array to mask observations.
        
        - *create_obj* (bool, optional): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI Multinomial distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the Multinomial distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#multinomial
        """

           
        d = numpyro.distributions.discrete.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def relaxed_bernoulli(self,temperature, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Relaxed Bernoulli 
        
        The Relaxed Bernoulli distribution is a continuous relaxation of the discrete Bernoulli distribution.
        It's useful for variational inference and other applications where a differentiable approximation of the Bernoulli is needed.
        The probability density function (PDF) is defined as:
        
        $$
           p(x) = \frac{1}{2} \left( 1 + \tanh\left(\frac{x - \beta \log(\frac{p}{1-p})}{1}\right) \right)
        $$
        
        #### Args:
            temperature (float): The temperature parameter.

        - *probs* (jnp.ndarray, optional): The probability of success. Must be in the interval `[0, 1]`. Only one of `probs` or `logits` can be specified.
            
            logits (jnp.ndarray, optional): The log-odds of success. Only one of `probs` or `logits` can be specified.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used
                with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
                used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI RelaxedBernoulli distribution object (for model building) when `sample=False`.
            A JAX array of samples drawn from the RelaxedBernoulli distribution (for direct sampling) when `sample=True`.
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.relaxed_bernoulli(temperature=1.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli
        """

           
        d = numpyro.distributions.continuous.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def truncated_cauchy(self,loc=0.0, scale=1.0, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Truncated Cauchy
        
        The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
        that appears frequently in various areas of mathematics and physics. It is characterized by its heavy tails,
        which extend to infinity. The truncated version limits the support of the Cauchy distribution to a specified interval.
        
        $$
           f(x) = \frac{1}{\pi \cdot c \cdot (1 + ((x - b) / c)^2)}  \text{ for } a < x < b
        $$
        
        #### Args:
        - *loc* (float): Location parameter of the Cauchy distribution.
        
        - *sample* (float): Scale parameter of the Cauchy distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI TruncatedCauchy distribution object (for model building) when `sample=False`.
        
            JAX array of samples drawn from the TruncatedCauchy distribution (for direct sampling) when `sample=True`.
        
            The raw BI distribution object (for advanced use cases) when `create_obj=True`.
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_cauchy(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncatedcauchy
        """

           
        d = numpyro.distributions.truncated.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def truncated_distribution(self,base_dist, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Truncated 
        
        Samples from a Truncated Distribution.
        
        This distribution represents a base distribution truncated between specified lower and upper bounds.
        The truncation modifies the probability density function (PDF) of the base distribution,
        effectively removing observations outside the defined interval.
        
        $$   p(x) = \frac{p(x)}{P(\text{lower} \le x \le \text{upper})}
        $$
        
        #### Args:
            base_dist: The base distribution to be truncated. This should be a univariate
                distribution. Currently, only the following distributions are supported:
                Cauchy, Laplace, Logistic, Normal, and StudentT.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the distribution's
                batch shape. When `sample=True` (direct sampling), this is used as `sample_shape`
                to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions
                (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of
                creating a sample site. This is essential for building complex distributions
                like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI TruncatedDistribution distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the TruncatedDistribution distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=1, low = 0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncateddistribution
        """

           
        d = numpyro.distributions.truncated.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def truncated_normal(self,loc=0.0, scale=1.0, low=None, high=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Truncated Normal 
        
        The Truncated Normal distribution is a normal distribution truncated
        to a specified interval. It is defined by its location (`loc`), scale
        (`scale`), lower bound (`low`), and upper bound (`high`).
        
        $$
        f(x) = \frac{p(x)}{\alpha}, \quad x \in [\text{low}, \text{high}]
        $$
        
        where
        
        $$
        p(x) = \frac{1}{\text{scale}\,\sqrt{2\pi}}
               \exp\!\left(-\tfrac{1}{2}\left(\tfrac{x - \text{loc}}{\text{scale}}\right)^2\right),
        $$
        
        and
        
        $$
        \alpha = \int_{\text{low}}^{\text{high}} p(x)\,dx.
        $$
        
        #### Args:
        - *loc* (float): The location parameter of the normal distribution.
        - *sample* (float): The scale parameter of the normal distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False`
                (model building), this is used with `.expand(shape)` to set the
                distribution's batch shape. When `sample=True` (direct sampling),
                this is used as `sample_shape` to draw a raw JAX array of the
                given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event
                dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution
                object instead of creating a sample site. This is essential for
                building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI TruncatedNormal distribution object (for model building).
            JAX array of samples drawn from the TruncatedNormal distribution (for
            direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.truncated_normal(loc=0.0, scale=1.0, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#truncatednormal_lowercase
        """

           
        d = numpyro.distributions.truncated.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def zero_inflated_distribution(self,base_dist, gate=None, gate_logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Generic Zero Inflated
        
        A Zero-Inflated distribution combines a base distribution with a Bernoulli distribution
        to model data with an excess of zero values. It assumes that each observation
        is either drawn from the base distribution or is a zero with probability determined
        by the Bernoulli distribution (the "gate"). This is useful for modeling data
        where zeros are more frequent than expected under a single distribution,
        often due to a different underlying process.
        
        $$
           P(x) = \pi \cdot I(x=0) + (1 - \pi) \cdot P_{base}(x)
        $$where:
        - $P_{base}(x)$ is the probability density function (PDF) or probability mass function (PMF) of the base distribution.
        - $\pi$ is the probability of generating a zero, governed by the Bernoulli gate.
        - $I(x=0)$ is an indicator function that equals 1 if x=0 and 0 otherwise.

        
        #### Args:
            base_dist (Distribution): The base distribution to be zero-inflated (e.g., Poisson,     NegativeBinomial).

            gate (jnp.ndarray, optional): Probability of extra zeros (between 0 and 1).

            gate_logits (jnp.ndarray, optional): Log-odds of extra zeros.

            validate_args (bool, optional): Whether to validate parameter values. Defaults to None.

        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI ZeroInflatedDistribution distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the ZeroInflatedDistribution distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_inflated_distribution(base_dist=m.dist.poisson(rate=5, create_obj = True), gate = 0.3, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution
        """

           
        d = numpyro.distributions.discrete.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def zero_inflated_negative_binomial2(self,mean, concentration, gate=None, gate_logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Zero-Inflated Negative Binomial
        
        This distribution combines a Negative Binomial distribution with a binary gate variable.
        Observations are either drawn from the Negative Binomial  distribution with probability
        (1 - gate) or are treated as zero with probability 'gate'. This models data with excess zeros
        compared to what a standard Negative Binomial distribution would predict.
        
        $$
           P(X = x) = (1 - gate) \cdot \frac{\Gamma(x + \alpha)}{\Gamma(x + \alpha + \beta) \Gamma(\alpha)} \left(\frac{\beta}{\alpha + \beta}\right)^x + gate \cdot \delta_{x, 0}
        $$
        
        #### Args:
            mean (jnp.ndarray or float): The mean of the Negative Binomial 2 distribution.
            concentration (jnp.ndarray or float): The concentration parameter of the Negative Binomial 2 distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building),
                this is used with `.expand(shape)` to set the distribution's batch shape.
                When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw
                JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site.
                This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
      - When `sample=False`: A BI ZeroInflatedNegativeBinomial2 distribution object (for model building).
      - When `sample=True`: A JAX array of samples drawn from the ZeroInflatedNegativeBinomial2 distribution (for direct sampling).
      - When `create_obj=True`: The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.zero_inflated_negative_binomial2(mean=2.0, concentration=3.0, gate = 0.3, sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#zeroinflatednegativebinomial2
        """

           
        d = numpyro.distributions.conjugate.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

    def negative_binomial(self,total_count, probs=None, logits=None, validate_args=None, name='x', obs=None, mask=None, sample=False, seed=None, shape=(), event=0,create_obj=False, to_jax = True):
        

        r"""### Negative Binomial 
        
        The NegativeBinomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials. It is characterized by two parameters: 'total_count' (r) and 'probs' or 'logits' (p).
        
        $$P(k) = \binom{k+r-1}{r-1} p^r (1-p)^k
        $$

                
        #### Args:
            total_count (jnp.ndarray): The total number of events.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        - *create_obj* (bool): If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
                
        - *sample* (bool, optional): A control-flow argument. If `True`, the function will directly sample a raw JAX array from the distribution, bypassing the BI model context. If `False`, it will create a `BI.sample` site within a model. Defaults to `False`.
            
        - *seed* (int, optional): An integer used to generate a JAX PRNGKey for reproducible sampling when `sample=True`. [7] This argument has no effect when `sample=False`, as randomness is handled by BI's inference engine. Defaults to 0.

        - *obs* (jnp.ndarray, optional): The observed value for this random variable. If provided, the sample site is conditioned on this value, and the    function returns the observed value. If `None`, the site is treated as a latent (unobserved) random variable. Defaults to `None`.
    
        - *name* (str, optional): The name of the sample site in a BI model. This is used to uniquely identify the random variable. Defaults to 'x'.
            
        #### Returns:
            BI NegativeBinomial distribution object (for model building).
            JAX array of samples drawn from the NegativeBinomial distribution (for direct sampling).
            The raw BI distribution object (for advanced use cases).
        
        #### Example Usage:
            from BI import bi
            m = bi('cpu')
            m.dist.negative_binomial(total_count=5.0,probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
        
        #### Wrapper of: https://num.pyro.ai/en/stable/distributions.html#negativebinomial
        """

           
        d = numpyro.distributions.conjugate.NegativeBinomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
        if sample:
            seed = self.normalize_seed(seed)
            seed_key = random.PRNGKey(seed)
            if to_jax :
                return d.sample(seed_key,   sample_shape=shape)
            else:
                return SampledData(d.sample(seed_key,   sample_shape=shape))
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

