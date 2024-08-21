import numpyro as numpyro
from numpyro import handlers

class sampler:

    def __init__(self):
        pass

    def asymmetriclaplace(self, loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        AsymmetricLaplace distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            asymmetry=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def asymmetriclaplacequantile(self, loc=0.0, scale=1.0, quantile=0.5, validate_args=None, sample_shape=[1], seed=0):
        """
        AsymmetricLaplaceQuantile distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            quantile=0.5
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def bernoulli(self, probs=None, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Bernoulli distribution.
    
            Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def bernoullilogits(self, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        BernoulliLogits distribution.
    
            Arguments:
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def bernoulliprobs(self, probs, validate_args=None, sample_shape=[1], seed=0):
        """
        BernoulliProbs distribution.
    
            Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def beta(self, concentration1, concentration0, validate_args=None, sample_shape=[1], seed=0):
        """
        Beta distribution.
    
            Arguments:
            concentration1
            concentration0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def betabinomial(self, concentration1, concentration0, total_count=1, validate_args=None, sample_shape=[1], seed=0):
        """
        BetaBinomial distribution.
    
            Arguments:
            concentration1
            concentration0
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def betaproportion(self, mean, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        BetaProportion distribution.
    
            Arguments:
            mean
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def binomial(self, total_count=1, probs=None, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Binomial distribution.
    
            Arguments:
            total_count=1
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def binomiallogits(self, logits, total_count=1, validate_args=None, sample_shape=[1], seed=0):
        """
        BinomialLogits distribution.
    
            Arguments:
            logits
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def binomialprobs(self, probs, total_count=1, validate_args=None, sample_shape=[1], seed=0):
        """
        BinomialProbs distribution.
    
            Arguments:
            probs
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def car(self, loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, sample_shape=[1], seed=0):
        """
        CAR distribution.
    
            Arguments:
            loc
            correlation
            conditional_precision
            adj_matrix
            is_sparse=False
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def categorical(self, probs=None, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Categorical distribution.
    
            Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def categoricallogits(self, logits, validate_args=None, sample_shape=[1], seed=0):
        """
        CategoricalLogits distribution.
    
            Arguments:
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def categoricalprobs(self, probs, validate_args=None, sample_shape=[1], seed=0):
        """
        CategoricalProbs distribution.
    
            Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def cauchy(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Cauchy distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def chi2(self, df, validate_args=None, sample_shape=[1], seed=0):
        """
        Chi2 distribution.
    
            Arguments:
            df
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Chi2(df=df, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def delta(self, v=0.0, log_density=0.0, event_dim=0, validate_args=None, sample_shape=[1], seed=0):
        """
        Delta distribution.
    
            Arguments:
            v=0.0
            log_density=0.0
            event_dim=0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def dirichlet(self, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        Dirichlet distribution.
    
            Arguments:
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def dirichletmultinomial(self, concentration, total_count=1, validate_args=None, sample_shape=[1], seed=0):
        """
        DirichletMultinomial distribution.
    
            Arguments:
            concentration
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def discreteuniform(self, low=0, high=1, validate_args=None, sample_shape=[1], seed=0):
        """
        DiscreteUniform distribution.
    
            Arguments:
            low=0
            high=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def distribution(self, batch_shape=(), event_shape=(), validate_args=None, sample_shape=[1], seed=0):
        """
        Distribution distribution.
    
            Arguments:
            batch_shape=()
            event_shape=()
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Distribution(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def eulermaruyama(self, t, sde_fn, init_dist, validate_args=None, sample_shape=[1], seed=0):
        """
        EulerMaruyama distribution.
    
            Arguments:
            t
            sde_fn
            init_dist
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def expandeddistribution(self, base_dist, batch_shape=(), sample_shape=[1], seed=0):
        """
        ExpandedDistribution distribution.
    
            Arguments:
            base_dist
            batch_shape=()
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def exponential(self, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Exponential distribution.
    
            Arguments:
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Exponential(rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def foldeddistribution(self, base_dist, validate_args=None, sample_shape=[1], seed=0):
        """
        FoldedDistribution distribution.
    
            Arguments:
            base_dist
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gamma(self, concentration, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Gamma distribution.
    
            Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gammapoisson(self, concentration, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        GammaPoisson distribution.
    
            Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gaussiancopula(self, marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, sample_shape=[1], seed=0):
        """
        GaussianCopula distribution.
    
            Arguments:
            marginal_dist
            correlation_matrix=None
            correlation_cholesky=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gaussiancopulabeta(self, concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, sample_shape=[1], seed=0):
        """
        GaussianCopulaBeta distribution.
    
            Arguments:
            concentration1
            concentration0
            correlation_matrix=None
            correlation_cholesky=None
            validate_args=False
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gaussianrandomwalk(self, scale=1.0, num_steps=1, validate_args=None, sample_shape=[1], seed=0):
        """
        GaussianRandomWalk distribution.
    
            Arguments:
            scale=1.0
            num_steps=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def geometric(self, probs=None, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Geometric distribution.
    
            Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def geometriclogits(self, logits, validate_args=None, sample_shape=[1], seed=0):
        """
        GeometricLogits distribution.
    
            Arguments:
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def geometricprobs(self, probs, validate_args=None, sample_shape=[1], seed=0):
        """
        GeometricProbs distribution.
    
            Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gompertz(self, concentration, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Gompertz distribution.
    
            Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def gumbel(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Gumbel distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def halfcauchy(self, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        HalfCauchy distribution.
    
            Arguments:
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def halfnormal(self, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        HalfNormal distribution.
    
            Arguments:
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def improperuniform(self, support, batch_shape, event_shape, validate_args=None, sample_shape=[1], seed=0):
        """
        ImproperUniform distribution.
    
            Arguments:
            support
            batch_shape
            event_shape
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def independent(self, base_dist, reinterpreted_batch_ndims, validate_args=None, sample_shape=[1], seed=0):
        """
        Independent distribution.
    
            Arguments:
            base_dist
            reinterpreted_batch_ndims
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def inversegamma(self, concentration, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        InverseGamma distribution.
    
            Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def kumaraswamy(self, concentration1, concentration0, validate_args=None, sample_shape=[1], seed=0):
        """
        Kumaraswamy distribution.
    
            Arguments:
            concentration1
            concentration0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def lkj(self, dimension, concentration=1.0, sample_method='onion', validate_args=None, sample_shape=[1], seed=0):
        """
        LKJ distribution.
    
            Arguments:
            dimension
            concentration=1.0
            sample_method='onion'
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def lkjcholesky(self, dimension, concentration=1.0, sample_method='onion', validate_args=None, sample_shape=[1], seed=0):
        """
        LKJCholesky distribution.
    
            Arguments:
            dimension
            concentration=1.0
            sample_method='onion'
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def laplace(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Laplace distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def lefttruncateddistribution(self, base_dist, low=0.0, validate_args=None, sample_shape=[1], seed=0):
        """
        LeftTruncatedDistribution distribution.
    
            Arguments:
            base_dist
            low=0.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def lognormal(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        LogNormal distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def loguniform(self, low, high, validate_args=None, sample_shape=[1], seed=0):
        """
        LogUniform distribution.
    
            Arguments:
            low
            high
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def logistic(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Logistic distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def lowrankmultivariatenormal(self, loc, cov_factor, cov_diag, validate_args=None, sample_shape=[1], seed=0):
        """
        LowRankMultivariateNormal distribution.
    
            Arguments:
            loc
            cov_factor
            cov_diag
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def maskeddistribution(self, base_dist, mask, sample_shape=[1], seed=0):
        """
        MaskedDistribution distribution.
    
            Arguments:
            base_dist
            mask
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def matrixnormal(self, loc, scale_tril_row, scale_tril_column, validate_args=None, sample_shape=[1], seed=0):
        """
        MatrixNormal distribution.
    
            Arguments:
            loc
            scale_tril_row
            scale_tril_column
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def mixture(self, mixing_distribution, component_distributions, validate_args=None, sample_shape=[1], seed=0):
        """
        Mixture distribution.
    
            Arguments:
            mixing_distribution
            component_distributions
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def mixturegeneral(self, mixing_distribution, component_distributions, support=None, validate_args=None, sample_shape=[1], seed=0):
        """
        MixtureGeneral distribution.
    
            Arguments:
            mixing_distribution
            component_distributions
            support=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def mixturesamefamily(self, mixing_distribution, component_distribution, validate_args=None, sample_shape=[1], seed=0):
        """
        MixtureSameFamily distribution.
    
            Arguments:
            mixing_distribution
            component_distribution
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def multinomial(self, total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Multinomial distribution.
    
            Arguments:
            total_count=1
            probs=None
            logits=None
            total_count_max=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def multinomiallogits(self, logits, total_count=1, total_count_max=None, validate_args=None, sample_shape=[1], seed=0):
        """
        MultinomialLogits distribution.
    
            Arguments:
            logits
            total_count=1
            total_count_max=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def multinomialprobs(self, probs, total_count=1, total_count_max=None, validate_args=None, sample_shape=[1], seed=0):
        """
        MultinomialProbs distribution.
    
            Arguments:
            probs
            total_count=1
            total_count_max=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def multivariatenormal(self, loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, sample_shape=[1], seed=0):
        """
        MultivariateNormal distribution.
    
            Arguments:
            loc=0.0
            covariance_matrix=None
            precision_matrix=None
            scale_tril=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def multivariatestudentt(self, df, loc=0.0, scale_tril=None, validate_args=None, sample_shape=[1], seed=0):
        """
        MultivariateStudentT distribution.
    
            Arguments:
            df
            loc=0.0
            scale_tril=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def negativebinomial2(self, mean, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        NegativeBinomial2 distribution.
    
            Arguments:
            mean
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def negativebinomiallogits(self, total_count, logits, validate_args=None, sample_shape=[1], seed=0):
        """
        NegativeBinomialLogits distribution.
    
            Arguments:
            total_count
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def negativebinomialprobs(self, total_count, probs, validate_args=None, sample_shape=[1], seed=0):
        """
        NegativeBinomialProbs distribution.
    
            Arguments:
            total_count
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def normal(self, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Normal distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def orderedlogistic(self, predictor, cutpoints, validate_args=None, sample_shape=[1], seed=0):
        """
        OrderedLogistic distribution.
    
            Arguments:
            predictor
            cutpoints
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def pareto(self, scale, alpha, validate_args=None, sample_shape=[1], seed=0):
        """
        Pareto distribution.
    
            Arguments:
            scale
            alpha
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def poisson(self, rate, is_sparse=False, validate_args=None, sample_shape=[1], seed=0):
        """
        Poisson distribution.
    
            Arguments:
            rate
            is_sparse=False
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def projectednormal(self, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        ProjectedNormal distribution.
    
            Arguments:
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def relaxedbernoulli(self, temperature, probs=None, logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        RelaxedBernoulli distribution.
    
            Arguments:
            temperature
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def relaxedbernoullilogits(self, temperature, logits, validate_args=None, sample_shape=[1], seed=0):
        """
        RelaxedBernoulliLogits distribution.
    
            Arguments:
            temperature
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def righttruncateddistribution(self, base_dist, high=0.0, validate_args=None, sample_shape=[1], seed=0):
        """
        RightTruncatedDistribution distribution.
    
            Arguments:
            base_dist
            high=0.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def sinebivariatevonmises(self, phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None, weighted_correlation=None, validate_args=None, sample_shape=[1], seed=0):
        """
        SineBivariateVonMises distribution.
    
            Arguments:
            phi_loc
            psi_loc
            phi_concentration
            psi_concentration
            correlation=None
            weighted_correlation=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def sineskewed(self, base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, sample_shape=[1], seed=0):
        """
        SineSkewed distribution.
    
            Arguments:
            base_dist: numpyro.distributions.distribution.Distribution
            skewness
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def softlaplace(self, loc, scale, validate_args=None, sample_shape=[1], seed=0):
        """
        SoftLaplace distribution.
    
            Arguments:
            loc
            scale
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def studentt(self, df, loc=0.0, scale=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        StudentT distribution.
    
            Arguments:
            df
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def transformeddistribution(self, base_distribution, transforms, validate_args=None, sample_shape=[1], seed=0):
        """
        TransformedDistribution distribution.
    
            Arguments:
            base_distribution
            transforms
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def truncatedcauchy(self, loc=0.0, scale=1.0, low=None, high=None, validate_args=None, sample_shape=[1], seed=0):
        """
        TruncatedCauchy distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            low=None
            high=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def truncateddistribution(self, base_dist, low=None, high=None, validate_args=None, sample_shape=[1], seed=0):
        """
        TruncatedDistribution distribution.
    
            Arguments:
            base_dist
            low=None
            high=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def truncatednormal(self, loc=0.0, scale=1.0, low=None, high=None, validate_args=None, sample_shape=[1], seed=0):
        """
        TruncatedNormal distribution.
    
            Arguments:
            loc=0.0
            scale=1.0
            low=None
            high=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def truncatedpolyagamma(self, batch_shape=(), validate_args=None, sample_shape=[1], seed=0):
        """
        TruncatedPolyaGamma distribution.
    
            Arguments:
            batch_shape=()
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def twosidedtruncateddistribution(self, base_dist, low=0.0, high=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        TwoSidedTruncatedDistribution distribution.
    
            Arguments:
            base_dist
            low=0.0
            high=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def uniform(self, low=0.0, high=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        Uniform distribution.
    
            Arguments:
            low=0.0
            high=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def unit(self, log_factor, validate_args=None, sample_shape=[1], seed=0):
        """
        Unit distribution.
    
            Arguments:
            log_factor
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def vonmises(self, loc, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        VonMises distribution.
    
            Arguments:
            loc
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def weibull(self, scale, concentration, validate_args=None, sample_shape=[1], seed=0):
        """
        Weibull distribution.
    
            Arguments:
            scale
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def wishart(self, concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, sample_shape=[1], seed=0):
        """
        Wishart distribution.
    
            Arguments:
            concentration
            scale_matrix=None
            rate_matrix=None
            scale_tril=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def wishartcholesky(self, concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, sample_shape=[1], seed=0):
        """
        WishartCholesky distribution.
    
            Arguments:
            concentration
            scale_matrix=None
            rate_matrix=None
            scale_tril=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def zeroinflateddistribution(self, base_dist, gate=None, gate_logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        ZeroInflatedDistribution distribution.
    
            Arguments:
            base_dist
            gate=None
            gate_logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def zeroinflatednegativebinomial2(self, mean, concentration, gate=None, gate_logits=None, validate_args=None, sample_shape=[1], seed=0):
        """
        ZeroInflatedNegativeBinomial2 distribution.
    
            Arguments:
            mean
            concentration
            gate=None
            gate_logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def zeroinflatedpoisson(self, gate, rate=1.0, validate_args=None, sample_shape=[1], seed=0):
        """
        ZeroInflatedPoisson distribution.
    
            Arguments:
            gate
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def zerosumnormal(self, scale, event_shape, validate_args=None, sample_shape=[1], seed=0):
        """
        ZeroSumNormal distribution.
    
            Arguments:
            scale
            event_shape
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args)
            return numpyro.sample('x', distribution.expand(sample_shape))

    def kl_divergence(self, sample_shape=[1], seed=0, *args, **kwargs):
        """
        kl_divergence distribution.
    
            Arguments:
            *args
            **kwargs
            sample_shape: Shape of samples to be drawn.
        """
        with handlers.seed(rng_seed=seed):
            distribution = numpyro.distributions.kl_divergence(args=args, kwargs=kwargs)
            return numpyro.sample('x', distribution.expand(sample_shape))

