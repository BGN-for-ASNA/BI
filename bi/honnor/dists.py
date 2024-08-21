import numpyro as numpyro

class Dist:

    def __init__(self):
        pass

    def asymmetriclaplace(self, name, loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, sample_shape=()):
        """
        AsymmetricLaplace distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            asymmetry=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args).expand(sample_shape))

    def asymmetriclaplacequantile(self, name, loc=0.0, scale=1.0, quantile=0.5, validate_args=None, sample_shape=()):
        """
        AsymmetricLaplaceQuantile distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            quantile=0.5
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args).expand(sample_shape))

    def bernoulli(self, name, probs=None, logits=None, validate_args=None, sample_shape=()):
        """
        Bernoulli distribution.
    
        Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args).expand(sample_shape))

    def bernoullilogits(self, name, logits=None, validate_args=None, sample_shape=()):
        """
        BernoulliLogits distribution.
    
        Arguments:
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args).expand(sample_shape))

    def bernoulliprobs(self, name, probs, validate_args=None, sample_shape=()):
        """
        BernoulliProbs distribution.
    
        Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args).expand(sample_shape))

    def beta(self, name, concentration1, concentration0, validate_args=None, sample_shape=()):
        """
        Beta distribution.
    
        Arguments:
            concentration1
            concentration0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(sample_shape))

    def betabinomial(self, name, concentration1, concentration0, total_count=1, validate_args=None, sample_shape=()):
        """
        BetaBinomial distribution.
    
        Arguments:
            concentration1
            concentration0
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args).expand(sample_shape))

    def betaproportion(self, name, mean, concentration, validate_args=None, sample_shape=()):
        """
        BetaProportion distribution.
    
        Arguments:
            mean
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def binomial(self, name, total_count=1, probs=None, logits=None, validate_args=None, sample_shape=()):
        """
        Binomial distribution.
    
        Arguments:
            total_count=1
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args).expand(sample_shape))

    def binomiallogits(self, name, logits, total_count=1, validate_args=None, sample_shape=()):
        """
        BinomialLogits distribution.
    
        Arguments:
            logits
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args).expand(sample_shape))

    def binomialprobs(self, name, probs, total_count=1, validate_args=None, sample_shape=()):
        """
        BinomialProbs distribution.
    
        Arguments:
            probs
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args).expand(sample_shape))

    def car(self, name, loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args).expand(sample_shape))

    def categorical(self, name, probs=None, logits=None, validate_args=None, sample_shape=()):
        """
        Categorical distribution.
    
        Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args).expand(sample_shape))

    def categoricallogits(self, name, logits, validate_args=None, sample_shape=()):
        """
        CategoricalLogits distribution.
    
        Arguments:
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args).expand(sample_shape))

    def categoricalprobs(self, name, probs, validate_args=None, sample_shape=()):
        """
        CategoricalProbs distribution.
    
        Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args).expand(sample_shape))

    def cauchy(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        Cauchy distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def chi2(self, name, df, validate_args=None, sample_shape=()):
        """
        Chi2 distribution.
    
        Arguments:
            df
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Chi2(df=df, validate_args=validate_args).expand(sample_shape))

    def delta(self, name, v=0.0, log_density=0.0, event_dim=0, validate_args=None, sample_shape=()):
        """
        Delta distribution.
    
        Arguments:
            v=0.0
            log_density=0.0
            event_dim=0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args).expand(sample_shape))

    def dirichlet(self, name, concentration, validate_args=None, sample_shape=()):
        """
        Dirichlet distribution.
    
        Arguments:
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def dirichletmultinomial(self, name, concentration, total_count=1, validate_args=None, sample_shape=()):
        """
        DirichletMultinomial distribution.
    
        Arguments:
            concentration
            total_count=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args).expand(sample_shape))

    def discreteuniform(self, name, low=0, high=1, validate_args=None, sample_shape=()):
        """
        DiscreteUniform distribution.
    
        Arguments:
            low=0
            high=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def distribution(self, name, batch_shape=(), event_shape=(), validate_args=None, sample_shape=()):
        """
        Distribution distribution.
    
        Arguments:
            batch_shape=()
            event_shape=()
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Distribution(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).expand(sample_shape))

    def eulermaruyama(self, name, t, sde_fn, init_dist, validate_args=None, sample_shape=()):
        """
        EulerMaruyama distribution.
    
        Arguments:
            t
            sde_fn
            init_dist
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args).expand(sample_shape))

    def expandeddistribution(self, name, base_dist, batch_shape=(), sample_shape=()):
        """
        ExpandedDistribution distribution.
    
        Arguments:
            base_dist
            batch_shape=()
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape).expand(sample_shape))

    def exponential(self, name, rate=1.0, validate_args=None, sample_shape=()):
        """
        Exponential distribution.
    
        Arguments:
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Exponential(rate=rate, validate_args=validate_args).expand(sample_shape))

    def foldeddistribution(self, name, base_dist, validate_args=None, sample_shape=()):
        """
        FoldedDistribution distribution.
    
        Arguments:
            base_dist
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args).expand(sample_shape))

    def gamma(self, name, concentration, rate=1.0, validate_args=None, sample_shape=()):
        """
        Gamma distribution.
    
        Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(sample_shape))

    def gammapoisson(self, name, concentration, rate=1.0, validate_args=None, sample_shape=()):
        """
        GammaPoisson distribution.
    
        Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args).expand(sample_shape))

    def gaussiancopula(self, name, marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, sample_shape=()):
        """
        GaussianCopula distribution.
    
        Arguments:
            marginal_dist
            correlation_matrix=None
            correlation_cholesky=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(sample_shape))

    def gaussiancopulabeta(self, name, concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(sample_shape))

    def gaussianrandomwalk(self, name, scale=1.0, num_steps=1, validate_args=None, sample_shape=()):
        """
        GaussianRandomWalk distribution.
    
        Arguments:
            scale=1.0
            num_steps=1
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args).expand(sample_shape))

    def geometric(self, name, probs=None, logits=None, validate_args=None, sample_shape=()):
        """
        Geometric distribution.
    
        Arguments:
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args).expand(sample_shape))

    def geometriclogits(self, name, logits, validate_args=None, sample_shape=()):
        """
        GeometricLogits distribution.
    
        Arguments:
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args).expand(sample_shape))

    def geometricprobs(self, name, probs, validate_args=None, sample_shape=()):
        """
        GeometricProbs distribution.
    
        Arguments:
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args).expand(sample_shape))

    def gompertz(self, name, concentration, rate=1.0, validate_args=None, sample_shape=()):
        """
        Gompertz distribution.
    
        Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args).expand(sample_shape))

    def gumbel(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        Gumbel distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def halfcauchy(self, name, scale=1.0, validate_args=None, sample_shape=()):
        """
        HalfCauchy distribution.
    
        Arguments:
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args).expand(sample_shape))

    def halfnormal(self, name, scale=1.0, validate_args=None, sample_shape=()):
        """
        HalfNormal distribution.
    
        Arguments:
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args).expand(sample_shape))

    def improperuniform(self, name, support, batch_shape, event_shape, validate_args=None, sample_shape=()):
        """
        ImproperUniform distribution.
    
        Arguments:
            support
            batch_shape
            event_shape
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).expand(sample_shape))

    def independent(self, name, base_dist, reinterpreted_batch_ndims, validate_args=None, sample_shape=()):
        """
        Independent distribution.
    
        Arguments:
            base_dist
            reinterpreted_batch_ndims
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args).expand(sample_shape))

    def inversegamma(self, name, concentration, rate=1.0, validate_args=None, sample_shape=()):
        """
        InverseGamma distribution.
    
        Arguments:
            concentration
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(sample_shape))

    def kumaraswamy(self, name, concentration1, concentration0, validate_args=None, sample_shape=()):
        """
        Kumaraswamy distribution.
    
        Arguments:
            concentration1
            concentration0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(sample_shape))

    def lkj(self, name, dimension, concentration=1.0, sample_method='onion', validate_args=None, sample_shape=()):
        """
        LKJ distribution.
    
        Arguments:
            dimension
            concentration=1.0
            sample_method='onion'
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(sample_shape))

    def lkjcholesky(self, name, dimension, concentration=1.0, sample_method='onion', validate_args=None, sample_shape=()):
        """
        LKJCholesky distribution.
    
        Arguments:
            dimension
            concentration=1.0
            sample_method='onion'
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(sample_shape))

    def laplace(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        Laplace distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def lefttruncateddistribution(self, name, base_dist, low=0.0, validate_args=None, sample_shape=()):
        """
        LeftTruncatedDistribution distribution.
    
        Arguments:
            base_dist
            low=0.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args).expand(sample_shape))

    def lognormal(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        LogNormal distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def loguniform(self, name, low, high, validate_args=None, sample_shape=()):
        """
        LogUniform distribution.
    
        Arguments:
            low
            high
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def logistic(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        Logistic distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def lowrankmultivariatenormal(self, name, loc, cov_factor, cov_diag, validate_args=None, sample_shape=()):
        """
        LowRankMultivariateNormal distribution.
    
        Arguments:
            loc
            cov_factor
            cov_diag
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args).expand(sample_shape))

    def maskeddistribution(self, name, base_dist, mask, sample_shape=()):
        """
        MaskedDistribution distribution.
    
        Arguments:
            base_dist
            mask
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask).expand(sample_shape))

    def matrixnormal(self, name, loc, scale_tril_row, scale_tril_column, validate_args=None, sample_shape=()):
        """
        MatrixNormal distribution.
    
        Arguments:
            loc
            scale_tril_row
            scale_tril_column
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args).expand(sample_shape))

    def mixture(self, name, mixing_distribution, component_distributions, validate_args=None, sample_shape=()):
        """
        Mixture distribution.
    
        Arguments:
            mixing_distribution
            component_distributions
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).expand(sample_shape))

    def mixturegeneral(self, name, mixing_distribution, component_distributions, support=None, validate_args=None, sample_shape=()):
        """
        MixtureGeneral distribution.
    
        Arguments:
            mixing_distribution
            component_distributions
            support=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args).expand(sample_shape))

    def mixturesamefamily(self, name, mixing_distribution, component_distribution, validate_args=None, sample_shape=()):
        """
        MixtureSameFamily distribution.
    
        Arguments:
            mixing_distribution
            component_distribution
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args).expand(sample_shape))

    def multinomial(self, name, total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args).expand(sample_shape))

    def multinomiallogits(self, name, logits, total_count=1, total_count_max=None, validate_args=None, sample_shape=()):
        """
        MultinomialLogits distribution.
    
        Arguments:
            logits
            total_count=1
            total_count_max=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(sample_shape))

    def multinomialprobs(self, name, probs, total_count=1, total_count_max=None, validate_args=None, sample_shape=()):
        """
        MultinomialProbs distribution.
    
        Arguments:
            probs
            total_count=1
            total_count_max=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(sample_shape))

    def multivariatenormal(self, name, loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(sample_shape))

    def multivariatestudentt(self, name, df, loc=0.0, scale_tril=None, validate_args=None, sample_shape=()):
        """
        MultivariateStudentT distribution.
    
        Arguments:
            df
            loc=0.0
            scale_tril=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args).expand(sample_shape))

    def negativebinomial2(self, name, mean, concentration, validate_args=None, sample_shape=()):
        """
        NegativeBinomial2 distribution.
    
        Arguments:
            mean
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def negativebinomiallogits(self, name, total_count, logits, validate_args=None, sample_shape=()):
        """
        NegativeBinomialLogits distribution.
    
        Arguments:
            total_count
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args).expand(sample_shape))

    def negativebinomialprobs(self, name, total_count, probs, validate_args=None, sample_shape=()):
        """
        NegativeBinomialProbs distribution.
    
        Arguments:
            total_count
            probs
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args).expand(sample_shape))

    def normal(self, name, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        Normal distribution.
    
        Arguments:
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def orderedlogistic(self, name, predictor, cutpoints, validate_args=None, sample_shape=()):
        """
        OrderedLogistic distribution.
    
        Arguments:
            predictor
            cutpoints
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args).expand(sample_shape))

    def pareto(self, name, scale, alpha, validate_args=None, sample_shape=()):
        """
        Pareto distribution.
    
        Arguments:
            scale
            alpha
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args).expand(sample_shape))

    def poisson(self, name, rate, is_sparse=False, validate_args=None, sample_shape=()):
        """
        Poisson distribution.
    
        Arguments:
            rate
            is_sparse=False
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args).expand(sample_shape))

    def projectednormal(self, name, concentration, validate_args=None, sample_shape=()):
        """
        ProjectedNormal distribution.
    
        Arguments:
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def relaxedbernoulli(self, name, temperature, probs=None, logits=None, validate_args=None, sample_shape=()):
        """
        RelaxedBernoulli distribution.
    
        Arguments:
            temperature
            probs=None
            logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args).expand(sample_shape))

    def relaxedbernoullilogits(self, name, temperature, logits, validate_args=None, sample_shape=()):
        """
        RelaxedBernoulliLogits distribution.
    
        Arguments:
            temperature
            logits
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args).expand(sample_shape))

    def righttruncateddistribution(self, name, base_dist, high=0.0, validate_args=None, sample_shape=()):
        """
        RightTruncatedDistribution distribution.
    
        Arguments:
            base_dist
            high=0.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args).expand(sample_shape))

    def sinebivariatevonmises(self, name, phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None, weighted_correlation=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args).expand(sample_shape))

    def sineskewed(self, name, base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, sample_shape=()):
        """
        SineSkewed distribution.
    
        Arguments:
            base_dist: numpyro.distributions.distribution.Distribution
            skewness
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args).expand(sample_shape))

    def softlaplace(self, name, loc, scale, validate_args=None, sample_shape=()):
        """
        SoftLaplace distribution.
    
        Arguments:
            loc
            scale
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def studentt(self, name, df, loc=0.0, scale=1.0, validate_args=None, sample_shape=()):
        """
        StudentT distribution.
    
        Arguments:
            df
            loc=0.0
            scale=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args).expand(sample_shape))

    def transformeddistribution(self, name, base_distribution, transforms, validate_args=None, sample_shape=()):
        """
        TransformedDistribution distribution.
    
        Arguments:
            base_distribution
            transforms
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args).expand(sample_shape))

    def truncatedcauchy(self, name, loc=0.0, scale=1.0, low=None, high=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def truncateddistribution(self, name, base_dist, low=None, high=None, validate_args=None, sample_shape=()):
        """
        TruncatedDistribution distribution.
    
        Arguments:
            base_dist
            low=None
            high=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def truncatednormal(self, name, loc=0.0, scale=1.0, low=None, high=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def truncatedpolyagamma(self, name, batch_shape=(), validate_args=None, sample_shape=()):
        """
        TruncatedPolyaGamma distribution.
    
        Arguments:
            batch_shape=()
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args).expand(sample_shape))

    def twosidedtruncateddistribution(self, name, base_dist, low=0.0, high=1.0, validate_args=None, sample_shape=()):
        """
        TwoSidedTruncatedDistribution distribution.
    
        Arguments:
            base_dist
            low=0.0
            high=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def uniform(self, name, low=0.0, high=1.0, validate_args=None, sample_shape=()):
        """
        Uniform distribution.
    
        Arguments:
            low=0.0
            high=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args).expand(sample_shape))

    def unit(self, name, log_factor, validate_args=None, sample_shape=()):
        """
        Unit distribution.
    
        Arguments:
            log_factor
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args).expand(sample_shape))

    def vonmises(self, name, loc, concentration, validate_args=None, sample_shape=()):
        """
        VonMises distribution.
    
        Arguments:
            loc
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def weibull(self, name, scale, concentration, validate_args=None, sample_shape=()):
        """
        Weibull distribution.
    
        Arguments:
            scale
            concentration
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args).expand(sample_shape))

    def wishart(self, name, concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(sample_shape))

    def wishartcholesky(self, name, concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(sample_shape))

    def zeroinflateddistribution(self, name, base_dist, gate=None, gate_logits=None, validate_args=None, sample_shape=()):
        """
        ZeroInflatedDistribution distribution.
    
        Arguments:
            base_dist
            gate=None
            gate_logits=None
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(sample_shape))

    def zeroinflatednegativebinomial2(self, name, mean, concentration, gate=None, gate_logits=None, validate_args=None, sample_shape=()):
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
        return numpyro.sample(name, numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(sample_shape))

    def zeroinflatedpoisson(self, name, gate, rate=1.0, validate_args=None, sample_shape=()):
        """
        ZeroInflatedPoisson distribution.
    
        Arguments:
            gate
            rate=1.0
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args).expand(sample_shape))

    def zerosumnormal(self, name, scale, event_shape, validate_args=None, sample_shape=()):
        """
        ZeroSumNormal distribution.
    
        Arguments:
            scale
            event_shape
            validate_args=None
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args).expand(sample_shape))

    def kl_divergence(self, name, sample_shape=(), *args, **kwargs):
        """
        kl_divergence distribution.
    
        Arguments:
            *args
            **kwargs
            sample_shape: Shape of samples to be drawn.
        """
        return numpyro.sample(name, numpyro.distributions.kl_divergence(args=args, kwargs=kwargs).expand(sample_shape))

