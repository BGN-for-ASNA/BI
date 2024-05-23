import numpyro as numpyro
from numpyro.distributions import*
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import sample 

def asymmetriclaplace(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.AsymmetricLaplace(*args, **kwargs).expand(sample_shape))
def asymmetriclaplacequantile(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.AsymmetricLaplaceQuantile(*args, **kwargs).expand(sample_shape))
def bernoulli(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Bernoulli(*args, **kwargs).expand(sample_shape))
def bernoullilogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BernoulliLogits(*args, **kwargs).expand(sample_shape))
def bernoulliprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BernoulliProbs(*args, **kwargs).expand(sample_shape))
def beta(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Beta(*args, **kwargs).expand(sample_shape))
def betabinomial(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BetaBinomial(*args, **kwargs).expand(sample_shape))
def betaproportion(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BetaProportion(*args, **kwargs).expand(sample_shape))
def binomial(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Binomial(*args, **kwargs).expand(sample_shape))
def binomiallogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BinomialLogits(*args, **kwargs).expand(sample_shape))
def binomialprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.BinomialProbs(*args, **kwargs).expand(sample_shape))
def car(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.CAR(*args, **kwargs).expand(sample_shape))
def categorical(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Categorical(*args, **kwargs).expand(sample_shape))
def categoricallogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.CategoricalLogits(*args, **kwargs).expand(sample_shape))
def categoricalprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.CategoricalProbs(*args, **kwargs).expand(sample_shape))
def cauchy(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Cauchy(*args, **kwargs).expand(sample_shape))
def chi2(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Chi2(*args, **kwargs).expand(sample_shape))
def delta(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Delta(*args, **kwargs).expand(sample_shape))
def dirichlet(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Dirichlet(*args, **kwargs).expand(sample_shape))
def dirichletmultinomial(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.DirichletMultinomial(*args, **kwargs).expand(sample_shape))
def discreteuniform(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.DiscreteUniform(*args, **kwargs).expand(sample_shape))
def distribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Distribution(*args, **kwargs).expand(sample_shape))
def eulermaruyama(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.EulerMaruyama(*args, **kwargs).expand(sample_shape))
def expandeddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ExpandedDistribution(*args, **kwargs).expand(sample_shape))
def exponential(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Exponential(*args, **kwargs).expand(sample_shape))
def foldeddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.FoldedDistribution(*args, **kwargs).expand(sample_shape))
def gamma(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Gamma(*args, **kwargs).expand(sample_shape))
def gammapoisson(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GammaPoisson(*args, **kwargs).expand(sample_shape))
def gaussiancopula(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GaussianCopula(*args, **kwargs).expand(sample_shape))
def gaussiancopulabeta(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GaussianCopulaBeta(*args, **kwargs).expand(sample_shape))
def gaussianrandomwalk(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GaussianRandomWalk(*args, **kwargs).expand(sample_shape))
def geometric(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Geometric(*args, **kwargs).expand(sample_shape))
def geometriclogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GeometricLogits(*args, **kwargs).expand(sample_shape))
def geometricprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.GeometricProbs(*args, **kwargs).expand(sample_shape))
def gompertz(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Gompertz(*args, **kwargs).expand(sample_shape))
def gumbel(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Gumbel(*args, **kwargs).expand(sample_shape))
def halfcauchy(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.HalfCauchy(*args, **kwargs).expand(sample_shape))
def halfnormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.HalfNormal(*args, **kwargs).expand(sample_shape))
def improperuniform(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ImproperUniform(*args, **kwargs).expand(sample_shape))
def independent(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Independent(*args, **kwargs).expand(sample_shape))
def inversegamma(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.InverseGamma(*args, **kwargs).expand(sample_shape))
def kumaraswamy(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Kumaraswamy(*args, **kwargs).expand(sample_shape))
def lkj(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LKJ(*args, **kwargs).expand(sample_shape))
def lkjcholesky(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LKJCholesky(*args, **kwargs).expand(sample_shape))
def laplace(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Laplace(*args, **kwargs).expand(sample_shape))
def lefttruncateddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LeftTruncatedDistribution(*args, **kwargs).expand(sample_shape))
def lognormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LogNormal(*args, **kwargs).expand(sample_shape))
def loguniform(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LogUniform(*args, **kwargs).expand(sample_shape))
def logistic(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Logistic(*args, **kwargs).expand(sample_shape))
def lowrankmultivariatenormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.LowRankMultivariateNormal(*args, **kwargs).expand(sample_shape))
def maskeddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MaskedDistribution(*args, **kwargs).expand(sample_shape))
def matrixnormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MatrixNormal(*args, **kwargs).expand(sample_shape))
def mixture(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Mixture(*args, **kwargs).expand(sample_shape))
def mixturegeneral(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MixtureGeneral(*args, **kwargs).expand(sample_shape))
def mixturesamefamily(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MixtureSameFamily(*args, **kwargs).expand(sample_shape))
def multinomial(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Multinomial(*args, **kwargs).expand(sample_shape))
def multinomiallogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MultinomialLogits(*args, **kwargs).expand(sample_shape))
def multinomialprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MultinomialProbs(*args, **kwargs).expand(sample_shape))
def multivariatenormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MultivariateNormal(*args, **kwargs).expand(sample_shape))
def multivariatestudentt(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.MultivariateStudentT(*args, **kwargs).expand(sample_shape))
def negativebinomial2(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.NegativeBinomial2(*args, **kwargs).expand(sample_shape))
def negativebinomiallogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.NegativeBinomialLogits(*args, **kwargs).expand(sample_shape))
def negativebinomialprobs(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.NegativeBinomialProbs(*args, **kwargs).expand(sample_shape))
def normal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Normal(*args, **kwargs).expand(sample_shape))
def orderedlogistic(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.OrderedLogistic(*args, **kwargs).expand(sample_shape))
def pareto(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Pareto(*args, **kwargs).expand(sample_shape))
def poisson(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Poisson(*args, **kwargs).expand(sample_shape))
def projectednormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ProjectedNormal(*args, **kwargs).expand(sample_shape))
def relaxedbernoulli(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.RelaxedBernoulli(*args, **kwargs).expand(sample_shape))
def relaxedbernoullilogits(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.RelaxedBernoulliLogits(*args, **kwargs).expand(sample_shape))
def righttruncateddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.RightTruncatedDistribution(*args, **kwargs).expand(sample_shape))
def sinebivariatevonmises(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.SineBivariateVonMises(*args, **kwargs).expand(sample_shape))
def sineskewed(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.SineSkewed(*args, **kwargs).expand(sample_shape))
def softlaplace(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.SoftLaplace(*args, **kwargs).expand(sample_shape))
def studentt(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.StudentT(*args, **kwargs).expand(sample_shape))
def transformeddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TransformedDistribution(*args, **kwargs).expand(sample_shape))
def truncatedcauchy(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TruncatedCauchy(*args, **kwargs).expand(sample_shape))
def truncateddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TruncatedDistribution(*args, **kwargs).expand(sample_shape))
def truncatednormal(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TruncatedNormal(*args, **kwargs).expand(sample_shape))
def truncatedpolyagamma(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TruncatedPolyaGamma(*args, **kwargs).expand(sample_shape))
def twosidedtruncateddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.TwoSidedTruncatedDistribution(*args, **kwargs).expand(sample_shape))
def uniform(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Uniform(*args, **kwargs).expand(sample_shape))
def unit(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Unit(*args, **kwargs).expand(sample_shape))
def vonmises(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.VonMises(*args, **kwargs).expand(sample_shape))
def weibull(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.Weibull(*args, **kwargs).expand(sample_shape))
def zeroinflateddistribution(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ZeroInflatedDistribution(*args, **kwargs).expand(sample_shape))
def zeroinflatednegativebinomial2(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ZeroInflatedNegativeBinomial2(*args, **kwargs).expand(sample_shape))
def zeroinflatedpoisson(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.ZeroInflatedPoisson(*args, **kwargs).expand(sample_shape))
def kl_divergence(name, sample_shape=(), *args, **kwargs):
    return numpyro.sample(name, numpyro.distributions.kl_divergence(*args, **kwargs).expand(sample_shape))
