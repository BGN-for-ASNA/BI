
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import*
root = tfd.JointDistributionCoroutine.Root
import random as r
from jax import*
import jax.numpy as jnp

def autocompositetensordistribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(AutoCompositeTensorDistribution(*args, **kwargs), sample_shape))
def autoregressive(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Autoregressive(*args, **kwargs), sample_shape))
def batchbroadcast(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(BatchBroadcast(*args, **kwargs), sample_shape))
def batchconcat(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(BatchConcat(*args, **kwargs), sample_shape))
def batchreshape(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(BatchReshape(*args, **kwargs), sample_shape))
def bates(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Bates(*args, **kwargs), sample_shape))
def bernoulli(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Bernoulli(*args, **kwargs), sample_shape))
def beta(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Beta(*args, **kwargs), sample_shape))
def betabinomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(BetaBinomial(*args, **kwargs), sample_shape))
def betaquotient(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(BetaQuotient(*args, **kwargs), sample_shape))
def binomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Binomial(*args, **kwargs), sample_shape))
def blockwise(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Blockwise(*args, **kwargs), sample_shape))
def categorical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Categorical(*args, **kwargs), sample_shape))
def cauchy(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Cauchy(*args, **kwargs), sample_shape))
def chi(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Chi(*args, **kwargs), sample_shape))
def chi2(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Chi2(*args, **kwargs), sample_shape))
def choleskylkj(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(CholeskyLKJ(*args, **kwargs), sample_shape))
def continuousbernoulli(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ContinuousBernoulli(*args, **kwargs), sample_shape))
def determinantalpointprocess(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(DeterminantalPointProcess(*args, **kwargs), sample_shape))
def deterministic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Deterministic(*args, **kwargs), sample_shape))
def dirichlet(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Dirichlet(*args, **kwargs), sample_shape))
def dirichletmultinomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(DirichletMultinomial(*args, **kwargs), sample_shape))
def distribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Distribution(*args, **kwargs), sample_shape))
def doublesidedmaxwell(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(DoublesidedMaxwell(*args, **kwargs), sample_shape))
def empirical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Empirical(*args, **kwargs), sample_shape))
def expgamma(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ExpGamma(*args, **kwargs), sample_shape))
def expinversegamma(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ExpInverseGamma(*args, **kwargs), sample_shape))
def exprelaxedonehotcategorical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ExpRelaxedOneHotCategorical(*args, **kwargs), sample_shape))
def exponential(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Exponential(*args, **kwargs), sample_shape))
def exponentiallymodifiedgaussian(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ExponentiallyModifiedGaussian(*args, **kwargs), sample_shape))
def finitediscrete(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(FiniteDiscrete(*args, **kwargs), sample_shape))
def gamma(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Gamma(*args, **kwargs), sample_shape))
def gammagamma(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GammaGamma(*args, **kwargs), sample_shape))
def gaussianprocess(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GaussianProcess(*args, **kwargs), sample_shape))
def gaussianprocessregressionmodel(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GaussianProcessRegressionModel(*args, **kwargs), sample_shape))
def generalizedextremevalue(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GeneralizedExtremeValue(*args, **kwargs), sample_shape))
def generalizednormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GeneralizedNormal(*args, **kwargs), sample_shape))
def generalizedpareto(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(GeneralizedPareto(*args, **kwargs), sample_shape))
def geometric(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Geometric(*args, **kwargs), sample_shape))
def gumbel(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Gumbel(*args, **kwargs), sample_shape))
def halfcauchy(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(HalfCauchy(*args, **kwargs), sample_shape))
def halfnormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(HalfNormal(*args, **kwargs), sample_shape))
def halfstudentt(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(HalfStudentT(*args, **kwargs), sample_shape))
def hiddenmarkovmodel(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(HiddenMarkovModel(*args, **kwargs), sample_shape))
def horseshoe(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Horseshoe(*args, **kwargs), sample_shape))
def independent(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Independent(*args, **kwargs), sample_shape))
def inflated(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Inflated(*args, **kwargs), sample_shape))
def inversegamma(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(InverseGamma(*args, **kwargs), sample_shape))
def inversegaussian(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(InverseGaussian(*args, **kwargs), sample_shape))
def johnsonsu(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JohnsonSU(*args, **kwargs), sample_shape))
def jointdistribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistribution(*args, **kwargs), sample_shape))
def jointdistributioncoroutine(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionCoroutine(*args, **kwargs), sample_shape))
def jointdistributioncoroutineautobatched(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionCoroutineAutoBatched(*args, **kwargs), sample_shape))
def jointdistributionnamed(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionNamed(*args, **kwargs), sample_shape))
def jointdistributionnamedautobatched(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionNamedAutoBatched(*args, **kwargs), sample_shape))
def jointdistributionsequential(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionSequential(*args, **kwargs), sample_shape))
def jointdistributionsequentialautobatched(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(JointDistributionSequentialAutoBatched(*args, **kwargs), sample_shape))
def kumaraswamy(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Kumaraswamy(*args, **kwargs), sample_shape))
def lkj(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LKJ(*args, **kwargs), sample_shape))
def lambertwdistribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LambertWDistribution(*args, **kwargs), sample_shape))
def lambertwnormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LambertWNormal(*args, **kwargs), sample_shape))
def laplace(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Laplace(*args, **kwargs), sample_shape))
def lineargaussianstatespacemodel(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LinearGaussianStateSpaceModel(*args, **kwargs), sample_shape))
def loglogistic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LogLogistic(*args, **kwargs), sample_shape))
def lognormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LogNormal(*args, **kwargs), sample_shape))
def logistic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Logistic(*args, **kwargs), sample_shape))
def logitnormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(LogitNormal(*args, **kwargs), sample_shape))
def markovchain(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MarkovChain(*args, **kwargs), sample_shape))
def masked(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Masked(*args, **kwargs), sample_shape))
def matrixnormallinearoperator(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MatrixNormalLinearOperator(*args, **kwargs), sample_shape))
def matrixtlinearoperator(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MatrixTLinearOperator(*args, **kwargs), sample_shape))
def mixture(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Mixture(*args, **kwargs), sample_shape))
def mixturesamefamily(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MixtureSameFamily(*args, **kwargs), sample_shape))
def moyal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Moyal(*args, **kwargs), sample_shape))
def multinomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Multinomial(*args, **kwargs), sample_shape))
def multivariatenormaldiag(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalDiag(*args, **kwargs), sample_shape))
def multivariatenormaldiagpluslowrank(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalDiagPlusLowRank(*args, **kwargs), sample_shape))
def multivariatenormaldiagpluslowrankcovariance(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalDiagPlusLowRankCovariance(*args, **kwargs), sample_shape))
def multivariatenormalfullcovariance(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalFullCovariance(*args, **kwargs), sample_shape))
def multivariatenormallinearoperator(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalLinearOperator(*args, **kwargs), sample_shape))
def multivariatenormaltril(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateNormalTriL(*args, **kwargs), sample_shape))
def multivariatestudenttlinearoperator(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(MultivariateStudentTLinearOperator(*args, **kwargs), sample_shape))
def negativebinomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(NegativeBinomial(*args, **kwargs), sample_shape))
def noncentralchi2(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(NoncentralChi2(*args, **kwargs), sample_shape))
def normal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Normal(*args, **kwargs), sample_shape))
def normalinversegaussian(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(NormalInverseGaussian(*args, **kwargs), sample_shape))
def onehotcategorical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(OneHotCategorical(*args, **kwargs), sample_shape))
def orderedlogistic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(OrderedLogistic(*args, **kwargs), sample_shape))
def pert(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(PERT(*args, **kwargs), sample_shape))
def pareto(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Pareto(*args, **kwargs), sample_shape))
def plackettluce(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(PlackettLuce(*args, **kwargs), sample_shape))
def poisson(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Poisson(*args, **kwargs), sample_shape))
def poissonlognormalquadraturecompound(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(PoissonLogNormalQuadratureCompound(*args, **kwargs), sample_shape))
def powerspherical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(PowerSpherical(*args, **kwargs), sample_shape))
def probitbernoulli(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ProbitBernoulli(*args, **kwargs), sample_shape))
def quantizeddistribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(QuantizedDistribution(*args, **kwargs), sample_shape))
def registerkl(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(RegisterKL(*args, **kwargs), sample_shape))
def relaxedbernoulli(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(RelaxedBernoulli(*args, **kwargs), sample_shape))
def relaxedonehotcategorical(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(RelaxedOneHotCategorical(*args, **kwargs), sample_shape))
def reparameterizationtype(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ReparameterizationType(*args, **kwargs), sample_shape))
def sample(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Sample(*args, **kwargs), sample_shape))
def sigmoidbeta(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(SigmoidBeta(*args, **kwargs), sample_shape))
def sinharcsinh(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(SinhArcsinh(*args, **kwargs), sample_shape))
def skellam(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Skellam(*args, **kwargs), sample_shape))
def sphericaluniform(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(SphericalUniform(*args, **kwargs), sample_shape))
def stoppingratiologistic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(StoppingRatioLogistic(*args, **kwargs), sample_shape))
def studentt(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(StudentT(*args, **kwargs), sample_shape))
def studenttprocess(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(StudentTProcess(*args, **kwargs), sample_shape))
def studenttprocessregressionmodel(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(StudentTProcessRegressionModel(*args, **kwargs), sample_shape))
def transformeddistribution(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(TransformedDistribution(*args, **kwargs), sample_shape))
def triangular(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Triangular(*args, **kwargs), sample_shape))
def truncatedcauchy(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(TruncatedCauchy(*args, **kwargs), sample_shape))
def truncatednormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(TruncatedNormal(*args, **kwargs), sample_shape))
def twopiecenormal(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(TwoPieceNormal(*args, **kwargs), sample_shape))
def twopiecestudentt(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(TwoPieceStudentT(*args, **kwargs), sample_shape))
def uniform(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Uniform(*args, **kwargs), sample_shape))
def variationalgaussianprocess(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(VariationalGaussianProcess(*args, **kwargs), sample_shape))
def vectordeterministic(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(VectorDeterministic(*args, **kwargs), sample_shape))
def vonmises(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(VonMises(*args, **kwargs), sample_shape))
def vonmisesfisher(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(VonMisesFisher(*args, **kwargs), sample_shape))
def weibull(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Weibull(*args, **kwargs), sample_shape))
def wishartlinearoperator(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(WishartLinearOperator(*args, **kwargs), sample_shape))
def wisharttril(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(WishartTriL(*args, **kwargs), sample_shape))
def zeroinflatednegativebinomial(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(ZeroInflatedNegativeBinomial(*args, **kwargs), sample_shape))
def zipf(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(Zipf(*args, **kwargs), sample_shape))
def independent_joint_distribution_from_structure(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(independent_joint_distribution_from_structure(*args, **kwargs), sample_shape))
def kl_divergence(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(kl_divergence(*args, **kwargs), sample_shape))
def mvn_conjugate_linear_update(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(mvn_conjugate_linear_update(*args, **kwargs), sample_shape))
def normal_conjugates_known_scale_posterior(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(normal_conjugates_known_scale_posterior(*args, **kwargs), sample_shape))
def normal_conjugates_known_scale_predictive(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(normal_conjugates_known_scale_predictive(*args, **kwargs), sample_shape))
def quadrature_scheme_lognormal_gauss_hermite(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(quadrature_scheme_lognormal_gauss_hermite(*args, **kwargs), sample_shape))
def quadrature_scheme_lognormal_quantiles(sample_shape=(), *args, **kwargs):
    return root(tfd.Sample(quadrature_scheme_lognormal_quantiles(*args, **kwargs), sample_shape))
