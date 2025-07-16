"""
This test file checks the sampling functionality of all distributions
defined in the UnifiedDist class.

To run this file, ensure that:
1. The `UnifiedDist` class is saved in a file named `unified_dist.py`
   in the same directory.
2. You have the necessary libraries installed (jax, numpy, numpyro).

The script will iterate through each distribution, attempt to draw a sample,
and report whether the operation was successful.
"""
#%%
import os
import sys
newPath = os.path.dirname(os.path.abspath(""))
if newPath not in sys.path:
    sys.path.append(newPath)
from BI import bi


UnifiedDist
def run_tests():
    """
    Tests the sampling functionality of all distributions in the UnifiedDist class.
    """
    print("--- Starting Distribution Sampling Tests ---")
    
    # List of distributions that cannot be sampled from directly
    # or are helper functions/base classes.
    untestable_samplers = [
        'distribution', 'improperuniform', 'truncatedpolyagamma',
        'unit', 'biject_to', 'kl_divergence'
    ]
    
    # Get all static methods from the class
    all_distributions = [
        func for func in dir(UnifiedDist) 
        if callable(getattr(UnifiedDist, func)) and not func.startswith("__")
    ]

    for dist_name in sorted(all_distributions):
        print(f"\n--- Testing: {dist_name} ---")

        if dist_name in untestable_samplers:
            print(f"Result: SKIPPED (Distribution cannot be sampled directly)")
            continue

        try:
            # --- Call the appropriate test function ---
            # This uses getattr to find the function in this script
            # with the name `test_{dist_name}` and executes it.
            test_function = globals()[f'test_{dist_name}']
            sample = test_function()
            # Truncate sample for cleaner printing
            sample_str = np.array2string(np.array(sample), max_line_width=100, precision=3)
            print(f"Result: SUCCESS")
            print(f"Sample: {sample_str}")

        except Exception as e:
            print(f"Result: FAILED")
            print(f"Error: {e}")

# --- Test function definitions for each distribution ---

def test_asymmetriclaplace():
    return UnifiedDist.asymmetriclaplace(loc=0., scale=1., asymmetry=0.7, sample=True, shape=(3,))

def test_asymmetriclaplacequantile():
    return UnifiedDist.asymmetriclaplacequantile(loc=0., scale=1., quantile=0.7, sample=True, shape=(3,))

def test_bernoulli():
    return UnifiedDist.bernoulli(probs=0.5, sample=True, shape=(5,))

def test_bernoullilogits():
    return UnifiedDist.bernoullilogits(logits=0.0, sample=True, shape=(5,))

def test_bernoulliprobs():
    return UnifiedDist.bernoulliprobs(probs=0.5, sample=True, shape=(5,))

def test_beta():
    return UnifiedDist.beta(concentration1=2.0, concentration0=2.0, sample=True, shape=(3,))

def test_betabinomial():
    return UnifiedDist.betabinomial(concentration1=2.0, concentration0=2.0, total_count=10, sample=True, shape=(3,))

def test_betaproportion():
    return UnifiedDist.betaproportion(mean=0.5, concentration=10.0, sample=True, shape=(3,))

def test_binomial():
    return UnifiedDist.binomial(total_count=20, probs=0.5, sample=True, shape=(3,))

def test_binomiallogits():
    return UnifiedDist.binomiallogits(total_count=20, logits=0.0, sample=True, shape=(3,))

def test_binomialprobs():
    return UnifiedDist.binomialprobs(total_count=20, probs=0.5, sample=True, shape=(3,))

def test_car():
    adj = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    loc = jnp.zeros(3)
    return UnifiedDist.car(loc=loc, correlation=0.9, conditional_precision=1.0, adj_matrix=adj, sample=True, shape=())

def test_categorical():
    return UnifiedDist.categorical(probs=jnp.array([0.1, 0.5, 0.4]), sample=True, shape=(5,))

def test_categoricallogits():
    return UnifiedDist.categoricallogits(logits=jnp.array([0.1, 0.2, 0.3]), sample=True, shape=(5,))

def test_categoricalprobs():
    return UnifiedDist.categoricalprobs(probs=jnp.array([0.1, 0.5, 0.4]), sample=True, shape=(5,))

def test_cauchy():
    return UnifiedDist.cauchy(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_chi2():
    return UnifiedDist.chi2(df=5.0, sample=True, shape=(3,))

def test_circulantnormal():
    loc = jnp.zeros(3)
    cov_row = jnp.array([1.0, 0.5, 0.25])
    return UnifiedDist.circulantnormal(loc=loc, covariance_row=cov_row, sample=True, shape=())

def test_delta():
    return UnifiedDist.delta(v=5.0, sample=True, shape=(3,))

def test_dirichlet():
    return UnifiedDist.dirichlet(concentration=jnp.array([1.0, 2.0, 3.0]), sample=True, shape=(2,))

def test_dirichletmultinomial():
    return UnifiedDist.dirichletmultinomial(concentration=jnp.array([1.0, 2.0]), total_count=10, sample=True, shape=(3,))

def test_discreteuniform():
    return UnifiedDist.discreteuniform(low=0, high=10, sample=True, shape=(5,))

def test_doublytruncatedpowerlaw():
    return UnifiedDist.doublytruncatedpowerlaw(alpha=2.5, low=1.0, high=10.0, sample=True, shape=(3,))

def test_eulermaruyama():
    def sde_fn(x): return 0.0, 1.0
    init_dist = UnifiedDist.normal(0.0, 1.0, create_obj=True)
    t = jnp.linspace(0, 1, 5)
    return UnifiedDist.eulermaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, sample=True)

def test_expandeddistribution():
    base = UnifiedDist.normal(loc=0, scale=1, create_obj=True)
    return UnifiedDist.expandeddistribution(base_dist=base, batch_shape=(3, 2), sample=True)

def test_exponential():
    return UnifiedDist.exponential(rate=2.0, sample=True, shape=(3,))

def test_foldeddistribution():
    base = UnifiedDist.normal(loc=0, scale=1, create_obj=True)
    return UnifiedDist.foldeddistribution(base_dist=base, sample=True, shape=(3,))

def test_gamma():
    return UnifiedDist.gamma(concentration=2.0, rate=1.0, sample=True, shape=(3,))

def test_gammapoisson():
    return UnifiedDist.gammapoisson(concentration=2.0, rate=1.0, sample=True, shape=(3,))
    
def test_gaussiancopula():
    marginal_dists = [
        UnifiedDist.normal(0, 1, create_obj=True),
        UnifiedDist.beta(2, 2, create_obj=True)
    ]
    corr_matrix = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    return UnifiedDist.gaussiancopula(marginal_dist=marginal_dists, correlation_matrix=corr_matrix, sample=True)

def test_gaussiancopulabeta():
    corr_matrix = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    c1 = jnp.array([2., 3.])
    c0 = jnp.array([2., 3.])
    return UnifiedDist.gaussiancopulabeta(concentration1=c1, concentration0=c0, correlation_matrix=corr_matrix, sample=True)

def test_gaussianrandomwalk():
    return UnifiedDist.gaussianrandomwalk(scale=1.0, num_steps=5, sample=True, shape=(2,))

def test_gaussianstatespace():
    trans_matrix = jnp.eye(2) * 0.5
    cov_matrix = jnp.eye(2)
    return UnifiedDist.gaussianstatespace(num_steps=5, transition_matrix=trans_matrix, covariance_matrix=cov_matrix, sample=True)
    
def test_geometric():
    return UnifiedDist.geometric(probs=0.3, sample=True, shape=(5,))

def test_geometriclogits():
    return UnifiedDist.geometriclogits(logits=-0.84, sample=True, shape=(5,))

def test_geometricprobs():
    return UnifiedDist.geometricprobs(probs=0.3, sample=True, shape=(5,))
    
def test_gompertz():
    return UnifiedDist.gompertz(concentration=2.0, rate=1.0, sample=True, shape=(3,))

def test_gumbel():
    return UnifiedDist.gumbel(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_halfcauchy():
    return UnifiedDist.halfcauchy(scale=1.0, sample=True, shape=(3,))

def test_halfnormal():
    return UnifiedDist.halfnormal(scale=1.0, sample=True, shape=(3,))

def test_independent():
    base = UnifiedDist.normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3)), create_obj=True)
    return UnifiedDist.independent(base_dist=base, reinterpreted_batch_ndims=1, sample=True)

def test_inversegamma():
    return UnifiedDist.inversegamma(concentration=2.0, rate=1.0, sample=True, shape=(3,))

def test_kumaraswamy():
    return UnifiedDist.kumaraswamy(concentration1=2.0, concentration0=5.0, sample=True, shape=(3,))

def test_lkj():
    return UnifiedDist.lkj(dimension=3, concentration=1.0, sample=True, shape=())
    
def test_lkjcholesky():
    return UnifiedDist.lkjcholesky(dimension=3, concentration=1.0, sample=True, shape=())

def test_laplace():
    return UnifiedDist.laplace(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_lefttruncateddistribution():
    base = UnifiedDist.normal(0, 1, create_obj=True)
    return UnifiedDist.lefttruncateddistribution(base, low=0.5, sample=True)

def test_levy():
    return UnifiedDist.levy(loc=0., scale=1., sample=True)

def test_lognormal():
    return UnifiedDist.lognormal(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_loguniform():
    return UnifiedDist.loguniform(low=0.1, high=10.0, sample=True, shape=(3,))
    
def test_logistic():
    return UnifiedDist.logistic(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_lowrankmultivariatenormal():
    loc = jnp.zeros(5)
    cov_factor = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
    cov_diag = jnp.ones(5)
    return UnifiedDist.lowrankmultivariatenormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, sample=True, shape=())
    
def test_lowertruncatedpowerlaw():
    return UnifiedDist.lowertruncatedpowerlaw(alpha=2.5, low=1.0, sample=True)

def test_maskeddistribution():
    base = UnifiedDist.normal(jnp.zeros(3), jnp.ones(3), create_obj=True)
    mask = jnp.array([True, False, True])
    return UnifiedDist.maskeddistribution(base_dist=base, mask=mask, sample=True)

def test_matrixnormal():
    loc = jnp.zeros((3, 2))
    row_cov_chol = jnp.eye(3)
    col_cov_chol = jnp.eye(2)
    return UnifiedDist.matrixnormal(loc=loc, scale_tril_row=row_cov_chol, scale_tril_column=col_cov_chol, sample=True)

def test_mixture():
    mix_dist = UnifiedDist.categorical(probs=jnp.array([0.5, 0.5]), create_obj=True)
    comp_dists = [
        UnifiedDist.normal(-1, 0.5, create_obj=True),
        UnifiedDist.normal(1, 0.5, create_obj=True)
    ]
    return UnifiedDist.mixture(mixing_distribution=mix_dist, component_distributions=comp_dists, sample=True, shape=(5,))
    
def test_mixturegeneral():
    mix_dist = UnifiedDist.categorical(probs=jnp.array([0.5, 0.5]), create_obj=True)
    comp_dists = [
        UnifiedDist.normal(-1, 0.5, create_obj=True),
        UnifiedDist.normal(1, 0.5, create_obj=True)
    ]
    return UnifiedDist.mixturegeneral(mixing_distribution=mix_dist, component_distributions=comp_dists, sample=True, shape=(5,))

def test_mixturesamefamily():
    mix_dist = UnifiedDist.categorical(probs=jnp.array([0.5, 0.5]), create_obj=True)
    comp_dist = UnifiedDist.normal(loc=jnp.array([-1., 1.]), scale=jnp.array([0.5, 0.5]), create_obj=True)
    return UnifiedDist.mixturesamefamily(mixing_distribution=mix_dist, component_distribution=comp_dist, sample=True, shape=(5,))

def test_multinomial():
    return UnifiedDist.multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]), sample=True, shape=(4,))

def test_multinomiallogits():
    return UnifiedDist.multinomiallogits(total_count=10, logits=jnp.array([-1., 0., 1.]), sample=True, shape=(4,))

def test_multinomialprobs():
    return UnifiedDist.multinomialprobs(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]), sample=True, shape=(4,))

def test_multivariatenormal():
    loc = jnp.zeros(2)
    scale_tril = jnp.array([[1.0, 0.0], [0.5, 0.8]])
    return UnifiedDist.multivariatenormal(loc=loc, scale_tril=scale_tril, sample=True, shape=(3,))
    
def test_multivariatestudentt():
    loc = jnp.zeros(2)
    scale_tril = jnp.array([[1.0, 0.0], [0.5, 0.8]])
    return UnifiedDist.multivariatestudentt(df=5., loc=loc, scale_tril=scale_tril, sample=True)

def test_negativebinomial2():
    return UnifiedDist.negativebinomial2(mean=10.0, concentration=5.0, sample=True, shape=(3,))
    
def test_negativebinomiallogits():
    return UnifiedDist.negativebinomiallogits(total_count=10, logits=0.5, sample=True, shape=(3,))
    
def test_negativebinomialprobs():
    return UnifiedDist.negativebinomialprobs(total_count=10, probs=0.7, sample=True, shape=(3,))
    
def test_normal():
    return UnifiedDist.normal(loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_orderedlogistic():
    predictor = jnp.array([0.5, 1.5, -1.0])
    cutpoints = jnp.array([0.0, 2.0])
    return UnifiedDist.orderedlogistic(predictor=predictor, cutpoints=cutpoints, sample=True)

def test_pareto():
    return UnifiedDist.pareto(scale=1.0, alpha=2.5, sample=True, shape=(3,))
    
def test_poisson():
    return UnifiedDist.poisson(rate=5.0, sample=True, shape=(3,))
    
def test_projectednormal():
    return UnifiedDist.projectednormal(concentration=jnp.array([0.5, 0.5]), sample=True, shape=(3,))
    
def test_relaxedbernoulli():
    return UnifiedDist.relaxedbernoulli(temperature=0.5, probs=0.5, sample=True, shape=(3,))

def test_relaxedbernoullilogits():
    return UnifiedDist.relaxedbernoullilogits(temperature=0.5, logits=0.0, sample=True, shape=(3,))

def test_righttruncateddistribution():
    base = UnifiedDist.normal(0, 1, create_obj=True)
    return UnifiedDist.righttruncateddistribution(base_dist=base, high=0.5, sample=True)

def test_sinebivariatevonmises():
    return UnifiedDist.sinebivariatevonmises(
        phi_loc=0., psi_loc=0., phi_concentration=1.,
        psi_concentration=1., correlation=0.5, sample=True
    )

def test_sineskewed():
    base_dist = UnifiedDist.vonmises(0, 1, create_obj=True)
    return UnifiedDist.sineskewed(base_dist, skewness=0.5, sample=True)

def test_softlaplace():
    return UnifiedDist.softlaplace(loc=0., scale=1., sample=True)

def test_studentt():
    return UnifiedDist.studentt(df=5.0, loc=0.0, scale=1.0, sample=True, shape=(3,))

def test_transformeddistribution():
    base_dist = UnifiedDist.normal(0, 1, create_obj=True)
    transform = numpyro.distributions.transforms.ExpTransform()
    return UnifiedDist.transformeddistribution(base_distribution=base_dist, transforms=transform, sample=True)

def test_truncatedcauchy():
    return UnifiedDist.truncatedcauchy(loc=0., scale=1., low=-1., high=1., sample=True)
    
def test_truncateddistribution():
    base = UnifiedDist.normal(0, 1, create_obj=True)
    return UnifiedDist.truncateddistribution(base, low=-1., high=1., sample=True)

def test_truncatednormal():
    return UnifiedDist.truncatednormal(loc=0., scale=1., low=-1., high=1., sample=True)

def test_twosidedtruncateddistribution():
    base = UnifiedDist.normal(0, 1, create_obj=True)
    return UnifiedDist.twosidedtruncateddistribution(base, low=-1., high=1., sample=True)

def test_uniform():
    return UnifiedDist.uniform(low=0.0, high=1.0, sample=True, shape=(3,))
    
def test_vonmises():
    return UnifiedDist.vonmises(loc=0.0, concentration=2.0, sample=True, shape=(3,))

def test_weibull():
    return UnifiedDist.weibull(scale=1.0, concentration=2.0, sample=True, shape=(3,))
    
def test_wishart():
    scale_matrix = jnp.eye(3)
    return UnifiedDist.wishart(concentration=5., scale_matrix=scale_matrix, sample=True)

def test_wishartcholesky():
    scale_tril = jnp.eye(3)
    return UnifiedDist.wishartcholesky(concentration=5., scale_tril=scale_tril, sample=True)

def test_zeroinflateddistribution():
    base = UnifiedDist.poisson(rate=5, create_obj=True)
    return UnifiedDist.zeroinflateddistribution(base_dist=base, gate=0.2, sample=True)

def test_zeroinflatednegativebinomial2():
    return UnifiedDist.zeroinflatednegativebinomial2(mean=10, concentration=5, gate=0.2, sample=True)

def test_zeroinflatedpoisson():
    return UnifiedDist.zeroinflatedpoisson(gate=0.2, rate=5, sample=True)
    
def test_zerosumnormal():
    return UnifiedDist.zerosumnormal(scale=1., event_shape=(4,), sample=True)

# Helper functions that are not distributions with samplers
def test_mask(): return "SKIPPED"
def test_plate(): return "SKIPPED"

if __name__ == "__main__":
    run_tests()
    print("\n--- All tests completed. ---")
# %%
