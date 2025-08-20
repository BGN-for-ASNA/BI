#%%
import os
import sys
newPath = os.path.dirname(os.path.abspath(""))
if newPath not in sys.path:
    sys.path.append(newPath)
from BI import bi
#%%
import jax.numpy as jnp
m = bi('cpu')
errors = []

try:
    m.dist.asymmetric_laplace(loc=0.0, scale=1.0, asymmetry=1.0, sample=True)
except Exception as e:
    print(f'Error in asymmetric_laplace: {e}')
    errors.append(f'Error in asymmetric_laplace: {e}')

try:
    m.dist.asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=0.5, sample=True)
except Exception as e:
    print(f'Error in asymmetric_laplace_quantile: {e}')
    errors.append(f'Error in asymmetric_laplace_quantile: {e}')

try:
    m.dist.bernoulli(probs=0.7, sample=True)
except Exception as e:
    print(f'Error in bernoulli: {e}')
    errors.append(f'Error in bernoulli: {e}')

try:
    m.dist.bernoulli_logits(logits=jnp.array([0.2, 1, 2]), sample=True)
except Exception as e:
    print(f'Error in bernoulli_logits: {e}')
    errors.append(f'Error in bernoulli_logits: {e}')

try:
    m.dist.bernoulli_probs(probs=jnp.array([0.2, 0.7, 0.5]), sample=True)
except Exception as e:
    print(f'Error in bernoulli_probs: {e}')
    errors.append(f'Error in bernoulli_probs: {e}')

try:
    m.dist.beta(concentration1=1.0, concentration0=1.0, sample=True)
except Exception as e:
    print(f'Error in beta: {e}')
    errors.append(f'Error in beta: {e}')

try:
    m.dist.beta_binomial(concentration1=1.0, concentration0=1.0, total_count=10, sample=True)
except Exception as e:
    print(f'Error in beta_binomial: {e}')
    errors.append(f'Error in beta_binomial: {e}')

try:
    samples = m.dist.beta_proportion(mean=0.5, concentration=2.0, sample=True, shape=(1000,))
except Exception as e:
    print(f'Error in beta_proportion: {e}')
    errors.append(f'Error in beta_proportion: {e}')

try:
    m.dist.binomial(total_count=10, probs=0.5, sample=True)
except Exception as e:
    print(f'Error in binomial: {e}')
    errors.append(f'Error in binomial: {e}')

try:
    m.dist.binomial_logits(logits=jnp.zeros(10), total_count=5, sample=True)
except Exception as e:
    print(f'Error in binomial_logits: {e}')
    errors.append(f'Error in binomial_logits: {e}')

try:
    m.dist.binomial_probs(probs=0.5, total_count=10, sample=True)
except Exception as e:
    print(f'Error in binomial_probs: {e}')
    errors.append(f'Error in binomial_probs: {e}')

try:
    m.dist.normal(loc = 0, scale = 1, sample = True)
except Exception as e:
    print(f'Error in car: {e}')
    errors.append(f'Error in car: {e}')

try:
    m.dist.categorical(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in categorical: {e}')
    errors.append(f'Error in categorical: {e}')

try:
    m.dist.categorical_logits(logits=jnp.zeros(5), sample=True)
except Exception as e:
    print(f'Error in categorical_logits: {e}')
    errors.append(f'Error in categorical_logits: {e}')

try:
    m.dist.categorical_probs(probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in categorical_probs: {e}')
    errors.append(f'Error in categorical_probs: {e}')

try:
    m.dist.cauchy(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in cauchy: {e}')
    errors.append(f'Error in cauchy: {e}')

try:
    m.dist.chi2(df=3.0, sample = True)
except Exception as e:
    print(f'Error in chi2: {e}')
    errors.append(f'Error in chi2: {e}')

try:
    m.dist.normal(loc = 0, scale = 1, sample = True)
except Exception as e:
    print(f'Error in circulant_normal: {e}')
    errors.append(f'Error in circulant_normal: {e}')

try:
    m.dist.delta(v=0.0, sample=True)
except Exception as e:
    print(f'Error in delta: {e}')
    errors.append(f'Error in delta: {e}')

try:
    m.dist.dirichlet(concentration=jnp.array([1.0, 1.0, 1.0]), sample=True)
except Exception as e:
    print(f'Error in dirichlet: {e}')
    errors.append(f'Error in dirichlet: {e}')

try:
    m.dist.dirichlet_multinomial(concentration=jnp.array([1.0, 1.0, 1.0]), total_count=10, sample=True)
except Exception as e:
    print(f'Error in dirichlet_multinomial: {e}')
    errors.append(f'Error in dirichlet_multinomial: {e}')

try:
    m.dist.discrete_uniform(low=0, high=5, sample=True)
except Exception as e:
    print(f'Error in discrete_uniform: {e}')
    errors.append(f'Error in discrete_uniform: {e}')

try:
    m.dist.doubly_truncated_power_law(low=0.1, high=10.0, alpha=2.0, sample=True)
except Exception as e:
    print(f'Error in doubly_truncated_power_law: {e}')
    errors.append(f'Error in doubly_truncated_power_law: {e}')

try:
    m.dist.euler_maruyama(t=jnp.array([0.0, 0.1, 0.2]), sde_fn=lambda x, t: (x, 1.0), init_dist=m.dist.normal(0.0, 1.0, create_obj=True), sample = True)
except Exception as e:
    print(f'Error in euler_maruyama: {e}')
    errors.append(f'Error in euler_maruyama: {e}')

try:
    m.dist.exponential(rate=1.0, sample=True)
except Exception as e:
    print(f'Error in exponential: {e}')
    errors.append(f'Error in exponential: {e}')

try:
    m.dist.folded_distribution(m.dist.normal(loc=0.0, scale=1.0, create_obj = True), sample=True)
except Exception as e:
    print(f'Error in folded_distribution: {e}')
    errors.append(f'Error in folded_distribution: {e}')

try:
    m.dist.gamma(concentration=2.0, rate=0.5, sample=True)
except Exception as e:
    print(f'Error in gamma: {e}')
    errors.append(f'Error in gamma: {e}')

try:
    m.dist.gamma_poisson(concentration=1.0, rate=2.0, sample=True)
except Exception as e:
    print(f'Error in gamma_poisson: {e}')
    errors.append(f'Error in gamma_poisson: {e}')

try:
    m.dist.gaussian_copula(m.dist.beta(2.0, 5.0, create_obj = True), correlation_matrix = jnp.array([
    [1.0, 0.7],  # Correlation between the two marginals
    [0.7, 1.0]
])
, sample = True)
except Exception as e:
    print(f'Error in gaussian_copula: {e}')
    errors.append(f'Error in gaussian_copula: {e}')

try:
    m.dist.gaussian_copula_beta(
        concentration1 = jnp.array([2.0, 3.0]), 
        concentration0 = jnp.array([5.0, 3.0]),
        correlation_cholesky = jnp.linalg.cholesky(jnp.array([[1.0, 0.7],[0.7, 1.0]])), 
        sample = True
    )
except Exception as e:
    print(f'Error in gaussian_copula_beta: {e}')
    errors.append(f'Error in gaussian_copula_beta: {e}')

try:
    m.dist.gaussian_random_walk(scale=1.0, sample=True)
except Exception as e:
    print(f'Error in gaussian_random_walk: {e}')
    errors.append(f'Error in gaussian_random_walk: {e}')

try:
    m.dist.gaussian_state_space(num_steps=5, transition_matrix=jnp.array([[0.5]]), covariance_matrix =  jnp.array([[1.0, 0.6],[0.6, 1.0]]), sample=True)
except Exception as e:
    print(f'Error in gaussian_state_space: {e}')
    errors.append(f'Error in gaussian_state_space: {e}')

try:
    m.dist.geometric(probs=0.5, sample=True)
except Exception as e:
    print(f'Error in geometric: {e}')
    errors.append(f'Error in geometric: {e}')

try:
    m.dist.geometric_logits(logits=jnp.zeros(10), sample=True)
except Exception as e:
    print(f'Error in geometric_logits: {e}')
    errors.append(f'Error in geometric_logits: {e}')

try:
    m.dist.geometric_probs(probs=0.5, sample=True)
except Exception as e:
    print(f'Error in geometric_probs: {e}')
    errors.append(f'Error in geometric_probs: {e}')

try:
    m.dist.gompertz(concentration=1.0, rate=1.0, sample=True)
except Exception as e:
    print(f'Error in gompertz: {e}')
    errors.append(f'Error in gompertz: {e}')

try:
    m.dist.gumbel(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in gumbel: {e}')
    errors.append(f'Error in gumbel: {e}')

try:
    m.dist.half_cauchy(scale=1.0, sample=True)
except Exception as e:
    print(f'Error in half_cauchy: {e}')
    errors.append(f'Error in half_cauchy: {e}')

try:
    m.dist.half_normal(scale=1.0, sample=True)
except Exception as e:
    print(f'Error in half_normal: {e}')
    errors.append(f'Error in half_normal: {e}')


try:
    m.dist.inverse_gamma(concentration=2.0, rate=1.0, sample=True)
except Exception as e:
    print(f'Error in inverse_gamma: {e}')
    errors.append(f'Error in inverse_gamma: {e}')

try:
    m.dist.kumaraswamy(concentration1=2.0, concentration0=3.0, sample=True)
except Exception as e:
    print(f'Error in kumaraswamy: {e}')
    errors.append(f'Error in kumaraswamy: {e}')

try:
    m.dist.laplace(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in laplace: {e}')
    errors.append(f'Error in laplace: {e}')

try:
    m.dist.normal(loc = 0, scale = 1, sample = True)
except Exception as e:
    print(f'Error in left_truncated_distribution: {e}')
    errors.append(f'Error in left_truncated_distribution: {e}')

try:
    m.dist.levy(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in levy: {e}')
    errors.append(f'Error in levy: {e}')

try:
    m.dist.lkj(dimension=2, concentration=1.0, sample=True)
except Exception as e:
    print(f'Error in lkj: {e}')
    errors.append(f'Error in lkj: {e}')

try:
    m.dist.normal(loc = 0, scale = 1, sample = True)
except Exception as e:
    print(f'Error in lkj_cholesky: {e}')
    errors.append(f'Error in lkj_cholesky: {e}')

try:
    m.dist.log_normal(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in log_normal: {e}')
    errors.append(f'Error in log_normal: {e}')

try:
    m.dist.log_uniform(low=0.1, high=10.0, sample=True)
except Exception as e:
    print(f'Error in log_uniform: {e}')
    errors.append(f'Error in log_uniform: {e}')

try:
    m.dist.logistic(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in logistic: {e}')
    errors.append(f'Error in logistic: {e}')

try:
    event_size = 100  # Our distribution has 100 dimensions
    rank = 5   
    m.dist.low_rank_multivariate_normal(
        loc=m.dist.normal(0,1, shape = (event_size,), sample=True)*2, 
        cov_factor=m.dist.normal(0,1, shape = (event_size, rank), sample=True),
        cov_diag=jnp.exp(m.dist.normal(0,1, shape = (event_size,), sample=True)) * 0.1, sample=True)
except Exception as e:
    print(f'Error in low_rank_multivariate_normal: {e}')
    errors.append(f'Error in low_rank_multivariate_normal: {e}')

try:
    m.dist.lower_truncated_power_law(alpha=-2.0, low=1.0, sample=True)
except Exception as e:
    print(f'Error in lower_truncated_power_law: {e}')
    errors.append(f'Error in lower_truncated_power_law: {e}')


try:
    n_rows, n_cols = 3, 4
    loc = jnp.zeros((n_rows, n_cols))
    U_row_cov = jnp.array([[1.0, 0.5, 0.2],
                           [0.5, 1.0, 0.3],
                           [0.2, 0.3, 1.0]])
    scale_tril_row = jnp.linalg.cholesky(U_row_cov)
    # Assume some column-wise covariance matrix V
    V_col_cov = jnp.array([[2.0, -0.8, 0.1, 0.4],
                           [-0.8, 2.0, 0.2, -0.2],
                           [0.1, 0.2, 2.0, 0.0],
                           [0.4, -0.2, 0.0, 2.0]])

    # The argument passed to the distribution is its Cholesky factor
    scale_tril_column = jnp.linalg.cholesky(V_col_cov)

    m.dist.matrix_normal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, sample=True)
except Exception as e:
    print(f'Error in matrix_normal: {e}')
    errors.append(f'Error in matrix_normal: {e}')


try:
    m.dist.mixture_general(
        mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True), component_distributions=[m.dist.normal(loc=0.0, scale=1.0,  create_obj=True),m.dist.normal(loc=0.0, scale=1.0,  create_obj=True)],
        sample = True)
except Exception as e:
    print(f'Error in mixture_general: {e}')
    errors.append(f'Error in mixture_general: {e}')

try:
    m.dist.mixture_same_family(
        mixing_distribution=m.dist.categorical(probs=jnp.array([0.3, 0.7]), create_obj = True), component_distribution=m.dist.normal(loc=0.0, scale=1.0, shape = (2,), create_obj=True),
        sample = True
    )

except Exception as e:
    print(f'Error in mixture_same_family: {e}')
    errors.append(f'Error in mixture_same_family: {e}')

try:
    m.dist.multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in multinomial: {e}')
    errors.append(f'Error in multinomial: {e}')

try:
    m.dist.multinomial_logits(logits=jnp.array([1.0, 0.5], dtype=jnp.float32), total_count=jnp.array(5, dtype=jnp.int32), sample=True)
except Exception as e:
    print(f'Error in multinomial_logits: {e}')
    errors.append(f'Error in multinomial_logits: {e}')

try:
    m.dist.multinomial_probs(probs=jnp.array([0.2, 0.3, 0.5]), total_count=10, sample=True)
except Exception as e:
    print(f'Error in multinomial_probs: {e}')
    errors.append(f'Error in multinomial_probs: {e}')

try:
    m.dist.multivariate_normal(
        loc=jnp.array([1.0, 0.0, -2.0]), 
        covariance_matrix=jnp.array([[ 2.0,  0.7, -0.3],
                                    [ 0.7,  1.0,  0.5],
                                    [-0.3,  0.5,  1.5]]), 
        sample=True
    )
except Exception as e:
    print(f'Error in multivariate_normal: {e}')
    errors.append(f'Error in multivariate_normal: {e}')

try:
    m.dist.multivariate_student_t(
        df = 2,
        loc=jnp.array([1.0, 0.0, -2.0]), 
        scale_tril=jnp.linalg.cholesky(
            jnp.array([[ 2.0,  0.7, -0.3],
                        [ 0.7,  1.0,  0.5],
                        [-0.3,  0.5,  1.5]])), 
        sample=True
    )   

except Exception as e:
    print(f'Error in multivariate_student_t: {e}')
    errors.append(f'Error in multivariate_student_t: {e}')

try:
    m.dist.negative_binomial(total_count=5.0,probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in negative_binomial: {e}')
    errors.append(f'Error in negative_binomial: {e}')

try:
    m.dist.negative_binomial2(mean=2.0, concentration=3.0, sample=True)
except Exception as e:
    print(f'Error in negative_binomial2: {e}')
    errors.append(f'Error in negative_binomial2: {e}')

try:
    m.dist.negative_binomial_logits(total_count=5.0, logits=0.0, sample=True)
except Exception as e:
    print(f'Error in negative_binomial_logits: {e}')
    errors.append(f'Error in negative_binomial_logits: {e}')

try:
    m.dist.negative_binomial_probs(total_count=10.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in negative_binomial_probs: {e}')
    errors.append(f'Error in negative_binomial_probs: {e}')

try:
    m.dist.normal(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in normal: {e}')
    errors.append(f'Error in normal: {e}')

try:
    m.dist.ordered_logistic(predictor=jnp.array([0.2, 0.5, 0.8]), cutpoints=jnp.array([-1.0, 0.0, 1.0]), sample=True)
except Exception as e:
    print(f'Error in ordered_logistic: {e}')
    errors.append(f'Error in ordered_logistic: {e}')

try:
    m.dist.pareto(scale=2.0, alpha=3.0, sample=True)
except Exception as e:
    print(f'Error in pareto: {e}')
    errors.append(f'Error in pareto: {e}')

try:
    m.dist.poisson(rate=2.0, sample=True)
except Exception as e:
    print(f'Error in poisson: {e}')
    errors.append(f'Error in poisson: {e}')

try:
    m.dist.projected_normal(concentration=jnp.array([1.0, 3.0, 2.0]), sample=True)
except Exception as e:
    print(f'Error in projected_normal: {e}')
    errors.append(f'Error in projected_normal: {e}')

try:
    m.dist.relaxed_bernoulli(temperature=1.0, probs = jnp.array([0.2, 0.3, 0.5]), sample=True)
except Exception as e:
    print(f'Error in relaxed_bernoulli: {e}')
    errors.append(f'Error in relaxed_bernoulli: {e}')

try:
    m.dist.relaxed_bernoulli_logits(temperature=1.0, logits=0.0, sample=True)
except Exception as e:
    print(f'Error in relaxed_bernoulli_logits: {e}')
    errors.append(f'Error in relaxed_bernoulli_logits: {e}')

try:
    m.dist.right_truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=0, sample=True)
except Exception as e:
    print(f'Error in right_truncated_distribution: {e}')
    errors.append(f'Error in right_truncated_distribution: {e}')


try:
    m.dist.soft_laplace(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in soft_laplace: {e}')
    errors.append(f'Error in soft_laplace: {e}')

try:
    m.dist.student_t(df = 2, loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in student_t: {e}')
    errors.append(f'Error in student_t: {e}')

try:
    m.dist.truncated_cauchy(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in truncated_cauchy: {e}')
    errors.append(f'Error in truncated_cauchy: {e}')

try:
    m.dist.truncated_distribution(base_dist=m.dist.normal(loc=0.0, scale=1.0, create_obj = True), low=0, high=1,sample=True)
except Exception as e:
    print(f'Error in truncated_distribution: {e}')
    errors.append(f'Error in truncated_distribution: {e}')

try:
    m.dist.truncated_normal(loc=0.0, scale=1.0, sample=True)
except Exception as e:
    print(f'Error in truncated_normal: {e}')
    errors.append(f'Error in truncated_normal: {e}')

try:
    m.dist.truncated_polya_gamma(batch_shape=(), sample=True)
except Exception as e:
    print(f'Error in truncated_polya_gamma: {e}')
    errors.append(f'Error in truncated_polya_gamma: {e}')

try:
    m.dist.truncated_distribution(base_dist = m.dist.normal(0,1, create_obj = True), high=1, low = 0, sample=True)
except Exception as e:
    print(f'Error in two_sided_truncated_distribution: {e}')
    errors.append(f'Error in two_sided_truncated_distribution: {e}')

try:
    m.dist.uniform(low=0.0, high=1.0, sample=True)
except Exception as e:
    print(f'Error in uniform: {e}')
    errors.append(f'Error in uniform: {e}')

try:
    m.dist.unit(log_factor=jnp.ones(5), sample=True)
except Exception as e:
    print(f'Error in unit: {e}')
    errors.append(f'Error in unit: {e}')

try:
    m.dist.weibull(scale=1.0, concentration=2.0, sample=True)
except Exception as e:
    print(f'Error in weibull: {e}')
    errors.append(f'Error in weibull: {e}')

try:
    m.dist.wishart(concentration=5.0, scale_matrix=jnp.eye(2), sample=True)
except Exception as e:
    print(f'Error in wishart: {e}')
    errors.append(f'Error in wishart: {e}')

try:
    m.dist.normal(loc = 0, scale = 1, sample = True)
except Exception as e:
    print(f'Error in wishart_cholesky: {e}')
    errors.append(f'Error in wishart_cholesky: {e}')

try:
    m.dist.zero_inflated_distribution(base_dist=m.dist.poisson(rate=5, create_obj = True), gate = 0.3, sample=True)
except Exception as e:
    print(f'Error in zero_inflated_distribution: {e}')
    errors.append(f'Error in zero_inflated_distribution: {e}')

try:
    m.dist.zero_inflated_negative_binomial2(mean=2.0, concentration=3.0, gate = 0.3, sample=True)
except Exception as e:
    print(f'Error in zero_inflated_negative_binomial2: {e}')
    errors.append(f'Error in zero_inflated_negative_binomial2: {e}')

try:
    m.dist.zero_inflated_poisson(gate = 0.3, rate=2.0, sample=True)
except Exception as e:
    print(f'Error in zero_inflated_poisson: {e}')
    errors.append(f'Error in zero_inflated_poisson: {e}')

try:
    m.dist.zero_sum_normal(scale=1.0, event_shape = (2,), sample = True)
except Exception as e:
    print(f'Error in zero_sum_normal: {e}')
    errors.append(f'Error in zero_sum_normal: {e}')

print(errors)

# %%
