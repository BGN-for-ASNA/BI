#%%
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from BI import bi

# ==========================================
# Common Parameters
# ==========================================
n = 1000
mu_a, mu_b = 0.0, 0.0
sigma_a, sigma_b = 1.0, 1.0
spearman_rho = 0.6
pearson_rho = 2 * np.sin((np.pi / 6) * spearman_rho)

# Covariance matrix for bivariate standard normal
Sigma = jnp.array([
    [sigma_a**2, pearson_rho * sigma_a * sigma_b],
    [pearson_rho * sigma_a * sigma_b, sigma_b**2]
])
mu = jnp.array([mu_a, mu_b])

seed = 123
rng_key = random.PRNGKey(seed)


def plot_results(x1, x2, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(x1, x2, alpha=0.5)
    axes[0].set_title(f"Joint Density\n{title}")
    
    axes[1].hist(x1, bins=30, density=True)
    axes[1].set_title("Marginal 1")
    
    axes[2].hist(x2, bins=30, density=True)
    axes[2].set_title("Marginal 2")
    
    plt.tight_layout()
    plt.show()

def display_correlation_recovery(x1, x2, original_rho):
    """
    Recover Pearson correlation based on rank correlation measures 
    for Gaussian copulas using mathematical transforms.
    tau_rho = (2/pi) * arcsin(rho) -> rho = sin(tau * pi / 2)
    rho_S = (6/pi) * arcsin(rho/2) -> rho = 2 * sin(rho_S * pi / 6)
    """
    # Convert jax arrays to numpy for scipy.stats
    x1_np = np.asarray(x1)
    x2_np = np.asarray(x2)
    
    # 1. Pearson on transformed data
    pearson_r, _ = stats.pearsonr(x1_np, x2_np)
    
    # 2. Kendall's Tau
    kendall_tau, _ = stats.kendalltau(x1_np, x2_np)
    
    # 3. Spearman's Rho
    spearman_rho, _ = stats.spearmanr(x1_np, x2_np)
    
    # Recover original Pearson rho
    recovered_rho_from_tau = np.sin(kendall_tau * np.pi / 2.0)
    recovered_rho_from_spearman = 2.0 * np.sin(spearman_rho * np.pi / 6.0)
    
    print("-" * 50)
    print("Correlation Recovery:")
    print(f"Original latent normal Pearson rho: {original_rho:.4f}")
    print(f"Pearson r on transformed data:      {pearson_r:.4f} (Not preserved)")
    print(f"Kendall's tau:                      {kendall_tau:.4f}")
    print(f"Spearman's rho (rho_S):             {spearman_rho:.4f}")
    print(f"Recovered Normal rho (from tau):    {recovered_rho_from_tau:.4f}")
    print(f"Recovered Normal rho (from rho_S):  {recovered_rho_from_spearman:.4f}")
    print("-" * 50)

#%%
# ==========================================
# Approach 1: Raw approach 
# Translating the R code directly
# ==========================================
print("\n" + "="*50)
print("Approach 1: Raw Gaussian Copula")
print("="*50)
import random as rd
rd.randint(0,100000000)
spearman_r = []
for i in range(10):
    rng_key = random.PRNGKey(rd.randint(0,100000000))
    # Step 1: Sample from bivariate standard normal
    samples_normal = random.multivariate_normal(rng_key, mean=mu, cov=Sigma, shape=(n,))

    # Step 2: Transform into standard uniforms using normal CDF
    # Note: jax.scipy.stats.norm.cdf takes loc=0, scale=1 by default
    samples_uniform = norm.cdf(samples_normal)

    # Step 3: Transform uniforms to desired marginals using quantile functions (inverse CDF)
    alpha_param, beta_param = 2.0, 5.0
    lambda_param = 3.0

    # SciPy allows easy application of ppf (percent point function / quantile function)
    u1 = np.asarray(samples_uniform[:, 0])
    u2 = np.asarray(samples_uniform[:, 1])

    # Beta quantile

    x_beta = stats.beta.ppf(u1, a=alpha_param, b=beta_param)
    # Poisson quantile
    x_poisson = stats.poisson.ppf(u2, mu=lambda_param)

    # Display rank correlation statistics and recovery
    display_correlation_recovery(x_beta, x_poisson, original_rho=pearson_rho)
    spearman_r.append(stats.spearmanr(x_beta, x_poisson)[0])

plt.hist(spearman_r)
plt.show()

# Plot
# plot_results(x_beta, x_poisson, "Raw Approach (Beta & Poisson)")

#%%
# ==========================================
# Approach 2: Built-in approach with BI
# ==========================================
print("\n" + "="*50)
print("Approach 2: Built-in bi.dist.gaussian_copula")
print("="*50)

m = bi('cpu')

# GaussianCopula in numpyro requires the marginal_dist to handle a batch shape,
# meaning the distributions of both variables must be of the exact same family.
# Since we want a Beta and a Poisson, we build a custom marginal wrapper that
# provides `icdf` resolving to Beta for dimension 0 and Poisson for dimension 1.

import jax.numpy as jnp
import jax.scipy.special as jss
import numpyro.distributions as dist
from numpyro.distributions import constraints
# Import TensorFlow Probability with JAX backend
from tensorflow_probability.substrates import jax as tfp

class BetaPoissonMarginal(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, alpha, beta, lam, validate_args=None):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        super().__init__(batch_shape=(2,), event_shape=(), validate_args=validate_args)

    def log_prob(self, value):
        v1 = value[..., 0]
        v2 = value[..., 1]
        lp1 = dist.Beta(self.alpha, self.beta).log_prob(v1)
        lp2 = dist.Poisson(self.lam).log_prob(v2)
        return jnp.stack([lp1, lp2], axis=-1)

    def cdf(self, value):
        v1 = value[..., 0]
        v2 = value[..., 1]
        
        # FIX: Use TFP's betainc, which supports gradients wrt alpha/beta
        cdf1 = tfp.math.betainc(self.alpha, self.beta, v1)
        
        # Native JAX Poisson CDF is fine here because we only need gradient 
        # wrt 'lam' (the 2nd arg), not the data 'v2' (the 1st arg).
        # P(X <= k) = Q(k+1, lambda) = gammaincc(k+1, lambda)
        cdf2 = jss.gammaincc(jnp.floor(v2) + 1.0, self.lam)
        
        return jnp.stack([cdf1, cdf2], axis=-1)

    def icdf(self, u):
        # ICDF is usually not used during inference (only log_prob and cdf),
        # but required for sampling.
        # Note: Scipy is CPU only, which is fine for post-inference sampling 
        # but not inside the model function if you wanted to do transformed distributions.
        import numpy as np
        import scipy.stats as stats
        
        u1 = np.asarray(u[..., 0])
        u2 = np.asarray(u[..., 1])
        x_b = stats.beta.ppf(u1, a=np.asarray(self.alpha), b=np.asarray(self.beta))
        x_p = stats.poisson.ppf(u2, mu=np.asarray(self.lam))
        return jnp.stack([jnp.array(x_b), jnp.array(x_p)], axis=-1)

# Cholesky of the correlation matrix is often required/preferred for numerical stability
Sigma_chol = jnp.linalg.cholesky(Sigma)

# Sample using the BI object with our mocked Marginal

for i in range(1):
    samples_bi = m.dist.gaussian_copula(
        marginal_dist=BetaPoissonMarginal(alpha_param, beta_param, lambda_param),
        correlation_cholesky=Sigma_chol,
        sample=True,
        shape=(n,),
        seed=seed
    )

    x_b1 = samples_bi[:, 0]
    x_b2 = samples_bi[:, 1]

display_correlation_recovery(x_b1, x_b2, original_rho=pearson_rho)
plot_results(x_b1, x_b2, "Built-in Copula (Beta & Poisson)")

# ==========================================
# Inference model
# ==========================================
print("\n" + "="*50)
print("Running BI Inference Model to recover parameters")
print("="*50)

def model(x_b1, x_b2):
    # Priors for marginal distributions
    alpha_est = m.dist.exponential(0.1, name='alpha_est')
    beta_est = m.dist.exponential(0.1, name='beta_est')
    lambda_est = m.dist.exponential(0.1, name='lambda_est')
    
    # Prior for Copula Correlation
    # We use LkjCholesky with dimension 2 for bivariate dependence
    rho_est = m.dist.lkj_cholesky(2, 2.0, name='rho_est')
    
    # Bundle the observations into shape (n, 2)
    obs_data = jnp.stack([x_b1, x_b2], axis=-1)
    m.dist.gaussian_copula(
        marginal_dist=BetaPoissonMarginal(alpha_est, beta_est, lambda_est),
        correlation_cholesky=rho_est,
        obs=obs_data,
        name='obs'
    )

m.data_on_model = {'x_b1': x_b1, 'x_b2': x_b2}
m.fit(model, num_samples=300, num_warmup=100, num_chains=1, progress_bar=True)
m.sampler.print_summary()

# Extract samples and plot posteriors vs true values
samples = m.sampler.get_samples()

alpha_samples = np.asarray(samples['alpha_est'])
beta_samples = np.asarray(samples['beta_est'])
lambda_samples = np.asarray(samples['lambda_est'])
# The correlation matrix is L @ L.T. 
# Since L is lower triangular with L[0,0]=1, the correlation is simply L[1,0].
rho_samples = np.asarray(samples['rho_est'][:, 1, 0])
spearman_rho_samples = (6.0 / np.pi) * np.arcsin(rho_samples / 2.0)

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Estimated Posteriors vs True Simulated Parameters", fontsize=16)

# Alpha
sns.kdeplot(alpha_samples, ax=axes[0, 0], fill=True, label='Estimated Posterior')
axes[0, 0].axvline(x=alpha_param, color='r', linestyle='--', label='True Value')
axes[0, 0].set_title('Alpha Parameter (Beta Dist)')
axes[0, 0].legend()

# Beta
sns.kdeplot(beta_samples, ax=axes[0, 1], fill=True, label='Estimated Posterior')
axes[0, 1].axvline(x=beta_param, color='r', linestyle='--', label='True Value')
axes[0, 1].set_title('Beta Parameter (Beta Dist)')
axes[0, 1].legend()

# Lambda
sns.kdeplot(lambda_samples, ax=axes[1, 0], fill=True, label='Estimated Posterior')
axes[1, 0].axvline(x=lambda_param, color='r', linestyle='--', label='True Value')
axes[1, 0].set_title('Lambda Parameter (Poisson Dist)')
axes[1, 0].legend()

# Rho (Pearson Correlation)
sns.kdeplot(rho_samples, ax=axes[1, 1], fill=True, label='Estimated Posterior')
axes[1, 1].axvline(x=pearson_rho, color='r', linestyle='--', label='True Value')
axes[1, 1].set_title('Latent Pearson Correlation (rho)')
axes[1, 1].legend()

# Spearman Rho
sns.kdeplot(spearman_rho_samples, ax=axes[2, 0], fill=True, label='Estimated Posterior')
axes[2, 0].axvline(x=spearman_rho, color='r', linestyle='--', label='True Value')
axes[2, 0].set_title("Spearman's Rho")
axes[2, 0].legend()

axes[2, 1].axis('off')

plt.tight_layout()
plt.show()