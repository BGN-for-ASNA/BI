
from BI import bi
import jax.numpy as jnp
import jax
import numpy as np
from jax.nn import softmax

# Setup device
m = bi(platform="cpu")

# Simulated data
np.random.seed(42)
N_regions = 5
N_groups_per_region = 4
N_groups = N_regions * N_groups_per_region
N_obs_per_group = 20
N = N_groups * N_obs_per_group

# Region level parameters
mu_a_region = 5.0
mu_b_region = -1.0
sigma_a_region = 1.0
sigma_b_region = 0.5
rho_region = -0.5

# Group level parameters (deviation from region)
sigma_a_group = 0.5
sigma_b_group = 0.2
rho_group = 0.3

sigma_obs = 0.5

# Generate region effects
mu_region = np.array([mu_a_region, mu_b_region])
sigmas_region = np.array([sigma_a_region, sigma_b_region])
Rho_region = np.array([[1, rho_region], [rho_region, 1]])
Sigma_region = np.outer(sigmas_region, sigmas_region) * Rho_region
region_effects = np.random.multivariate_normal(mu_region, Sigma_region, size=N_regions)

# Generate group effects nested in regions
group_to_region = np.repeat(np.arange(N_regions), N_groups_per_region)
sigmas_group = np.array([sigma_a_group, sigma_b_group])
Rho_group = np.array([[1, rho_group], [rho_group, 1]])
Sigma_group = np.outer(sigmas_group, sigmas_group) * Rho_group

group_effects = np.zeros((N_groups, 2))
for g in range(N_groups):
    reg = group_to_region[g]
    group_effects[g] = np.random.multivariate_normal(region_effects[reg], Sigma_group)

# Generate observations
group_id = np.repeat(np.arange(N_groups), N_obs_per_group)
region_id = group_to_region[group_id]
x = np.random.normal(0, 1, size=N)

a_g = group_effects[:, 0]
b_g = group_effects[:, 1]
mu = a_g[group_id] + b_g[group_id] * x
y = np.random.normal(mu, sigma_obs)

print(f"True mu_a_region: {mu_a_region}, mu_b_region: {mu_b_region}")
print(f"True sigma_region (a, b): {sigma_a_region}, {sigma_b_region}")
print(f"True sigma_group (a, b): {sigma_a_group}, {sigma_b_group}")

# Model
def model_nested(group_id, x, y, group_to_region, N_regions, N_groups):
    sigma = m.dist.exponential(1, name='sigma')
    
    # 1. Region level
    mu_a_reg = m.dist.normal(5, 2, name='mu_a_reg')
    mu_b_reg = m.dist.normal(-1, 1, name='mu_b_reg')
    sigma_reg = m.dist.exponential(1, shape=(2,), name='sigma_reg')
    Rho_reg = m.dist.lkj(2, 2, name='Rho_reg')
    cov_reg = jnp.outer(sigma_reg, sigma_reg) * Rho_reg
    
    region_effects = m.dist.multivariate_normal(
        jnp.stack([mu_a_reg, mu_b_reg]), 
        cov_reg, 
        shape=(N_regions,), 
        name='region_effects'
    )
    
    # 2. Group level
    sigma_grp = m.dist.exponential(1, shape=(2,), name='sigma_grp')
    Rho_grp = m.dist.lkj(2, 2, name='Rho_grp')
    cov_grp = jnp.outer(sigma_grp, sigma_grp) * Rho_grp
    
    # Nested mean
    mu_grp = region_effects[group_to_region]
    
    group_effects = m.dist.multivariate_normal(
        mu_grp, 
        cov_grp, 
        name='group_effects'
    )
    
    a_g_est = group_effects[:, 0]
    b_g_est = group_effects[:, 1]
    
    mu_est = a_g_est[group_id] + b_g_est[group_id] * x
    m.dist.normal(mu_est, sigma, obs=y)

m.data_on_model = {
    "group_id": group_id.astype(np.int32),
    "x": x.astype(np.float32),
    "y": y.astype(np.float32),
    "group_to_region": group_to_region.astype(np.int32),
    "N_regions": N_regions,
    "N_groups": N_groups
}

m.fit(model_nested, num_samples=1000, num_warmup=500, progress_bar=False)

summ = m.summary()
print("\nPosterior Means for hyperparams:")
params = ['mu_a_reg', 'mu_b_reg', 'sigma_reg[0]', 'sigma_reg[1]', 'sigma_grp[0]', 'sigma_grp[1]', 'sigma']
print(summ.loc[params, ['mean']])
