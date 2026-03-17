
from BI import bi, jnp
import jax
import numpy as np

m = bi(platform='cpu')
np.random.seed(42)
N_regions = 5
N_groups_per_region = 4
N_groups = N_regions * N_groups_per_region
N_obs_per_group = 20
N = N_groups * N_obs_per_group

mu_a_region, mu_b_region = 5.0, -1.0
sigma_a_region, sigma_b_region = 1.0, 0.5
rho_region = -0.5
sigma_a_group, sigma_b_group = 0.5, 0.2
rho_group = 0.3
sigma_obs = 0.5

mu_region = np.array([mu_a_region, mu_b_region])
sigmas_region = np.array([sigma_a_region, sigma_b_region])
Rho_region = np.array([[1, rho_region], [rho_region, 1]])
Sigma_region = np.outer(sigmas_region, sigmas_region) * Rho_region
region_effects = np.random.multivariate_normal(mu_region, Sigma_region, size=N_regions)

group_to_region = np.repeat(np.arange(N_regions), N_groups_per_region)
sigmas_group = np.array([sigma_a_group, sigma_b_group])
Rho_group = np.array([[1, rho_group], [rho_group, 1]])
Sigma_group = np.outer(sigmas_group, sigmas_group) * Rho_group
group_effects = np.zeros((N_groups, 2))
for g in range(N_groups):
    reg = group_to_region[g]
    group_effects[g] = np.random.multivariate_normal(region_effects[reg], Sigma_group)

group_id = np.repeat(np.arange(N_groups), N_obs_per_group)
region_id = group_to_region[group_id]
x = np.random.normal(0, 1, size=N)
a_g = group_effects[:, 0]
b_g = group_effects[:, 1]
mu = a_g[group_id] + b_g[group_id] * x
y = np.random.normal(mu, sigma_obs)

m.data_on_model = {
    "group_id": group_id.astype(np.int32),
    "region_id": region_id.astype(np.int32),
    "x": x.astype(np.float32),
    "y": y.astype(np.float32),
}

def model(group_id, region_id, x, y):
    sigma = m.dist.exponential(1, name='sigma')
    a_reg, b_reg = m.effects.varying_effects(
        N_vars = 1,
        N_group = 5,
        group_id = region_id,
        group_name = 'region'
    )
    a_grp, b_grp = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        group_id = group_id,
        alpha_bar = a_reg, 
        beta_bar = b_reg,  
        group_name = 'group'
    )
    mu = a_grp + b_grp * x
    m.dist.normal(mu, sigma, obs=y)

print("Starting fit (num_chains=1)...")
m.fit(model, num_samples=1000, num_warmup=500, num_chains=1, progress_bar=False)
print("Fit 1 done. Starting summary...")
m.summary()
print("Summary 1 done.")

print("\nStarting fit (num_chains=2)...")
try:
    m.fit(model, num_samples=100, num_warmup=100, num_chains=2, progress_bar=False)
    print("Fit 2 done. Starting summary...")
    m.summary()
    print("Summary 2 done.")
except Exception as e:
    print(f"Fit 2 failed: {e}")
    import traceback
    traceback.print_exc()
