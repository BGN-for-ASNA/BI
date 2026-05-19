import numpy as np
import os

np.random.seed(42)

# Simulated Data
N_regions = 5
N_groups_per_region = 4
N_groups = N_regions * N_groups_per_region
N_obs_per_group = 20
N = N_groups * N_obs_per_group

# Region level (Top level) parameters
mu_a_region, mu_b_region = 5.0, -1.0
sigma_a_region, sigma_b_region = 1.0, 0.5
rho_region = -0.5

# Group level (Nested level) parameters
sigma_a_group, sigma_b_group = 0.5, 0.2
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
x = np.random.normal(0, 1, size=N) # 1D
a_g = group_effects[:, 0]
b_g = group_effects[:, 1]
mu = a_g[group_id] + b_g[group_id] * x
y = np.random.normal(mu, sigma_obs) # 1D

data = {
    "group_id": group_id.astype(np.int32),
    "x": x.astype(np.float32),
    "y": y.astype(np.float32),
    "group_to_region": group_to_region.astype(np.int32),
    "N_regions": N_regions,
}

np.savez("simulated_data.npz", **data)

true_params = {
    "mu_a_reg": mu_a_region,
    "mu_b_reg": mu_b_region,
    "sigma_reg_a": sigma_a_region,
    "sigma_reg_b": sigma_b_region,
    "sigma": sigma_obs
}

with open("true_params.txt", "w") as f:
    for k, v in true_params.items():
        f.write(f"{k}: {v}\n")

print("Generative data created.")
