
import sys
import os
import traceback
from BI import bi, jnp
import jax
import numpy as np

def run_debug():
    try:
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

        group_id = np.repeat(np.arange(N_groups), N_obs_per_group).astype(np.int32)
        x = np.random.normal(0, 1, size=N).astype(np.float32)
        a_g = group_effects[:, 0]
        b_g = group_effects[:, 1]
        mu = a_g[group_id] + b_g[group_id] * x
        y = np.random.normal(mu, sigma_obs).astype(np.float32)

        m.data_on_model = {
            "group_id": group_id,
            "x": x,
            "y": y,
            "group_to_region": group_to_region.astype(np.int32),
            "N_regions": N_regions,
        }

        def model_nested(group_id, x, y, group_to_region, N_regions):
            sigma = m.dist.exponential(1, name='sigma')
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
            sigma_grp = m.dist.exponential(1, shape=(2,), name='sigma_grp')
            Rho_grp = m.dist.lkj(2, 2, name='Rho_grp')
            cov_grp = jnp.outer(sigma_grp, sigma_grp) * Rho_grp
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

        print("DEBUG: Starting fit...")
        m.fit(model_nested, num_samples=100, num_warmup=100, num_chains=1, progress_bar=False) 
        print("DEBUG: Starting summary...")
        m.summary()
        print("DEBUG: All good.")
    except Exception:
        print("DEBUG: Caught an exception!")
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
