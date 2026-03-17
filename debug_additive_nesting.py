
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

group_to_region = np.repeat(np.arange(N_regions), N_groups_per_region)
group_id = np.repeat(np.arange(N_groups), N_obs_per_group)
region_id = group_to_region[group_id]
x = np.random.normal(0, 1, size=N).astype(np.float32)
y = np.random.normal(0, 1, size=N).astype(np.float32)

m.data_on_model = {
    "group_id": group_id.astype(np.int32),
    "region_id": region_id.astype(np.int32),
    "x": x,
    "y": y,
}

def model(group_id, region_id, x, y):
    sigma = m.dist.exponential(1, name='sigma')
    
    # Region Level
    a_reg, b_reg = m.effects.varying_effects(
        N_vars = 1,
        N_group = 5,
        group_id = region_id,
        group_name = 'region'
    )
    a_reg, b_reg = a_reg.flatten(), b_reg.flatten()
    
    # Group Level (Residuals)
    a_grp_res, b_grp_res = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        group_id = group_id,
        group_name = 'group'
    )
    a_grp_res, b_grp_res = a_grp_res.flatten(), b_grp_res.flatten()
    
    # Combine (Additive nesting)
    a_total = a_reg + a_grp_res
    b_total = b_reg + b_grp_res
    
    mu = a_total + b_total * x
    m.dist.normal(mu, sigma, obs=y)

print("Starting debug fit...")
try:
    m.fit(model, num_samples=100, num_warmup=100, num_chains=1, progress_bar=False)
    print("DEBUG: Fit success!")
    m.summary()
    print("DEBUG: Summary success!")
except Exception:
    import traceback
    traceback.print_exc()
