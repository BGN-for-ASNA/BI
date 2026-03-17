
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
    
    # Region Level (indexed to 400)
    a_reg, b_reg = m.effects.varying_effects(
        N_vars = 1,
        N_group = 5,
        group_id = region_id,
        group_name = 'region'
    )
    # CRITICAL: Flatten to ensure consistent 1D
    a_reg, b_reg = a_reg.flatten(), b_reg.flatten()
    
    # Group Level (indexed to 400)
    a_grp, b_grp = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        group_id = group_id,
        alpha_bar = a_reg, 
        beta_bar = b_reg,  
        group_name = 'group'
    )
    # Ensure 1D for linear model
    a_grp, b_grp = a_grp.flatten(), b_grp.flatten()
    
    mu = a_grp + b_grp * x
    m.dist.normal(mu, sigma, obs=y)

print("Starting debug fit...")
try:
    m.fit(model, num_samples=10, num_warmup=10, num_chains=1, progress_bar=False)
    print("DEBUG: Fit success!")
    m.summary()
    print("DEBUG: Summary success!")
except Exception:
    import traceback
    traceback.print_exc()
