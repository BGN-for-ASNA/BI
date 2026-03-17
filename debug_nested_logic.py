
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
    "group_to_region": group_to_region.astype(np.int32),
    "x": x,
    "y": y,
}

def model(group_id, region_id, group_to_region, x, y):
    sigma = m.dist.exponential(1, name='sigma')
    
    # Region Level (size 5)
    a_reg, b_reg = m.effects.varying_effects(
        N_vars = 1,
        N_group = 5,
        group_id = jnp.arange(5),
        group_name = 'region'
    )
    
    print(f"DEBUG: a_reg shape: {a_reg.shape}") # Should be (5,) or (5, 1)
    
    # Map to groups (size 20)
    alpha_bar = a_reg.flatten()[group_to_region]
    beta_bar = b_reg.flatten()[group_to_region]
    
    # Group Level (size 400 via group_id)
    a_grp, b_grp = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        group_id = group_id,
        alpha_bar = alpha_bar, 
        beta_bar = beta_bar,  
        group_name = 'group'
    )
    
    print(f"DEBUG: a_grp shape: {a_grp.shape}") # Should be (400,) or (400, 1)
    
    mu = a_grp.flatten() + b_grp.flatten() * x
    m.dist.normal(mu, sigma, obs=y)

print("Starting debug fit...")
try:
    m.fit(model, num_samples=1, num_warmup=1, num_chains=1, progress_bar=False)
    print("DEBUG: Fit success!")
except Exception as e:
    import traceback
    traceback.print_exc()
