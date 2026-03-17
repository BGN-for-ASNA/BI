
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
x = np.random.normal(0, 1, size=N).astype(np.float32)
y = np.random.normal(0, 1, size=N).astype(np.float32)

m.data_on_model = {
    "group_id": group_id.astype(np.int32),
    "group_to_region": group_to_region.astype(np.int32),
    "x": x,
    "y": y,
}

def model(group_id, group_to_region, x, y):
    sigma = m.dist.exponential(1, name='sigma')
    
    # Region Level (returns arrays for the 5 regions)
    a_reg, b_reg = m.effects.varying_effects(
        N_vars = 1,
        N_group = 5,
        group_name = 'region'
    )
    
    # Group Level (returns arrays for the 20 groups)
    # alpha_bar and beta_bar must have size 20 (one for each group)
    a_grp, b_grp = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        alpha_bar = a_reg.flatten()[group_to_region],
        beta_bar = b_reg.flatten()[group_to_region],
        group_name = 'group'
    )
    
    # Likelihood (indexing to observations)
    mu = a_grp.flatten()[group_id] + b_grp.flatten()[group_id] * x
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
