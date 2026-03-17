
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
    
    # 1. Region Level (Manual)
    a_reg = m.dist.normal(5, 2, shape=(5,), name='a_reg')
    b_reg = m.dist.normal(-1, 1, shape=(5,), name='b_reg')
    
    # 2. Group Level (Built-in)
    # alpha_bar and beta_bar of size 400 (indexed)
    a_grp, b_grp = m.effects.varying_effects(
        N_vars = 1,
        N_group = 20,
        group_id = group_id,
        alpha_bar = a_reg[group_to_region[group_id]], # Map region to group then obs
        beta_bar = b_reg[group_to_region[group_id]],
        group_name = 'group'
    )
    # Ensure 1D
    a_grp, b_grp = a_grp.flatten(), b_grp.flatten()
    
    mu = a_grp + b_grp * x
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
