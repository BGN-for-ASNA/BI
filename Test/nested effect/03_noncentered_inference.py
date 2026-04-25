# %%
from BI import bi
import jax.numpy as jnp
import numpy as np
import io
from contextlib import redirect_stdout

# Setup device
m = bi(platform="cpu")

# Load Data
data = np.load("simulated_data.npz")
N_regions = int(data["N_regions"])
group_id = data["group_id"]
x = data["x"]
y = data["y"]
group_to_region = data["group_to_region"]

N_groups = len(np.unique(group_id))

m.data_on_model = {
    "group_id": group_id,
    "x": x,
    "y": y,
    "group_to_region": group_to_region,
    "N_regions": N_regions,
    "N_groups": N_groups,
}


# Define model
def model_nested_noncentered(group_id, x, y, group_to_region, N_regions, N_groups):
    sigma = m.dist.exponential(1, name="sigma")

    # 1. Region level
    mu_a_reg = m.dist.normal(5, 2, name="mu_a_reg")
    mu_b_reg = m.dist.normal(-1, 1, name="mu_b_reg")
    sigma_reg = m.dist.exponential(1, shape=(2,), name="sigma_reg")
    L_reg = m.dist.lkj_cholesky(2, 2, name="L_reg")

    z_reg = m.dist.normal(0, 1, name="z_reg", shape=(2, N_regions))
    effects_deviation_reg = ((sigma_reg[..., None] * L_reg) @ z_reg).T

    region_effects = jnp.stack([mu_a_reg, mu_b_reg]) + effects_deviation_reg

    # 2. Group level
    sigma_grp = m.dist.exponential(1, shape=(2,), name="sigma_grp")
    L_grp = m.dist.lkj_cholesky(2, 2, name="L_grp")

    z_grp = m.dist.normal(0, 1, name="z_grp", shape=(2, N_groups))
    effects_deviation_grp = ((sigma_grp[..., None] * L_grp) @ z_grp).T

    # Nested mean
    mu_grp = region_effects[group_to_region]

    group_effects = mu_grp + effects_deviation_grp

    a_g_est = group_effects[:, 0]
    b_g_est = group_effects[:, 1]

    # Likelihood
    mu_est = a_g_est[group_id] + b_g_est[group_id] * x
    m.dist.normal(mu_est, sigma, obs=y)


# Run sampler
m.fit(
    model_nested_noncentered,
    num_samples=1000,
    num_warmup=500,
    num_chains=1,
    progress_bar=True,
)

# Diagnostic
with redirect_stdout(io.StringIO()):
    m.summary()

# Read true params
true_params = {}
with open("true_params.txt", "r") as f:
    for line in f:
        k, v = line.strip().split(": ")
        true_params[k] = float(v)

print("\n### Parameter Comparison")
print(f"{'Parameter':<15} | {'True':<10} | {'Posterior Mean':<15}")
print("-" * 45)
params_to_check = {
    "mu_a_reg": true_params["mu_a_reg"],
    "mu_b_reg": true_params["mu_b_reg"],
    "sigma_reg[0]": true_params["sigma_reg_a"],
    "sigma_reg[1]": true_params["sigma_reg_b"],
    "sigma": true_params["sigma"],
}

for p, true_val in params_to_check.items():
    if p in m.tab_summary.index:
        post_mean = m.tab_summary.loc[p, "mean"]
        print(f"{p:<15} | {true_val:<10.2f} | {post_mean:<15.2f}")
    elif p.replace("[0]", "[v0]") in m.tab_summary.index:
        post_mean = m.tab_summary.loc[p.replace("[0]", "[v0]"), "mean"]
        print(
            f"{p.replace('[0]', '[v0]'):<15} | {true_val:<10.2f} | {post_mean:<15.2f}"
        )

# %%
