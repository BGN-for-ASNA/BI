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

m.data_on_model = {
    "group_id": group_id,
    "x": x,
    "y": y,
    "group_to_region": group_to_region,
    "N_regions": N_regions,
}


# Define model
def model_nested(group_id, x, y, group_to_region, N_regions):
    sigma = m.dist.exponential(1, name="sigma")

    # 1. Region level
    mu_a_reg = m.dist.normal(5, 2, name="mu_a_reg")
    mu_b_reg = m.dist.normal(-1, 1, name="mu_b_reg")
    sigma_reg = m.dist.exponential(1, shape=(2,), name="sigma_reg")
    Rho_reg = m.dist.lkj(2, 2, name="Rho_reg")
    cov_reg = jnp.outer(sigma_reg, sigma_reg) * Rho_reg

    region_effects = m.dist.multivariate_normal(
        jnp.stack([mu_a_reg, mu_b_reg]),
        cov_reg,
        shape=(N_regions,),
        name="region_effects",
    )

    # 2. Group level
    sigma_grp = m.dist.exponential(1, shape=(2,), name="sigma_grp")
    Rho_grp = m.dist.lkj(2, 2, name="Rho_grp")
    cov_grp = jnp.outer(sigma_grp, sigma_grp) * Rho_grp

    # Nested mean
    mu_grp = region_effects[group_to_region]

    group_effects = m.dist.multivariate_normal(mu_grp, cov_grp, name="group_effects")

    a_g_est = group_effects[:, 0]
    b_g_est = group_effects[:, 1]

    # Likelihood
    mu_est = a_g_est[group_id] + b_g_est[group_id] * x
    m.dist.normal(mu_est, sigma, obs=y)


# Run sampler
m.fit(model_nested, num_samples=1000, num_warmup=500, num_chains=1, progress_bar=True)

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
    elif (
        p.replace("[0]", "[v0]") in m.tab_summary.index
    ):  # Arviz format sometimes uses [v0]
        post_mean = m.tab_summary.loc[p.replace("[0]", "[v0]"), "mean"]
        print(
            f"{p.replace('[0]', '[v0]'):<15} | {true_val:<10.2f} | {post_mean:<15.2f}"
        )
# %%
