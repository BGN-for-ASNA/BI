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


# Define model (non-centered built-in)
def model_nested_builtin_noncentered(
    group_id, x, y, group_to_region, N_regions, N_groups
):
    sigma = m.dist.exponential(1, name="sigma")

    levels = [
        {"name": "region", "N_groups": N_regions, "parent_id": None},
        {"name": "group", "N_groups": N_groups, "parent_id": group_to_region},
    ]

    a_g_est, b_g_est = m.effects.nested_varying_effects(
        N_vars=2, levels=levels, group_id=group_id, centered=False
    )

    # Likelihood
    mu_est = a_g_est + b_g_est * x
    m.dist.normal(mu_est, sigma, obs=y)


# Define model (centered built-in)
def model_nested_builtin_centered(group_id, x, y, group_to_region, N_regions, N_groups):
    sigma = m.dist.exponential(1, name="sigma")

    levels = [
        {"name": "region", "N_groups": N_regions, "parent_id": None},
        {"name": "group", "N_groups": N_groups, "parent_id": group_to_region},
    ]

    a_g_est, b_g_est = m.effects.nested_varying_effects(
        N_vars=2, levels=levels, group_id=group_id, centered=True
    )

    # Likelihood
    mu_est = a_g_est + b_g_est * x
    m.dist.normal(mu_est, sigma, obs=y)


# Read true params
true_params = {}
with open("true_params.txt", "r") as f:
    for line in f:
        k, v = line.strip().split(": ")
        true_params[k] = float(v)

params_to_check = {
    "global_intercept[0]": true_params["mu_a_reg"],
    "global_beta[0]": true_params["mu_b_reg"],
    "sigma_region[0]": true_params["sigma_reg_a"],
    "sigma_region[1]": true_params["sigma_reg_b"],
    "sigma": true_params["sigma"],
}


def evaluate_model(model_func, title):
    print(f"\nEvaluating: {title}")
    m.fit(model_func, num_samples=1000, num_warmup=500, num_chains=1, progress_bar=True)
    with redirect_stdout(io.StringIO()):
        m.summary()
    print(f"\n### Parameter Comparison ({title})")
    print(f"{'Parameter':<20} | {'True':<10} | {'Posterior Mean':<15}")
    print("-" * 50)

    for p, true_val in params_to_check.items():
        if p in m.tab_summary.index:
            post_mean = m.tab_summary.loc[p, "mean"]
            print(f"{p:<20} | {true_val:<10.2f} | {post_mean:<15.2f}")
        elif p.replace("[0]", "[v0]") in m.tab_summary.index:
            post_mean = m.tab_summary.loc[p.replace("[0]", "[v0]"), "mean"]
            print(
                f"{p.replace('[0]', '[v0]'):<20} | {true_val:<10.2f} | {post_mean:<15.2f}"
            )


evaluate_model(model_nested_builtin_noncentered, "Non-centered (Built-in Mode)")
evaluate_model(model_nested_builtin_centered, "Centered (Built-in Mode)")

# %%
