# %%
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
import numpy as np
import jax.numpy as jnp
from BI import bi

# Initialize BI with float64
m = bi("cpu")

# Load data
data_slopes = pd.read_table("data_slopes.txt", sep=r"\s+")
N = len(data_slopes)

# Mapping species to indices
N_species = data_slopes["phylo"].nunique()
matrix_species = [f"sp_{i+1}" for i in range(N_species)]
species_to_idx = {sp: i for i, sp in enumerate(matrix_species)}
data_slopes["phylo_idx"] = data_slopes["phylo"].map(species_to_idx)

# Center x for better recovery
x_mean = data_slopes["x"].mean()
data_slopes["x_centered"] = data_slopes["x"] - x_mean

# Prepare data for BI
m.data_on_model = {
    "y": jnp.array(data_slopes["y"].values, dtype=jnp.float64),
    "x": jnp.array(data_slopes["x_centered"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_slopes["phylo_idx"].values, dtype=jnp.int32),
    "N_species": N_species,
}


def model(y, x, phylo_idx, N_species):
    # Residual error
    sigma = m.dist.exponential(1.0, name="sigma")

    # High-level API for varying effects (Non-centered)
    # Note: This approach currently assumes independent groups (no phylogeny A)
    varying_intercept, varying_slope = m.effects.varying_effects(
        N_vars=1,
        N_group=N_species,
        group_id=phylo_idx,
        group_name="phylo",
        centered=False,
    )

    # varying_slope is (N, 1), so we squeeze or index [:, 0]
    mu = varying_intercept + varying_slope[:, 0] * x

    # Likelihood
    m.dist.normal(mu, sigma, name="Y", obs=y)


# Fit model
print("Fitting BI Model 5 (Varying Slopes) using effects.varying_effects API...")
print("Note: This specific implementation assumes independent groups (Standard MLM).")
m.fit(model, num_samples=2000, num_warmup=2000, num_chains=2)

# Summary
print(m.summary())

# Posteriors
post = m.posteriors
# In varying_effects, correlation is saved as {group_name}_L_corr
L_corr = post["phylo_L_corr"]
cor_mat = jnp.matmul(L_corr, jnp.transpose(L_corr, (0, 2, 1)))
rho = cor_mat[:, 0, 1]

post_df = pd.DataFrame(
    {
        "Intercept": np.array(post["global_intercept"]).flatten(),
        "b_x": np.array(post["global_beta"]).flatten(),
        "sigma": np.array(post["sigma"]).flatten(),
        "sd_intercept": np.array(post["phylo_sd_intercept"]).flatten(),
        "sd_slope": np.array(post["phylo_sd_beta"]).flatten(),
        "rho": np.array(rho).flatten(),
    }
)

post_df.to_csv("bi_post_slopes_effects.csv", index=False)
print("Results saved to bi_post_slopes_effects.csv")

# %%
