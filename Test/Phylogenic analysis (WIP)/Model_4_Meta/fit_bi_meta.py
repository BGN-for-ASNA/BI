# %%
import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BI import bi

# Initialize BI with float64
jax.config.update("jax_enable_x64", True)
m = bi("cpu")

# Load data
url = "https://paul-buerkner.github.io/data/data_effect.txt"
data_fisher = pd.read_table(url, sep="\s+", header=0, names=["Zr", "N", "phylo"])

data_fisher["obs_idx"] = np.arange(len(data_fisher))
data_fisher["se"] = np.sqrt(1.0 / (data_fisher["N"].values - 3.0))

# Load Cholesky of A
L_df = pd.read_csv("L_meta.csv")
L = L_df.values
matrix_species = L_df.columns.tolist()
species_to_idx = {sp: i for i, sp in enumerate(matrix_species)}

data_fisher["phylo_idx"] = data_fisher["phylo"].map(species_to_idx)

# Fixed SE
data_fisher["se"] = np.sqrt(1.0 / (data_fisher["N"] - 3.0))

# Prepare data for BI
m.data_on_model = {
    "Zr": jnp.array(data_fisher["Zr"].values, dtype=jnp.float64),
    "se": jnp.array(data_fisher["se"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_fisher["phylo_idx"].values, dtype=jnp.int32),
    "obs_idx": jnp.array(data_fisher["obs_idx"].values, dtype=jnp.int32),
    "A_cholesky": jnp.array(L, dtype=jnp.float64),
}


def model(Zr, se, phylo_idx, obs_idx, A_cholesky):
    # Priors
    intercept = m.dist.normal(0.0, 10.0, name="Intercept")

    # Hyperparameters
    sd_obs = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 10, create_obj=True), low=0.0, name="sd_obs"
    )
    sd_phylo = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 10, create_obj=True), low=0.0, name="sd_phylo"
    )

    # Effects
    z_phylo = m.dist.normal(jnp.zeros(200), 1.0, name="z_phylo")
    u_phylo = jnp.matmul(A_cholesky, z_phylo) * sd_phylo

    z_obs = m.dist.normal(jnp.zeros(len(Zr)), 1.0, name="z_obs")
    u_obs = z_obs * sd_obs

    # Mean
    mu = intercept + u_phylo[phylo_idx] + u_obs[obs_idx]

    # Likelihood (with fixed SE)
    m.dist.normal(mu, se, name="Y", obs=Zr)


# Fit model
print("Fitting BI meta-analysis model...")
m.fit(model, num_samples=3000, num_warmup=2000, num_chains=2)

# Summary
print(m.summary())

# Posteriors
post = m.posteriors
params_of_interest = ["Intercept", "sd_obs", "sd_phylo"]
post_df = pd.DataFrame({k: np.array(post[k]).flatten() for k in params_of_interest})
post_df.to_csv("bi_post_meta.csv", index=False)

# %%
