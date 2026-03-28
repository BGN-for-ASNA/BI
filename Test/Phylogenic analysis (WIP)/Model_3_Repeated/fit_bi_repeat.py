import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BI import bi

# Initialize BI
# Initialize BI with float64
# Note: BI might need jax_enable_x64=True
jax.config.update("jax_enable_x64", True)
m = bi("cpu")

# Load data
data_repeat = pd.read_table("data_repeat.txt", sep=r"\s+")
# Calculate spec_mean_cf
spec_mean_cf_map = data_repeat.groupby("species")["cofactor"].mean()
data_repeat["spec_mean_cf"] = data_repeat["species"].map(spec_mean_cf_map)

# Centering
mean_spec_mean_cf = data_repeat["spec_mean_cf"].mean()
data_repeat["spec_mean_cf_centered"] = data_repeat["spec_mean_cf"] - mean_spec_mean_cf

# Map species factors to indices
# IMPORTANT: Indices must match the order of species in the matrix L
L_df = pd.read_csv("L_repeat.csv")
L = L_df.values
matrix_species = L_df.columns.tolist()
species_to_idx = {sp: i for i, sp in enumerate(matrix_species)}

data_repeat["phylo_idx"] = data_repeat["phylo"].map(species_to_idx)
data_repeat["species_idx"] = data_repeat["species"].map(species_to_idx)
# Since we already ran R, let's create a small R script to save L to a file.
# OR, use a simpler approach if L is already available.
# Actually, I'll update get_stan_repeat.R to save the Cholesky factor L.

# For now, let's assume L is saved as L_repeat.csv
if not os.path.exists("L_repeat.csv"):
    import subprocess
    with open("save_L.R", "w") as f:
        f.write('library(ape)\nphylo <- read.nexus("phylo.nex")\nA <- vcv.phylo(phylo)\nL <- t(chol(A))\nwrite.csv(L, "L_repeat.csv", row.names=FALSE)\n')
    subprocess.run(["C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe", "save_L.R"])

L = pd.read_csv("L_repeat.csv").values

# Prepare data for BI
m.data_on_model = {
    "phen": jnp.array(data_repeat["phen"].values, dtype=jnp.float64),
    "spec_mean_cf": jnp.array(data_repeat["spec_mean_cf_centered"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_repeat["phylo_idx"].values, dtype=jnp.int32),
    "species_idx": jnp.array(data_repeat["species_idx"].values, dtype=jnp.int32),
    "A_cholesky": jnp.array(L, dtype=jnp.float64),
}

def model(phen, spec_mean_cf, phylo_idx, species_idx, A_cholesky):
    # Priors
    intercept = m.dist.normal(0, 50, name="Intercept")
    b_spec_mean_cf = m.dist.normal(0, 10, name="b_spec_mean_cf")

    # Hyperparameters
    sd_phylo = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 20, create_obj=True),
        low=0.0, name="sd_phylo"
    )
    sd_species = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 20, create_obj=True),
        low=0.0, name="sd_species"
    )
    sigma = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 20, create_obj=True),
        low=0.0, name="sigma"
    )

    # Phylogenetic effects
    z_phylo = m.dist.normal(jnp.zeros(200), 1.0, name="z_phylo")
    u_phylo = jnp.matmul(A_cholesky, z_phylo) * sd_phylo

    # Species-specific effects (indep of phylogeny)
    z_species = m.dist.normal(jnp.zeros(200), 1.0, name="z_species")
    u_species = z_species * sd_species

    # Mean
    mu = intercept + b_spec_mean_cf * spec_mean_cf + \
         u_phylo[phylo_idx] + u_species[species_idx]

    # Likelihood
    m.dist.normal(mu, sigma, name="obs", obs=phen)

# Fit model
print("Fitting BI model...")
m.fit(model, num_samples=3000, num_warmup=2000, num_chains=2)

# Summary
print(m.summary())

# Recover uncentered intercept
post = m.posteriors
post["b_Intercept"] = post["Intercept"] - mean_spec_mean_cf * post["b_spec_mean_cf"]

# Parameters of interest for comparison
params_of_interest = ["b_Intercept", "b_spec_mean_cf", "sd_phylo", "sd_species", "sigma"]

# Save posteriors for comparison
post_df = pd.DataFrame({k: np.array(post[k]).flatten() for k in params_of_interest})
post_df.to_csv("bi_post_repeat.csv", index=False)
