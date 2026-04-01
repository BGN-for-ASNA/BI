# %%
import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BI import bi

# Initialize BI with float64
m = bi("cpu")

# Load data
data_slopes = pd.read_table("data_slopes.txt", sep=r"\s+")
N = len(data_slopes)

# Load Cholesky of A
L_df = pd.read_csv("L_slopes.csv")
L_A = L_df.values
N_species = L_A.shape[0]

# Mapping species to indices
matrix_species = [f"sp_{i+1}" for i in range(N_species)]
species_to_idx = {sp: i for i, sp in enumerate(matrix_species)}
data_slopes["phylo_idx"] = data_slopes["phylo"].map(species_to_idx)

# Prepare data for BI
m.data_on_model = {
    "y": jnp.array(data_slopes["y"].values, dtype=jnp.float64),
    "x": jnp.array(data_slopes["x"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_slopes["phylo_idx"].values, dtype=jnp.int32),
    "A_cholesky": jnp.array(L_A, dtype=jnp.float64),
}


def model(y, x, phylo_idx, A_cholesky):
    # Population-level effects
    intercept = m.dist.normal(0.0, 10.0, name="Intercept")
    b_x = m.dist.normal(0.0, 10.0, name="b_x")

    # Residual error
    sigma = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 5.2, create_obj=True), low=0.0, name="sigma"
    )

    # Group-level parameters
    sd_intercept = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 10, create_obj=True), low=0.0, name="sd_intercept"
    )
    sd_slope = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 10, create_obj=True), low=0.0, name="sd_slope"
    )
    sd_1 = jnp.array([sd_intercept, sd_slope])

    # Cholesky factor of correlation matrix
    L_corr = m.dist.lkj_cholesky(2, concentration=2.0, name="L_corr")

    # Standardized species effects (N_species x 2)
    z_1 = m.dist.normal(jnp.zeros((N_species, 2)), 1.0, name="z_1")

    # Compose group-level effects
    # L_sigma = diag(sd_1) @ L_corr
    L_sigma = jnp.diag(sd_1) @ L_corr

    # U = (L_A @ z_1) @ L_sigma.T
    U = jnp.matmul(jnp.matmul(A_cholesky, z_1), L_sigma.T)

    u_intercept = U[:, 0]
    u_slope = U[:, 1]

    # Linear predictor
    mu = intercept + u_intercept[phylo_idx] + (b_x + u_slope[phylo_idx]) * x

    # Likelihood
    m.dist.normal(mu, sigma, name="Y", obs=y)


# Fit model
print("Fitting BI Model 6 (Varying Slopes)...")
m.fit(model, num_samples=1000, num_warmup=1000, num_chains=2)

# Summary
print(m.summary())

# Posteriors
post = m.posteriors
# Parameters of interest: Intercept, b_x, sigma, sd_1[0], sd_1[1], correlation
cor_mat = jnp.matmul(post["L_corr"], jnp.transpose(post["L_corr"], (0, 2, 1)))
rho = cor_mat[:, 0, 1]

post_df = pd.DataFrame(
    {
        "Intercept": np.array(post["Intercept"]).flatten(),
        "b_x": np.array(post["b_x"]).flatten(),
        "sigma": np.array(post["sigma"]).flatten(),
        "sd_intercept": np.array(post["sd_intercept"]).flatten(),
        "sd_slope": np.array(post["sd_slope"]).flatten(),
        "rho": np.array(rho).flatten(),
    }
)

post_df.to_csv("bi_post_slopes.csv", index=False)
print("Results saved to bi_post_slopes.csv")

# %%
