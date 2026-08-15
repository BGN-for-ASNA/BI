# %%
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BayesForge import bf

# Enable float64 for numerical parity
m = bf("cpu")

# Load data
data_simple = pd.read_table("data_simple.txt", sep=r"\s+")
mean_cofactor = data_simple["cofactor"].mean()
data_simple["cofactor_centered"] = data_simple["cofactor"] - mean_cofactor

# Map species factors to indices
L_df = pd.read_csv("L_simple.csv")
L = L_df.values
data_simple["phylo_idx"] = pd.Categorical(
    data_simple["phylo"], categories=L_df.columns
).codes

# Prepare data for BF
m.data_on_model = {
    "phen": jnp.array(data_simple["phen"].values, dtype=jnp.float64),
    "cofactor": jnp.array(data_simple["cofactor_centered"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_simple["phylo_idx"].values, dtype=jnp.int32),
    "A_cholesky": jnp.array(L, dtype=jnp.float64),
}


def model(phen, cofactor, phylo_idx, A_cholesky):
    # Priors - Aligned with brms student_t(3, 0, 20)
    intercept = m.dist.normal(0, 50, name="Intercept")
    b_cofactor = m.dist.normal(0, 10, name="b_cofactor")

    sd_phylo = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 20, create_obj=True), low=0.0, name="sd_phylo"
    )
    sigma = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 20, create_obj=True), low=0.0, name="sigma"
    )

    # Phylogenetic effects
    z_phylo = m.dist.normal(jnp.zeros(200), 1.0, name="z_phylo")
    u_phylo = jnp.matmul(A_cholesky, z_phylo) * sd_phylo

    # Mean
    mu = intercept + b_cofactor * cofactor + u_phylo[phylo_idx]

    # Likelihood
    m.dist.normal(mu, sigma, name="obs", obs=phen)


# Fit model
print("Fitting BF Simple Model...")
m.fit(model, num_samples=2000, num_warmup=1000, num_chains=2)

# Recover uncentered intercept
post = m.posteriors
post["b_Intercept"] = post["Intercept"] - mean_cofactor * post["b_cofactor"]

# Save posteriors
params = ["b_Intercept", "b_cofactor", "sd_phylo", "sigma"]
post_df = pd.DataFrame({k: np.array(post[k]).flatten() for k in params})
post_df.to_csv("BF_post_simple.csv", index=False)
print("Results saved to BF_post_simple.csv")

# %%
