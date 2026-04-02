# %%
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BI import bi

# Enable float64 for numerical parity
jax.config.update("jax_enable_x64", True)
m = bi("cpu")

# Load data
data_pois = pd.read_table("data_pois.txt", sep=r"\s+")
data_pois["obs_idx"] = range(len(data_pois))
mean_cofactor = data_pois["cofactor"].mean()
data_pois["cofactor_centered"] = data_pois["cofactor"] - mean_cofactor

# Map species factors to indices
L_df = pd.read_csv("L_pois.csv")
L = L_df.values
data_pois["phylo_idx"] = pd.Categorical(
    data_pois["phylo"], categories=L_df.columns
).codes

# Prepare data for BI
m.data_on_model = {
    "phen": jnp.array(data_pois["phen_pois"].values, dtype=jnp.int32),
    "cofactor": jnp.array(data_pois["cofactor_centered"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_pois["phylo_idx"].values, dtype=jnp.int32),
    "obs_idx": jnp.array(data_pois["obs_idx"].values, dtype=jnp.int32),
    "A_cholesky": jnp.array(L, dtype=jnp.float64),
}


def model(phen, cofactor, phylo_idx, obs_idx, A_cholesky):
    # Priors - Aligned with brms
    intercept = m.dist.student_t(3, 0.3, 2.6, name="Intercept")
    b_cofactor = m.dist.normal(0, 10, name="b_cofactor")

    sd_phylo = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 2.6, create_obj=True), low=0.0, name="sd_phylo"
    )
    sd_obs = m.dist.left_truncated_distribution(
        m.dist.student_t(3, 0, 2.6, create_obj=True), low=0.0, name="sd_obs"
    )

    # Phylogenetic effects
    z_phylo = m.dist.normal(jnp.zeros(200), 1.0, name="z_phylo")
    u_phylo = jnp.matmul(A_cholesky, z_phylo) * sd_phylo

    # Observation-level random effects
    z_obs = m.dist.normal(jnp.zeros(200), 1.0, name="z_obs")
    u_obs = z_obs * sd_obs

    # Mean
    mu = intercept + b_cofactor * cofactor + u_phylo[phylo_idx] + u_obs[obs_idx]

    # Likelihood
    m.dist.poisson(jnp.exp(mu), name="obs", obs=phen)


# Fit model
print("Fitting BI Poisson Model...")
m.fit(model, num_samples=2000, num_warmup=1000, num_chains=2)

# Recover uncentered intercept
post = m.posteriors
post["b_Intercept"] = post["Intercept"] - mean_cofactor * post["b_cofactor"]

# Save posteriors
params = ["b_Intercept", "b_cofactor", "sd_phylo", "sd_obs"]
post_df = pd.DataFrame({k: np.array(post[k]).flatten() for k in params})
post_df.to_csv("bi_post_pois.csv", index=False)
print("Results saved to bi_post_pois.csv")

# %%
