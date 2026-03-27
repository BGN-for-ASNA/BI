library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

# Mock data
N <- 10
phen <- jnp$zeros(N)
cofactor <- jnp$zeros(N)
phylo_idx <- jnp$array(rep(0L:9L, length.out = N))
A_chol <- jnp$eye(N)

m$data_on_model <- list(
  phen = phen,
  cofactor = cofactor,
  phylo_idx = phylo_idx,
  A_cholesky = A_chol,
  num_species = as.integer(N)
)

model <- function(phen, cofactor, phylo_idx, A_cholesky, num_species) {
  intercept <- m$dist$normal(0, 1, name = "Intercept")
  beta <- m$dist$normal(0, 1, name = "beta")

  # Normal half
  sd_p <- m$dist$half_normal(1.0, name = "sd_p")

  # Use user suggestion: jnp$array([10], dtype=jnp.int32)
  z_p <- m$dist$normal(jnp$zeros(jnp$array(c(10L), dtype = jnp$int32)), 1.0, name = "z_p")
  u_p <- jnp$matmul(A_cholesky, z_p) * sd_p

  mu <- intercept + beta * cofactor + jnp$take(u_p, phylo_idx)

  m$dist$normal(mu, 1.0, name = "obs", obs = phen)
}

m$fit(model, num_samples = 10L, num_warmup = 10L)
