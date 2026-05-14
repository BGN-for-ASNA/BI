.libPaths(c("/home/sebastian_sosa/R/x86_64-pc-linux-gnu-library/4.6", .libPaths()))
library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

source("Test/Network/Network edges/bi_model_duration.R")

# Dummy data
data <- list(
  num_rows = 10L,
  num_edges = 2L,
  num_fixed = 1L,
  num_random = 0L,
  num_random_groups = 0L,
  num_fixed_cov = 1L,
  num_random_cov = 0L,
  event = jnp$ones(10L),
  event_count = jnp$array(as.integer(c(5, 5)), dtype=jnp$int32),
  divisor = jnp$ones(10L),
  dyad_ids = jnp$array(c(1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L), dtype=jnp$int32),
  design_fixed = jnp$ones(as.integer(c(10, 1))),
  design_random = jnp$zeros(as.integer(c(10, 0))),
  random_group_index = jnp$zeros(0L, dtype=jnp$int32),
  partial_pooling = 0L,
  zero_inflated = 0L,
  prior_edge_mu = 0.0,
  prior_edge_sigma = 1.0,
  prior_fixed_mu = 0.0,
  prior_fixed_sigma = 1.0,
  prior_rate_sigma = 1.0,
  prior_zero_prob_alpha = 1.0,
  prior_zero_prob_beta = 1.0,
  model_type = "duration"
)

m$data_on_model <- list(data = data)
cat("Fitting...\n")
m$fit(bi_model_duration, num_warmup = 100L, num_samples = 100L, num_chains = 1L)
cat("Names of posteriors:", paste(names(m$posteriors), collapse=", "), "\n")

post <- m$posteriors
for (nm in names(post)) {
  arr <- post[[nm]]
  cat(nm, "shape:", paste(arr$shape, collapse="x"), "\n")
}
