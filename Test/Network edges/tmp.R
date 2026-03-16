bi_model_binary <- function(data) {
  # Edge weights prior
  edge_weight <- m$dist$normal(data$prior_edge_mu, data$prior_edge_sigma, shape = tuple(data$num_edges), name = "edge_weight")

  # 0-based indexing for JAX
  dyad_ids_0 <- data$dyad_ids - 1L

  # Build predictor
  predictor <- edge_weight[dyad_ids_0]

  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(data$prior_fixed_mu, data$prior_fixed_sigma, shape = tuple(data$num_fixed), name = "beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }

  # For simplicity, ignoring partial pooling and random effects if num_random == 0,
  # but in a complete port we would add them here.

  p <- jax_scipy$expit(predictor)

  m$dist$binomial(data$divisor, p, obs = data$event, name = "event")
}
