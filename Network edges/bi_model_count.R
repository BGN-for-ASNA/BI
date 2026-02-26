library(reticulate)
library(BayesianInference)

jnp <- import("jax.numpy")
jax_scipy <- import("jax.scipy.special")

bi_model_count <- function(data) {
  # Edge weights prior
  edge_weight <- m$dist$normal(data$prior_edge_mu, data$prior_edge_sigma, shape=tuple(data$num_edges), name="edge_weight")
  
  dyad_ids_0 <- data$dyad_ids - 1L
  predictor <- edge_weight[dyad_ids_0]
  
  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(data$prior_fixed_mu, data$prior_fixed_sigma, shape=tuple(data$num_fixed), name="beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }
  
  rate <- jnp$exp(predictor) * data$divisor
  
  m$dist$poisson(rate, obs=data$event, name="event")
}
