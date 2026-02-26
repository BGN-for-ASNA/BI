library(reticulate)
library(BayesianInference)

jnp <- import("jax.numpy")
jax_scipy <- import("jax.scipy.special")

bi_model_duration <- function(data) {
  # Edge weights prior
  edge_weight <- m$dist$normal(data$prior_edge_mu, data$prior_edge_sigma, shape=tuple(data$num_edges), name="edge_weight")
  
  # Additional rate parameter for duration models
  rate_param <- m$dist$normal(0.0, data$prior_rate_sigma, shape=tuple(data$num_edges), name="rate")
  rate_positive <- jnp$abs(rate_param) # or exponential depending on Stan implementation, Stan says normal(0, sigma) with lower=0. Half-normal.
  
  dyad_ids_0 <- data$dyad_ids - 1L
  predictor <- edge_weight[dyad_ids_0]
  
  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(data$prior_fixed_mu, data$prior_fixed_sigma, shape=tuple(data$num_fixed), name="beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }

  p <- jax_scipy$expit(predictor)
  
  # event ~ exponential(rate[dyad_ids] ./ inv_logit(predictor));
  # event_count ~ poisson(rate .* divisor);
  
  lambda_exp <- rate_positive[dyad_ids_0] / p
  lambda_pois <- rate_positive * data$divisor
  
  m$dist$exponential(lambda_exp, obs=data$event, name="event")
  m$dist$poisson(lambda_pois, obs=data$event_count, name="event_count")
}
