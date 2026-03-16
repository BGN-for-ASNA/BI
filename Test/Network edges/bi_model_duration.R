library(reticulate)
library(BayesianInference)

m <- importBI("cpu")

jnp <- import("jax.numpy")
jax_scipy <- import("jax.scipy.special")

bi_model_duration <- function(data) {
  predictor <- jnp$zeros(as.integer(data$num_rows))

  if (as.numeric(data$partial_pooling) == 0) {
    edge_weight <- m$dist$normal(0.0, 2.0, shape = tuple(data$num_edges), name = "edge_weight")
  } else {
    edge_sigma <- m$dist$half_normal(2.0, name = "edge_sigma")
    edge_weight <- m$dist$normal(0.0, edge_sigma, shape = tuple(data$num_edges), name = "edge_weight")
  }

  # Additional rate parameter for duration models
  # Stan says normal(0, sigma) with lower=0. Half-normal.
  rate_positive <- m$dist$half_normal(1.0, shape = tuple(data$num_edges), name = "rate")

  dyad_ids_0 <- data$dyad_ids - 1L
  predictor <- predictor + edge_weight[dyad_ids_0]

  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(0.0, 1.0, shape = tuple(data$num_fixed), name = "beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }

  if (as.numeric(data$num_random) > 0) {
    random_group_mu <- m$dist$normal(0.0, 1.0, shape = tuple(data$num_random_groups), name = "random_group_mu")
    random_group_sigma <- m$dist$half_normal(1.0, shape = tuple(data$num_random_groups), name = "random_group_sigma")

    group_idx_0 <- data$random_group_index - 1L
    beta_random <- m$dist$normal(random_group_mu[group_idx_0], random_group_sigma[group_idx_0], shape = tuple(data$num_random), name = "beta_random")

    predictor <- predictor + jnp$dot(data$design_random, beta_random)
  }

  p <- m$link$inv_logit(predictor)

  lambda_exp <- rate_positive[dyad_ids_0] / p
  lambda_pois <- rate_positive * data$divisor

  m$dist$exponential(lambda_exp, obs = data$event, name = "event")
  m$dist$poisson(lambda_pois, obs = data$event_count, name = "event_count")
}
