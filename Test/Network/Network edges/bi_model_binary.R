library(reticulate)
library(BayesianInference)

m <- importBI("cpu")

# Ensure JAX is available and uses 64-bit precision to match Stan
jax <- import("jax")
jax$config$update("jax_enable_x64", TRUE)
jnp <- import("jax.numpy")
jax_scipy <- import("jax.scipy.special")

bi_model_binary <- function(data) {
  predictor <- jnp$zeros(as.integer(data$num_rows))

  if (as.numeric(data$partial_pooling) == 0) {
    edge_weight <- m$dist$normal(as.numeric(data$prior_edge_mu), as.numeric(data$prior_edge_sigma), shape = tuple(data$num_edges), name = "edge_weight")
  } else {
    edge_sigma <- m$dist$half_normal(as.numeric(data$prior_edge_sigma), name = "edge_sigma")
    edge_weight <- m$dist$normal(as.numeric(data$prior_edge_mu), edge_sigma, shape = tuple(data$num_edges), name = "edge_weight")
  }

  # 0-based indexing for JAX
  dyad_ids_0 <- data$dyad_ids - 1L
  predictor <- predictor + edge_weight[dyad_ids_0]


  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(as.numeric(data$prior_fixed_mu), as.numeric(data$prior_fixed_sigma), shape = tuple(data$num_fixed), name = "beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }

  if (as.numeric(data$num_random) > 0) {
    random_group_mu <- m$dist$normal(as.numeric(data$prior_random_mean_mu), as.numeric(data$prior_random_mean_sigma), shape = tuple(data$num_random_groups), name = "random_group_mu")
    # Half normal for random group sigma — scale from prior_random_std_sigma (matches Stan: random_group_sigma ~ normal(0, prior_random_std_sigma))
    random_group_sigma <- m$dist$half_normal(as.numeric(data$prior_random_std_sigma), shape = tuple(data$num_random_groups), name = "random_group_sigma")

    group_idx_0 <- data$random_group_index - 1L
    beta_random <- m$dist$normal(random_group_mu[group_idx_0], random_group_sigma[group_idx_0], shape = tuple(data$num_random), name = "beta_random")

    predictor <- predictor + jnp$dot(data$design_random, beta_random)
  }

  p <- m$link$inv_logit(predictor)

  if (as.numeric(data$zero_inflated) == 1) {
    zero_prob <- m$dist$beta(as.numeric(data$prior_zero_prob_alpha), as.numeric(data$prior_zero_prob_beta), shape = tuple(1L), name = "zero_prob")
    base_dist <- m$dist$binomial(data$divisor, p, create_obj = TRUE)
    m$dist$zero_inflated_distribution(base_dist, gate = zero_prob[0L], obs = data$event, name = "event")
  } else {
    m$dist$binomial(data$divisor, p, obs = data$event, name = "event")
  }
}
