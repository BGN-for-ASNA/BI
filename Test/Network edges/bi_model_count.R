library(reticulate)
library(BayesianInference)

m <- importBI("cpu")

jnp <- import("jax.numpy")
jax_scipy <- import("jax.scipy.special")

bi_model_count <- function(data) {
  predictor <- jnp$zeros(as.integer(data$num_rows))

  if (as.numeric(data$num_edges) > 0) {
    if (as.numeric(data$partial_pooling) == 0) {
      edge_weight <- m$dist$normal(0.0, 2.0, shape = tuple(data$num_edges), name = "edge_weight")
    } else {
      edge_sigma <- m$dist$half_normal(2.0, name = "edge_sigma")
      edge_weight <- m$dist$normal(0.0, edge_sigma, shape = tuple(data$num_edges), name = "edge_weight")
    }

    # 0-based indexing for JAX
    dyad_ids_0 <- data$dyad_ids - 1L
    predictor <- predictor + edge_weight[dyad_ids_0]
  }

  if (as.numeric(data$num_fixed) > 0) {
    beta_fixed <- m$dist$normal(0.0, 1.0, shape = tuple(data$num_fixed), name = "beta_fixed")
    predictor <- predictor + jnp$dot(data$design_fixed, beta_fixed)
  }

  if (as.numeric(data$num_random) > 0) {
    # Extract group ID from design_random matrix
    # Assumes design_random is a one-hot matrix (N_obs, N_groups)
    group_ids_0 <- jnp$argmax(data$design_random, axis = 1L)

    if (as.numeric(data$num_random) == 1L) {
      varying_intercepts <- m$effects$varying_intercept(
        N_groups = as.integer(data$num_random_groups),
        group_id = group_ids_0,
        a_bar = m$dist$normal(0.0, 1.0, shape = tuple(1L), name = "random_group_mu_new"),
        sigma = m$dist$exponential(1.0, shape = tuple(1L), name = "random_group_sigma_new"),
        group_name = "node_id"
      )
      predictor <- predictor + varying_intercepts
    } else {
      # Use varying_effects if we have more than 1 effect (intercept + slopes)
      out <- m$effects$varying_effects(
        N_vars = as.integer(data$num_random) - 1L,
        N_group = as.integer(data$num_random_groups),
        group_id = group_ids_0,
        group_name = "node_id",
        alpha_bar = m$dist$normal(0.0, 1.0, shape = tuple(1L), name = "random_group_mu_new"),
        sd_intercept = m$dist$exponential(1.0, shape = tuple(1L), name = "random_group_sigma_new")
      )
      predictor <- predictor + out[[1]]
    }
  }

  rate <- jnp$exp(predictor) * data$divisor

  if (as.numeric(data$zero_inflated) == 1) {
    zero_prob <- m$dist$beta(1.0, 1.0, shape = tuple(1L), name = "zero_prob")
    base_dist <- m$dist$poisson(rate, create_obj = TRUE)
    m$dist$zero_inflated_distribution(base_dist, gate = zero_prob[0L], obs = data$event, name = "event")
  } else {
    m$dist$poisson(rate, obs = data$event, name = "event")
  }
}
