# BF Model for deriving edge connection strengths from posterior probability models
BF_model_posterior_edges <- function(data) {
  # STEP 1: Setting up core intrinsic vs social base factors
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  lambda_0 <- jnp$exp(log_lambda_0_mean)
  s_prime <- jnp$exp(log_s_prime_mean)
  
  # STEP 2: Estimating Edges from Posteriors
  # Instead of relying strictly on an observed (A) network matrix, the model samples varying 
  # logit weights for specific node dyads mapping unobserved dyadic associations.
  # data$logit_edge_mu should be (N_networks, N_dyad)
  # data$logit_edge_cov should be (N_networks, N_dyad, N_dyad)
  edge_logit <- m$dist$multivariate_normal(
    loc = data$logit_edge_mu,
    covariance_matrix = data$logit_edge_cov,
    name = "edge_logit"
  )
  
  # w: the generated unconstrained network connection probabilities: (N_networks, N_dyad)
  jax_nn <- import("jax.nn")
  w <- jax_nn$sigmoid(edge_logit)
  
  # Initialize the empty matrix array to construct adjacency graphs from sampled edge logs
  # A_init: (N_networks, P, P)
  A_init <- jnp$zeros(tuple(data$N_networks, data$P, data$P))
  
  # Using JAX advanced indexing to set symmetric undirected edges across the network dimension
  # Align network index
  network_idx <- jnp$arange(data$N_networks)[, jnp$newaxis] # (N_networks, 1)
  
  # JAX 0-indexed translation! R traditionally passes base-1 arrays.
  focal_idx <- data$focal_ID - 1L
  other_idx <- data$other_ID - 1L
  
  # Populate both bidirectional halves of the association matrix using the evaluated dyad samples `w`
  A <- A_init$at[network_idx, focal_idx, other_idx]$set(w)
  A <- A$at[network_idx, other_idx, focal_idx]$set(w)
  
  # STEP 3: Map the custom networks into diffusion probability vectors
  # A_sum: (P, P) -- collapse the dimension over constructed networks
  A_sum <- jnp$sum(A, axis=0L)
  
  # Expand A_sum to broadcast alongside Z dimensional shapes: (K, T_max, P) -> (1, 1, P, P)
  A_sum_exp <- jnp$expand_dims(jnp$expand_dims(A_sum, axis=0L), axis=0L) 
  
  # Multiply by known knowledge states across history Z: (K, T_max, P) -> (K, T_max, P, 1)
  Z_exp <- jnp$expand_dims(data$Z, axis=-1L)
  A_Z <- jnp$squeeze(jnp$matmul(A_sum_exp, Z_exp), axis=-1L)
  
  soc_term <- s_prime * A_Z
  
  # Adjust by time interval weighting 
  D_exp <- jnp$expand_dims(data$D, axis=-1L)  # D: (K, T_max) -> (K, T_max, 1)
  lambda_all <- (lambda_0 + soc_term) * D_exp
  
  # STEP 4: Base Vectorized Survival Likelihood Calculation (cTADA core logic)
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L) # (1, T_max, 1)
  
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L) # (K, 1, P)
  
  sum_mask <- time_grid_exp < obs_end_time_exp
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) # Evaluate cumulative hazard pre-event: shape (K, P)
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) # Probability of event: shape (K, P)
  
  # Consolidate estimates for individual occurrences across sample frames
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  m$dist$unit(log_factor=total_log_lik, name="posterior_edges_lik")
}

print("posterior_edges vectorized model compiled.")
