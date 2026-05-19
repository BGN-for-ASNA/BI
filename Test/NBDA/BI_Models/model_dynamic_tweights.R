# BI Model for Dynamic Networks with Dynamic Transmission Weights (dini functional mapping)
bi_model_dynamic_networks_dynamic_tweights <- function(data) {
  # STEP 1: Setting up core intrinsic vs social base factors
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  # k_raw governs the curve shape (k) in the dini functional transformation
  k_raw <- m$dist$normal(0, 3, name="k_raw")
  
  lambda_0 <- jnp$exp(log_lambda_0_mean)
  s_prime <- jnp$exp(log_s_prime_mean)
  
  # Map unbounded k_raw sampling input to standard bound interval (-1, 1) using a sigmoid curve
  jax_nn <- import("jax.nn")
  k_shape <- 2.0 * jax_nn$sigmoid(k_raw) - 1.0
  
  # STEP 2: The Dini Functional Representation mapping transmission based on local proportion
  dini_func <- function(x, k) {
    x_transformed <- 2.0 * x - 1.0
    y <- ((x_transformed - k * x_transformed) / (k - 2.0 * k * jnp$abs(x_transformed) + 1.0) + 1.0) / 2.0
    return(y)
  }
  
  # STEP 3: Mapping Local Network Dynamics
  # A is shaped (N_networks, K, T_max, P, P) capturing moving structures at differing time instances.
  A <- data$A
  
  # Z indicates active individuals who attained the targeted behavior: (K, T_max, P) -> (1, K, T_max, P, 1)
  Z_exp <- jnp$expand_dims(jnp$expand_dims(data$Z, axis=-1L), axis=0L)
  
  # Zn indicates total relevant interactions
  Zn_exp <- jnp$expand_dims(jnp$expand_dims(data$Zn, axis=-1L), axis=0L)
  Zn_complement <- 1.0 - Zn_exp # Derive uninvolved contacts
  
  # Multiply to derive the number of transmitting/active neighbors per node: (N_networks, K, T_max, P)
  numer <- jnp$squeeze(jnp$matmul(A, Z_exp), axis=-1L)
  
  # Number of inactive networks
  denom_part2 <- jnp$squeeze(jnp$matmul(A, Zn_complement), axis=-1L)
  
  # Calculate overall local connections
  denom <- numer + denom_part2
  
  # STEP 4: Network Impact Derivation 
  # Compute the relative proportion of exhibiting active connections vs total local connections
  # Prevent division by zero
  prop <- jnp$where(denom > 0, numer / denom, 0.0)
  
  # Transmitting effect transformation via evaluated k-shape sampled priors 
  dini_transformed <- dini_func(prop, k_shape)
  
  # Determine overall connection multiplier summated across varying network classifications
  net_effect <- jnp$sum(dini_transformed, axis=0L)
  
  # soc_term: Total social connectivity strength factoring the transmission modifier function
  soc_term <- s_prime * net_effect
  
  # Adjust by event observation period (K, T_max) -> (K, T_max, 1)
  D_exp <- jnp$expand_dims(data$D, axis=-1L)
  lambda_all <- (lambda_0 + soc_term) * D_exp
  
  # STEP 5: Vectorized Likelihood Calculation based on generic cTADA structure
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L) # (1, T_max, 1)
  
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L) # (K, 1, P)
  
  sum_mask <- time_grid_exp < obs_end_time_exp
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  # Vector operations calculating accumulated risk without action (-lambda sum) vs executed events (+log lambda)
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) # (K, P)
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) # (K, P)
  
  # Individual evaluations
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  
  # Compile combined estimation across verified individuals only 
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  m$dist$unit(log_factor=total_log_lik, name="dynamic_networks_tweights_lik")
}

print("dynamic_networks_dynamic_tweights vectorized model compiled.")
