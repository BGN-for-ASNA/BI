# BF Model for Complex f Frequency-Dependent Transmission Networks
BF_model_complex_f <- function(data) {
  # STEP 1: Setting up Primary Factor Models
  # Log space sampling of fundamental intrinsic and transmission values
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  # f modifier controlling power bias towards higher frequencies 
  log_f_mean <- m$dist$normal(0, 1, name="log_f_mean")
  
  lambda_0 <- jnp$exp(log_lambda_0_mean)
  s_prime <- jnp$exp(log_s_prime_mean)
  f <- jnp$exp(log_f_mean)
  
  # STEP 2: Network Alignment mapping associations vs transmission capabilities
  # A tracks relationships shaped (N_networks, K, T_max, P, P)
  A <- data$A
  
  # Z records occurrences: (K, T_max, P). Expand to vector shapes (1, K, T_max, P, 1)
  Z_exp <- jnp$expand_dims(jnp$expand_dims(data$Z, axis=-1L), axis=0L)
  
  # Zn identifies available uninformed candidates 
  Zn_exp <- jnp$expand_dims(jnp$expand_dims(data$Zn, axis=-1L), axis=0L)
  Zn_complement <- 1.0 - Zn_exp
  
  # Summate the count of neighbors exhibiting behavior
  active <- jnp$squeeze(jnp$matmul(A, Z_exp), axis=-1L)
  
  # Summate the count of ignorant neighbors
  inactive <- jnp$squeeze(jnp$matmul(A, Zn_complement), axis=-1L)
  
  # STEP 3: Fraction Mapping based on sampled 'f' frequency response constraint
  # Complex f scales both active/inactive occurrences by f probability factors
  active_f <- jnp$power(active, f)
  inactive_f <- jnp$power(inactive, f)
  denom_f <- active_f + inactive_f
  
  # Compute scaled likelihood fraction protecting against 0/0 exceptions
  frac <- jnp$where((active + inactive) > 0, active_f / denom_f, 0.0)
  
  # Calculate social coefficient accumulated across multiple separate network boundaries 
  # Shape transforms to (K, T_max, P)
  net_effect <- jnp$sum(frac, axis=0L)
  soc_term <- s_prime * net_effect
  
  # Match arrays with timed observational weights
  D_exp <- jnp$expand_dims(data$D, axis=-1L)
  lambda_all <- (lambda_0 + soc_term) * D_exp
  
  # STEP 4: Vectorized Survival Likelihood Aggregation 
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L) 
  
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L) 
  
  sum_mask <- time_grid_exp < obs_end_time_exp
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  # Vectorized summations
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) 
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) 
  
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  
  # Register the completed array values internally to the NumPyro Unit density module
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  m$dist$unit(log_factor=total_log_lik, name="complex_f_lik")
}

print("complex_f vectorized model compiled.")
