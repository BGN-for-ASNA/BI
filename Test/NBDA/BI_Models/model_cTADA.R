# BI Model for cTADA (Continuous Time of Acquisition Data Analysis)
bi_model_cTADA <- function(data) {
  # STEP 1: Define Model Parameters
  # cTADA models both a baseline intrinsic learning rate (lambda_0) and a social transmission rate (s_prime).
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  # Convert from log scale
  lambda_0 <- jnp$exp(log_lambda_0_mean)
  s_prime <- jnp$exp(log_s_prime_mean)
  
  # STEP 2: Aggregate Networks and Calculate Social Effect
  # A: (N_networks, K, T_max, P, P) -> Sum over network dimension -> (K, T_max, P, P)
  A_sum <- jnp$sum(data$A, axis=0L)
  
  # Expand knowledge state Z (K, T_max, P) for vectorized matrix multiplication
  Z_exp <- jnp$expand_dims(data$Z, axis=-1L)
  
  # Perform vectorized network multiplication to count active, informed connections per individual
  A_Z <- jnp$squeeze(jnp$matmul(A_sum, Z_exp), axis=-1L)
  
  # Social rate term representing the influence of informed neighbors
  soc_term <- s_prime * A_Z
  
  # STEP 3: Time-Varying Covariates and Total Lambda
  # D is the matrix of durations or time steps (K, T_max). Expand for broadcasting.
  D_exp <- jnp$expand_dims(data$D, axis=-1L)
  
  # Combine intrinsic and social rates, modulated by the time duration D
  lambda_all <- (lambda_0 + soc_term) * D_exp
  
  # STEP 4: Vectorized Survival Likelihood Calculation
  # In cTADA, an individual survives without learning (hazard accumulates) until an event occurs, or they are right-censored.
  
  # Create a time grid (1, T_max, 1) and observed end time structure (K, 1, P)
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L)
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L)
  
  # Mask out time steps before the individual actually learned or was censored
  # sum_mask applies to all time_steps prior to the end_time where the individual is naive
  sum_mask <- time_grid_exp < obs_end_time_exp
  
  # log_mask isolates the exact specific time step that an event actually occurred at (excluding censored cases)
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  # STEP 5: Compute Log-Likelihood
  # Individuals accumulate negative hazard over all time periods prior to learning
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) # sum over time_steps, resulting in shape (K, P)
  
  # Individuals log the likelihood of actually learning during their event time period
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) # shape (K, P)
  
  # Sum the hazard component and the event probability component
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  
  # Mask out padded invalid individuals and sum across entire population/trials
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  
  # Inject target log likelihood into the numPyro sampling graph
  m$dist$unit(log_factor=total_log_lik, name="cTADA_lik")
}

print("cTADA vectorized model compiled.")
