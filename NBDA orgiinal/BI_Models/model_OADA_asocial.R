# BI Model for OADA Asocial (Order of Acquisition Data Analysis, No Social Learning)
bi_model_OADA_asocial <- function(data) {
  # STEP 1: No Parameters Evaluated
  # In a purely asocial OADA model, there is no generic intrinsic rate estimated directly or social rate
  # since all individuals learn asocially at a constant rate that cancels out when normalized by order.
  
  # Extract event masking objects instead of indices
  is_event_3d <- data$is_event_3d # (K, T_max, P)
  event_at_time <- data$event_at_time # (K, T_max)
  
  # Mask for valid individuals present in the trial (j < Q) vs padded (P)
  Q_mask <- jnp$arange(data$P) < data$Q
  
  # STEP 3: Denominator - Baseline Hazard
  # Because the models assumes constant rate across individuals (i_lambda = 1.0),
  # the denominator is simply the count of individuals who haven't learned yet (1.0 - Z)
  j_rates <- (1.0 - data$Z) * Q_mask
  j_rates_sum_all <- jnp$sum(j_rates, axis=-1L)
  
  # STEP 4: Event Log-likelihood
  # i_lambda is 1.0, so log(i_lambda) is 0. 
  # This makes log(i_lambda / j_rates_sum_all) equal to -log(j_rates_sum_all)
  j_rates_sum_safe <- jnp$where(event_at_time, j_rates_sum_all, 1.0)
  log_lik <- -jnp$log(j_rates_sum_safe)
  
  # Silence the calculation for padding/invalid trial events using the pre-calculated mask
  log_lik_masked <- jnp$where(event_at_time, log_lik, 0.0)
  
  # STEP 5: Register the Total Probability
  total_log_lik <- jnp$sum(log_lik_masked)
  m$dist$unit(log_factor=total_log_lik, name="OADA_asocial_lik")
}

print("OADA asocial vectorized model compiled.")
