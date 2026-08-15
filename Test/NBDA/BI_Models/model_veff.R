# BF Model for varying-effects (veff) formulation
BF_model_veff <- function(data) {
  # STEP 1: Setting up main factors
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  # STEP 2: Establishing Random Effects
  # This model includes random effects for s_prime (social) and lambda_0 (intrinsic)
  # at the individual level. Each individual gets their own offset drawn from a
  # zero-mean normal with estimated standard deviations.
  #
  # KEY: We derive P from data$Z$shape (the static .shape attribute) rather than
  # data$P. Even though data$P and data$Z are both JAX arrays, their .shape attributes
  # are always concrete Python integers known at JIT-compile time — never traced values.
  # This is what allows shape=tuple(P) to work correctly inside the NumPyro NUTS tracer.
  #
  # data$Z has shape (K, T_max, P), so:
  #   shape index 0 = K (trials)
  #   shape index 1 = T_max
  #   shape index 2 = P (individuals)   <-- this is what we need
  P <- data$Z$shape[[3]]  # R 1-indexed: [[3]] is Python index 2
  
  sigma_sprime <- m$dist$half_normal(1.0, name="sigma_sprime")  # SD for s_prime offsets
  sigma_lambda0 <- m$dist$half_normal(1.0, name="sigma_lambda0") # SD for lambda_0 offsets
  
  # Individual-level offsets: shape (P,)
  v_sprime <- m$dist$normal(0.0, sigma_sprime, shape=tuple(P), name="v_sprime")
  v_lambda0 <- m$dist$normal(0.0, sigma_lambda0, shape=tuple(P), name="v_lambda0")
  
  # Apply random effects to get individual-specific rates
  s_prime <- jnp$exp(log_s_prime_mean + v_sprime)   # shape (P,) - individual social rates
  lambda_0 <- jnp$exp(log_lambda_0_mean + v_lambda0) # shape (P,) - individual intrinsic rates

  # STEP 3: Vectorized Network Connection Evaluation
  # Merge network layers into base view A (K, T_max, P, P)
  A_sum <- jnp$sum(data$A, axis=0L)
  
  # State matrix Z expanded to multiply networks indicating informed neighbors
  Z_exp <- jnp$expand_dims(data$Z, axis=-1L)
  A_Z <- jnp$squeeze(jnp$matmul(A_sum, Z_exp), axis=-1L)
  
  # Broadcasting individually varied s_prime rates across connections
  # s_prime shape (P,) broadcasts against A_Z shape (K, T_max, P)
  soc_term <- s_prime * A_Z
  
  # Time Step Weight adjustment (K, T_max) -> (K, T_max, 1)
  D_exp <- jnp$expand_dims(data$D, axis=-1L)
  
  # Resolve to base individual hazard: lambda_all shape (K, T_max, P)
  # lambda_0 shape (P,) broadcasts correctly
  lambda_all <- (lambda_0 + soc_term) * D_exp
  
  # STEP 4: Vectorized Survival Likelihood Calculation (cTADA-style)
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L) # (1, T_max, 1)
  
  # Align with individual timeline
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L) # (K, 1, P)
  
  sum_mask <- time_grid_exp < obs_end_time_exp
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  # Summate the prior hazard -lambda log weights
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) # (K, P)
  
  # Identify likelihood on successful event period
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) # (K, P)
  
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  
  # Register the resulting cumulative probabilities with BF framework
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  m$dist$unit(log_factor=total_log_lik, name="veff_lik")
}

print("veff vectorized model compiled.")
