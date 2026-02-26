# BI Model for Individual-Level Variable (ILV) Extentions
bi_model_ILV <- function(data) {
  # STEP 1: Define Model Parameters
  # ILVs models introduce external dataset factors into baseline/social effects.
  log_lambda_0_mean <- m$dist$normal(-4, 2, name="log_lambda_0_mean")
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  
  # Define the Beta priors for ILV effects based on data type (boolean, continuous, categorical)
  beta_ILVi_bool_ILV <- m$dist$normal(0, 1, shape=tuple(1L), name="beta_ILVi_bool_ILV")
  beta_ILVs_cont_ILV <- m$dist$normal(0, 1, name="beta_ILVs_cont_ILV")
  beta_ILVm_cat_ILV <- m$dist$normal(0, 1, shape=tuple(3L), name="beta_ILVm_cat_ILV")
  
  lambda_0 <- jnp$exp(log_lambda_0_mean)
  s_prime <- jnp$exp(log_s_prime_mean)
  
  # STEP 2: Applying Evaluated ILV Effects
  # Vectorized computation of beta application over P individuals
  # - boolean trait mapping
  bool_ILV_i <- jnp$matmul(data$ILV_bool_ILV, beta_ILVi_bool_ILV)
  # - continuous scaling (weight * beta)
  cont_ILV_s <- data$ILV_cont_ILV * beta_ILVs_cont_ILV
  # - categorical encoding transformation
  cat_ILV_m <- jnp$matmul(data$ILV_cat_ILV, beta_ILVm_cat_ILV)
  
  # Generate the unique intrinsic probability multiplier for all P individuals
  ind_term <- jnp$exp(bool_ILV_i) # shape (P,)
  
  # STEP 3: Evaluating Social Diffusion Matrix
  # A: (N_networks, K, T_max, P, P) -> (K, T_max, P, P)
  A_sum <- jnp$sum(data$A, axis=0L)
  
  # Expand knowledge State Z: (K, T_max, P) -> (K, T_max, P, 1)
  Z_exp <- jnp$expand_dims(data$Z, axis=-1L)
  
  # Social connections to knowledgeable peers: A_Z (K, T_max, P)
  A_Z <- jnp$squeeze(jnp$matmul(A_sum, Z_exp), axis=-1L)
  
  # Compute base social connectivity factor modified by continuous individual traits
  # soc_term uses broadcasting of s_prime (scalar), A_Z (K, T_max, P), and trait modulators exp(cont_ILV_s) (P,)
  soc_term <- s_prime * A_Z * jnp$exp(cont_ILV_s)
  
  # Expand Duration D: (K, T_max) -> (K, T_max, 1)
  D_exp <- jnp$expand_dims(data$D, axis=-1L)
  
  # STEP 4: Lambda Application (Network Effect + ILV modification)
  # The unified lambda (hazard rate) scales with categorical traits across combinations of social and intrinsic learning.
  # lambda_all: shape (K, T_max, P)
  lambda_all <- jnp$exp(cat_ILV_m) * (lambda_0 * ind_term + soc_term) * D_exp
  
  # STEP 5: Vectorized Survival Likelihood Calculation (cTADA-style)
  # Create a time grid (1, T_max, 1) and observed end time bounds
  T_max <- data$T_max
  time_grid <- jnp$arange(T_max)
  time_grid_exp <- jnp$expand_dims(jnp$expand_dims(time_grid, axis=0L), axis=-1L) # (1, T_max, 1)
  obs_end_time_exp <- jnp$expand_dims(data$obs_end_time, axis=1L) # (K, 1, P)
  
  # sum_mask applies to all time_steps prior to the end_time where the individual is naive
  sum_mask <- time_grid_exp < obs_end_time_exp
  # log_mask isolates the exact specific time step that an event actually occurred at (excluding censored cases)
  log_mask <- (time_grid_exp == (obs_end_time_exp - 1L)) & jnp$expand_dims(data$is_event, axis=1L)
  
  # Compute specific log-likelihood components
  neg_lambda_sum <- jnp$sum(-lambda_all * sum_mask, axis=1L) # sum hazard, shape: (K, P)
  log_lambda_term <- jnp$sum(jnp$log(lambda_all) * log_mask, axis=1L) # capture event odds, shape: (K, P)
  
  log_lik_ind <- neg_lambda_sum + log_lambda_term
  
  # STEP 6: Execute Model Target Target Summation (Unit Distribution)
  total_log_lik <- jnp$sum(log_lik_ind * data$valid_ind) 
  m$dist$unit(log_factor=total_log_lik, name="ILV_lik")
}

print("ILV vectorized model compiled.")
