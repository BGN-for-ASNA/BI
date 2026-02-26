library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

# BI Model for OADA (Order of Acquisition Data Analysis)
bi_model_OADA <- function(data) {
  # STEP 1: Define Model Parameters
  # We extract the base log learning rate modifiers from normal distributions.
  # The parameters are mapped from STbayes 'parameters' block.
  log_s_prime_mean <- m$dist$normal(-4, 2, name="log_s_prime_mean")
  s_prime <- jnp$exp(log_s_prime_mean) # Exponentiated to ensure strictly positive rate
  
  # STEP 2: Pre-process Network and State Data
  # A: (N_networks, K, T_max, P, P) -> Summed across multiple network layers into (K, T_max, P, P)
  # This merges multiple networks (like 'assoc', 'kin', etc.) if present.
  A_sum <- jnp$sum(data$A, axis=0L)
  
  # Z: Binary matrix (K, T_max, P) indicating if individual `p` has acquired the behavior at time `t`.
  # Expanded to (K, T_max, P, 1) for JAX broadcasting dot products.
  Z_exp <- jnp$expand_dims(data$Z, axis=-1L)
  
  # STEP 3: Vectorized Network Effect Calculation
  # We multiply the adjacency matrix by the knowledge state to find how many *informed* connections each node has.
  # jnp$matmul handles the (P, P) x (P, 1) dot product smoothly over the leading K and T_max dimensions.
  # Resulting A_Z shape: (K, T_max, P)
  A_Z <- jnp$squeeze(jnp$matmul(A_sum, Z_exp), axis=-1L)
  
  # Total social transmission rate at every time step
  net_effect_all <- s_prime * A_Z
  # Assume base intrinsic lambda_0 is 1.0 for generic OADA (without explicitly estimating lambda_0)
  lambda_all <- 1.0 + net_effect_all
  
  # STEP 4: Mask Target Events for Likelihood Computation
  # OADA calculates likelihood *only* at the specific timesteps when diffusion events occur.
  # We execute this entirely via shape-agnostic masking to accommodate NumPyro sampling batches
  is_event_3d <- data$is_event_3d # (K, T_max, P)
  event_at_time <- data$event_at_time # (K, T_max)
  
  # i_lambda isolates the rate for the *specific* individual that learned
  i_lambda <- jnp$sum(lambda_all * is_event_3d, axis=-1L) # shape (*batch, K, T_max)
  
  # STEP 5: Calculate Denominator (Total hazard of all naive individuals)
  # Q_mask protects padded entries explicitly vs true counts (Q)
  Q_mask <- jnp$arange(data$P) < data$Q
  
  # j_rates calculates lambda for individuals who haven't learned (1.0 - Z) and are valid (Q_mask).
  j_rates <- lambda_all * (1.0 - data$Z) * Q_mask
  j_rates_sum_all <- jnp$sum(j_rates, axis=-1L) # Sum of all active risks: (*batch, K, T_max)
  
  # STEP 6: Compute Final Event Log-Likelihood
  # Inject safe values to prevent negative log(0) NaN propagation across empty frames
  i_lambda_safe <- jnp$where(event_at_time, i_lambda, 1.0)
  j_rates_sum_safe <- jnp$where(event_at_time, j_rates_sum_all, 1.0)
  
  log_lik <- jnp$log(i_lambda_safe) - jnp$log(j_rates_sum_safe)
  
  # Silence the calculation for padding/invalid events
  log_lik_masked <- jnp$where(event_at_time, log_lik, 0.0)
  
  # Sum total log probabilities and register as unit distribution for NumPyro sampling
  total_log_lik <- jnp$sum(log_lik_masked)
  m$dist$unit(log_factor=total_log_lik, name="OADA_lik")
}

print("OADA vectorized model compiled.")
