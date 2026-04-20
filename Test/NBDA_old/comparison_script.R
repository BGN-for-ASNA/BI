# Comparison Script for STbayes and BI R (cTADA Model)

# 1. Source required libraries and STbayes internal functions
library(reticulate)
library(BayesianInference)

cat("Loading STbayes resources...\n")
library(STbayes)

# 2. Load the dummy datasets from STbayes
event_data <- STbayes::event_data
edge_list <- STbayes::edge_list

cat("Formatting data...\n")
# 3. Format data using STbayes internal tool
# This generates 'data_list' with all required properties for cTADA
data_list <- import_user_STb(event_data, edge_list, network_type = "undirected")

# 4. Initialize BI and JAX
cat("Initializing BayesInference backend...\n")
m <- importBI("cpu")
jnp <- import("jax.numpy")

# Source our translated model
source("BI_Models/model_cTADA.R")

# 5. Adapt STbayes data for our vectorized JAX cTADA model
K <- data_list$K
P <- data_list$P

# We need obs_end_time (K, P)
obs_end_time <- matrix(0, nrow = K, ncol = P)
is_event <- matrix(FALSE, nrow = K, ncol = P)
valid_ind <- matrix(1, nrow = K, ncol = P) # 1 if observed, 0 if left censored

# Fill in values
for (k in 1:K) {
  # valid events
  for (n in 1:data_list$N[k]) {
    id <- data_list$ind_id[k, n]
    # Check if pre-trained (time = 0)
    if (data_list$t[k, id] > 0) {
      obs_end_time[k, id] <- data_list$t[k, id]
      is_event[k, id] <- TRUE
    } else {
      valid_ind[k, id] <- 0
    }
  }
  # censored individuals
  if (data_list$N_c[k] > 0) {
    for (c in 1:data_list$N_c[k]) {
      id <- data_list$ind_id[k, data_list$N[k] + c]
      obs_end_time[k, id] <- data_list$T[k]
      is_event[k, id] <- FALSE
    }
  }
}

# Add vectors to data_list for JAX (converting types)
to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) {
    return(x)
  }
  return(jnp$array(x))
}

data_list_jax <- lapply(data_list, to_jax)
data_list_jax$obs_end_time <- jnp$array(obs_end_time)
data_list_jax$is_event <- jnp$array(is_event)
data_list_jax$valid_ind <- jnp$array(valid_ind)

cat("Fitting BI cTADA model using NumPyro NUTS...\n")
# 6. Fit the model using BI!
m$data_on_model <- list(data = data_list_jax)

# Start sampling (2 chains, low warmup/samples for demonstration)
fit <- m$fit(bi_model_cTADA, num_chains = 2L, num_warmup = 500L, num_samples = 500L)

cat("\n--- BI R Model Fitting Complete ---\n")
bi_summary <- m$summary(fit)
print(bi_summary)

cat("\n============================================\n")
cat("Fitting STbayes model via cmdstanr...\n")
model_code <- generate_STb_model(data_list, gq = FALSE, est_acqTime = FALSE)
write(model_code, file="temp_model.stan")

fit_stb <- fit_STb(data_list,
                   model_obj = "temp_model.stan",
                   parallel_chains = 2,
                   chains = 2,
                   iter = 1000,
                   refresh=500)

cat("\n--- STbayes Model Fitting Complete ---\n")
stb_summary <- STb_summary(fit_stb, digits=3)
print(stb_summary)

cat("\nDONE: Comparison Script executed successfully.\n")
