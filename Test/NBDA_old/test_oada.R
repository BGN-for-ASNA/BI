# Load all R functions from STbayes manually
r_files <- list.files("STbayes_repo/R", pattern="\\.R$", full.names=TRUE)
r_files <- r_files[!grepl("RcppExports\\.R", r_files)]
for (f in r_files) source(f)
library(reticulate)
library(BayesianInference)

m <- importBI("cpu")
jnp <- import("jax.numpy")

# source the model definition
source("BI_Models/model_OADA.R")

load("STbayes_repo/data/event_data.rda")
load("STbayes_repo/data/edge_list.rda")

# Format data
data_list <- import_user_STb(event_data, edge_list)

# We need to map the data_list properties to arrays for JAX
# Let's build a processed data list with trial_idx, time_idx, ind_idx, event_mask
K <- data_list$K
Q <- data_list$Q
P <- data_list$P

trial_idx <- c()
time_idx <- c()
ind_idx <- c()

# Note: R is 1-indexed, Python/JAX is 0-indexed!!
for (k in 1:K) {
  for (n in 1:data_list$N[k]) {
    id <- data_list$ind_id[k, n]
    learn_time <- data_list$t[k, id]
    if (learn_time > 0) {
      # subtract 1 for 0-indexing!
      trial_idx <- c(trial_idx, k - 1)
      time_idx <- c(time_idx, learn_time - 1)
      ind_idx <- c(ind_idx, id - 1)
    }
  }
}

# Add these directly to data_list for JAX
data_list$trial_idx <- as.integer(trial_idx)
data_list$time_idx <- as.integer(time_idx)
data_list$ind_idx <- as.integer(ind_idx)
data_list$event_mask <- rep(TRUE, length(trial_idx)) # all these are valid events!

# For JAX mapping, we also need to convert Matrices/Arrays to jax arrays 
# and subtract 1 from elements where we use them as indices if STbayes gave them 1-indexed.
# Wait, A, Z, etc. are just values, so they don't need -1.
# Let's create a wrapper function to convert list elements to JAX arrays:
to_jax <- function(x) {
  if (is.list(x) || type(x) == "closure") {
    return(x)
  }
  return(jnp$array(x))
}

data_list_jax <- lapply(data_list, to_jax)

# Attach data to model
m$data_on_model <- list(data = data_list_jax)

# Test if it runs correctly!
print("Starting fit...")
m$fit(bi_model_OADA)

print("Fit completed.")
