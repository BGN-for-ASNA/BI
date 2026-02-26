library(reticulate)
library(BayesianInference)
library(STbayes)

m <- importBI("cpu")
jnp <- import("jax.numpy")

source("BI_Models/model_veff.R")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  return(jnp$array(x))
}

data_list <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
K <- data_list$K
P <- data_list$P
T_max <- max(data_list$T)

obs_end_time <- matrix(0, nrow=K, ncol=P)
is_event     <- matrix(FALSE, nrow=K, ncol=P)
valid_ind    <- matrix(1, nrow=K, ncol=P)

for (k in 1:K) {
  for (n in 1:data_list$N[k]) {
    id <- data_list$ind_id[k, n]
    if (data_list$t[k, id] > 0) {
      obs_end_time[k, id] <- data_list$t[k, id]
      is_event[k, id] <- TRUE
    } else {
      valid_ind[k, id] <- 0
    }
  }
  if (data_list$N_c[k] > 0) {
    for (c in 1:data_list$N_c[k]) {
      id <- data_list$ind_id[k, data_list$N[k] + c]
      obs_end_time[k, id] <- data_list$T[k]
    }
  }
}

jax <- lapply(data_list, to_jax)
jax$obs_end_time <- jnp$array(obs_end_time)
jax$is_event     <- jnp$array(is_event)
jax$valid_ind    <- jnp$array(valid_ind)

m$data_on_model <- list(data = jax)

cat("Fitting veff model (shape from data$Z$shape)...\n")
t0 <- proc.time()
fit <- m$fit(bi_model_veff, num_chains=2L, num_warmup=500L, num_samples=500L)
elapsed <- (proc.time() - t0)["elapsed"]
cat("Done in", elapsed, "seconds.\n")
print(m$summary(fit))
