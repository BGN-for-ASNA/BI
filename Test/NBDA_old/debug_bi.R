library(reticulate)
library(BayesianInference)
library(STbayes)
m <- importBI("cpu")
jnp <- import("jax.numpy")
np  <- import("numpy")
source("BI_Models/model_cTADA.R")
to_jax <- function(x) { if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x); jnp$array(x) }
dl <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
K <- dl$K; P <- dl$P
obs_end_time <- matrix(0,K,P); is_event <- matrix(FALSE,K,P); valid_ind <- matrix(1,K,P)
for (k in 1:K) {
  for (n in 1:dl$N[k]) { id <- dl$ind_id[k,n]
    if (dl$t[k,id] > 0) { obs_end_time[k,id] <- dl$t[k,id]; is_event[k,id] <- TRUE } else valid_ind[k,id] <- 0 }
  if (dl$N_c[k] > 0) for (c in 1:dl$N_c[k]) { id <- dl$ind_id[k, dl$N[k]+c]; obs_end_time[k,id] <- dl$T[k] }
}
jax <- lapply(dl, to_jax)
jax$obs_end_time <- jnp$array(obs_end_time); jax$is_event <- jnp$array(is_event); jax$valid_ind <- jnp$array(valid_ind)
m$data_on_model <- list(data = jax)

m$fit(bi_model_cTADA, num_chains=1L, num_warmup=100L, num_samples=200L)

raw <- m$posteriors
cat("m$posteriors names:", paste(names(raw), collapse=", "), "\n")

# Check log_lambda_0_mean
lam <- raw[["log_lambda_0_mean"]]
cat("\nlog_lambda_0_mean Python class:", class(lam), "\n")
cat("log_lambda_0_mean type:", py_str(lam), "\n")

# Try np$asarray
np_lam <- np$asarray(lam)
cat("np_lam class:", class(np_lam), "\n")
r_lam <- py_to_r(np_lam)
cat("r_lam class:", class(r_lam), "\n")
cat("r_lam length:", length(as.numeric(r_lam)), "\n")
cat("r_lam head:", head(as.numeric(r_lam), 5), "\n")
