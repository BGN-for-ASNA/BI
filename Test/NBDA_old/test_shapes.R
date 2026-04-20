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
m$fit(bi_model_cTADA, num_chains=2L, num_warmup=10L, num_samples=20L)

raw <- m$posteriors
for (nm in names(raw)) {
  if (grepl("_lik$", nm)) next
  jax_arr <- raw[[nm]]
  np_arr  <- np$asarray(jax_arr)
  r_arr <- py_to_r(np_arr)
  cat(nm, "dims:", paste(dim(r_arr), collapse="x"), "\n")
}
