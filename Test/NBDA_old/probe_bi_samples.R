library(reticulate)
library(BayesianInference)
library(STbayes)
m <- importBI("cpu")
jnp <- import("jax.numpy")
source("BI_Models/model_cTADA.R")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  jnp$array(x)
}
dl <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
K <- dl$K; P <- dl$P; T_max <- max(dl$T)
obs_end_time <- matrix(0, K, P); is_event <- matrix(FALSE, K, P); valid_ind <- matrix(1, K, P)
for (k in 1:K) {
  for (n in 1:dl$N[k]) {
    id <- dl$ind_id[k,n]
    if (dl$t[k,id] > 0) { obs_end_time[k,id] <- dl$t[k,id]; is_event[k,id] <- TRUE } else valid_ind[k,id] <- 0
  }
  if (dl$N_c[k] > 0) for (c in 1:dl$N_c[k]) { id <- dl$ind_id[k, dl$N[k]+c]; obs_end_time[k,id] <- dl$T[k] }
}
jax <- lapply(dl, to_jax)
jax$obs_end_time <- jnp$array(obs_end_time); jax$is_event <- jnp$array(is_event); jax$valid_ind <- jnp$array(valid_ind)

m$data_on_model <- list(data = jax)
fit <- m$fit(bi_model_cTADA, num_chains=1L, num_warmup=50L, num_samples=100L)

cat("Class of fit:", class(fit), "\n")
cat("Type of fit:", typeof(fit), "\n")

# Check the m object itself for get_samples or similar
cat("\nm methods:\n")
bi_doc <- tryCatch(m$get_samples, error=function(e) e$message)
cat("m$get_samples:", class(bi_doc), "\n")

# the fit itself is a Python object; inspect it
cat("\nfit Python type:", py_str(fit), "\n")

# Try accessing posterior_samples on the BI object m
cat("\nm$mcmc_obj:\n")
mcmc <- tryCatch(m$mcmc_obj, error=function(e) { cat("no mcmc_obj\n"); NULL })

# The BayesianInference fit object may store the arviz trace
cat("\nfit$idata:\n")
idat <- tryCatch(fit$idata, error=function(e) { cat("no idata:", e$message, "\n"); NULL })
if (!is.null(idat)) cat("idata type:", class(idat), "\n")

# Try direct draw extraction through m
cat("\npy_to_r(fit):\n")
r_fit <- tryCatch(py_to_r(fit), error=function(e) { cat("py_to_r failed:", e$message, "\n"); NULL })
if (!is.null(r_fit)) { cat("names:", names(r_fit), "\n") }

# Check what reticulate sees as attributes
cat("\nreticulate py_list_attributes:\n")
attrs <- tryCatch(py_list_attributes(fit), error=function(e) { cat("Error:", e$message, "\n"); NULL })
if (!is.null(attrs)) cat(attrs, "\n")
