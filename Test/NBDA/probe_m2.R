library(reticulate)
library(BayesianInference)
library(STbayes)
m <- importBI("cpu")
jnp <- import("jax.numpy")
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

m$fit(bi_model_cTADA, num_chains=1L, num_warmup=50L, num_samples=100L)

# INSPECT m$posteriors
cat("--- m$posteriors ---\n")
post <- tryCatch(m$posteriors, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(post) && !inherits(post, "character")) {
  cat("type:", class(post), "\n")
  r_post <- tryCatch(py_to_r(post), error=function(e) { cat("py_to_r failed:", e$message, "\n"); NULL })
  if (!is.null(r_post)) { cat("names:", paste(names(r_post), collapse=", "), "\n") }
} else { cat("value:", post, "\n") }

# INSPECT m$trace
cat("\n--- m$trace ---\n")
tr <- tryCatch(m$trace, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(tr) && !inherits(tr, "character")) {
  cat("type:", class(tr), "\n")
  r_tr <- tryCatch(py_to_r(tr), error=function(e) { cat("py_to_r failed:", e$message, "\n"); NULL })
  if (!is.null(r_tr)) { cat("names:", paste(names(r_tr), collapse=", "), "\n") }
} else { cat("value:", tr, "\n") }

# INSPECT m$history
cat("\n--- m$history ---\n")
hist <- tryCatch(m$history, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(hist) && !inherits(hist, "character")) {
  cat("type:", class(hist), "\n")
  cat("length:", length(hist), "\n")
} else { cat("value:", hist, "\n") }

cat("\nDone.\n")
