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
K <- dl$K; P <- dl$P
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
cat("fit type:", typeof(fit), "\n")
cat("fit class:", class(fit), "\n")
cat("fit is null:", is.null(fit), "\n\n")

# Inspect the m object itself after fitting
cat("m class:", class(m), "\n")
cat("m names:", paste(names(m), collapse=", "), "\n\n")

# Try all documented BI methods
cat("--- m$get_samples ---\n")
s <- tryCatch(m$get_samples(), error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(s)) { cat("type:", class(s), "\n"); cat("names:", paste(names(s), collapse=", "), "\n") }

cat("\n--- m$samples ---\n")
s2 <- tryCatch(m$samples, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(s2)) cat("type:", class(s2), "\n")

cat("\n--- m$posterior ---\n")
s3 <- tryCatch(m$posterior, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(s3)) { cat("type:", class(s3), "\n"); cat("names:", paste(names(s3), collapse=", "), "\n") }

cat("\n--- m$trace ---\n")
s4 <- tryCatch(m$trace, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(s4)) cat("type:", class(s4), "\n")

cat("\n--- m$mcmc ---\n")
s5 <- tryCatch(m$mcmc, error=function(e) cat("Error:", e$message, "\n"))
if (!is.null(s5)) cat("type:", class(s5), "\n")

cat("\nDone.\n")
