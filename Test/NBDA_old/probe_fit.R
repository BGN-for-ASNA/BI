library(reticulate)
library(BayesianInference)
library(STbayes)

m <- importBI("cpu")
jnp <- import("jax.numpy")

source("BI_Models/model_OADA.R")
source("BI_Models/model_cTADA.R")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  jnp$array(x)
}

data_list <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
K <- data_list$K; P <- data_list$P; T_max <- max(data_list$T)

obs_end_time <- matrix(0, K, P); is_event <- matrix(FALSE, K, P)
valid_ind <- matrix(1, K, P)
is_event_3d <- array(0, c(K, T_max, P)); event_at_time <- array(FALSE, c(K, T_max))

for (k in 1:K) {
  for (n in 1:data_list$N[k]) {
    id <- data_list$ind_id[k,n]
    if (data_list$t[k,id] > 0) {
      obs_end_time[k,id] <- data_list$t[k,id]; is_event[k,id] <- TRUE
      is_event_3d[k, data_list$t[k,id], id] <- 1; event_at_time[k, data_list$t[k,id]] <- TRUE
    } else { valid_ind[k,id] <- 0 }
  }
  if (data_list$N_c[k] > 0) for (c in 1:data_list$N_c[k]) {
    id <- data_list$ind_id[k, data_list$N[k]+c]; obs_end_time[k,id] <- data_list$T[k]
  }
}

jax <- lapply(data_list, to_jax)
jax$obs_end_time <- jnp$array(obs_end_time); jax$is_event <- jnp$array(is_event)
jax$valid_ind <- jnp$array(valid_ind); jax$is_event_3d <- jnp$array(is_event_3d)
jax$event_at_time <- jnp$array(event_at_time)

m$data_on_model <- list(data = jax)

cat("Fitting cTADA...\n")
fit <- m$fit(bi_model_cTADA, num_chains=1L, num_warmup=100L, num_samples=200L)

cat("\n--- Exploring fit object ---\n")
cat("Class:", class(fit), "\n")
cat("Names/attributes:", paste(names(fit), collapse=", "), "\n")

# Try various sample extraction approaches
cat("\n--- Trying posterior_samples ---\n")
ps <- tryCatch(fit$posterior_samples, error=function(e) { cat("  Error:", e$message, "\n"); NULL })
if (!is.null(ps)) {
  cat("  Available: ", paste(names(py_to_r(ps)), collapse=", "), "\n")
}

cat("\n--- Trying get_samples() ---\n")
gs <- tryCatch(fit$get_samples(), error=function(e) { cat("  Error:", e$message, "\n"); NULL })
if (!is.null(gs)) cat("  Type:", class(gs), "\n")

cat("\n--- Trying posterior attribute ---\n")
post <- tryCatch(fit$posterior, error=function(e) { cat("  Error:", e$message, "\n"); NULL })
if (!is.null(post)) cat("  Type:", class(post), "\n")

cat("\n--- m$summary row names ---\n")
smry <- m$summary(fit)
cat(rownames(smry), "\n")
cat("\nDone exploring.\n")
