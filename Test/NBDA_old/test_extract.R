library(reticulate)
library(BayesianInference)
library(STbayes)

m <- importBI("cpu")
jnp <- import("jax.numpy")
np  <- import("numpy")

source("BI_Models/model_veff.R")
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
m$fit(bi_model_veff, num_chains=2L, num_warmup=10L, num_samples=10L)

# Test the new extraction function
bi_draws_from_m <- function() {
  raw <- m$posteriors
  if (is.null(raw)) return(list())
  nms <- names(raw)
  nms <- nms[!grepl("_lik$", nms)]
  out <- list()
  for (nm in nms) {
    v <- tryCatch({
      jax_arr <- raw[[nm]]
      np_arr  <- np$asarray(jax_arr)
      r_arr <- py_to_r(np_arr)
      
      if (length(dim(r_arr)) == 2) {
        as.numeric(r_arr)
      } else if (length(dim(r_arr)) == 3) {
        for (i in seq_len(dim(r_arr)[3])) {
           out_name <- paste0(nm, "[", i, "]")
           out[[out_name]] <- as.numeric(r_arr[,,i])
        }
        NULL
      } else {
        NULL
      }
    }, error = function(e) NULL)
    if (!is.null(v) && length(v) > 0) out[[nm]] <- v
  }
  out
}

res <- bi_draws_from_m()
cat("\nExtracted keys:\n")
print(names(res))

cat("\nLengths:\n")
for (k in names(res)) cat(k, ": length", length(res[[k]]), ", class", class(res[[k]]), "\n")

cat("\nSample of log_lambda_0_mean:\n")
print(head(res[["log_lambda_0_mean"]]))
