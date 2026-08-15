# ============================================================
# Full Comparison Script: STbayes vs BF R Models
# Runs all 8 working models with 500 warmup + 500 samples
# Includes timing benchmarks per model
# ============================================================

BF_root <- normalizePath(file.path(getwd(), "../.."), mustWork = FALSE)
if (!nzchar(Sys.getenv("PYTHONPATH"))) Sys.setenv(PYTHONPATH = BF_root)

library(reticulate)
library(BayesForge)
library(STbayes)

m <- importBF("cpu")
jnp <- import("jax.numpy")

# ---- Source all BF models ----
source("BF_Models/model_OADA.R")
source("BF_Models/model_OADA_asocial.R")
source("BF_Models/model_cTADA.R")
source("BF_Models/model_ILV.R")
source("BF_Models/model_veff.R")
source("BF_Models/model_posterior_edges.R")
source("BF_Models/model_dynamic_tweights.R")
source("BF_Models/model_complex_f.R")

# ---- Helper: safely convert R matrix/vector to JAX ----
to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  return(jnp$array(x))
}

# ---- Build the shared JAX data list ----
build_data_list <- function() {
  data_list <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
  K <- data_list$K
  P <- data_list$P
  T_max <- max(data_list$T)
  
  obs_end_time <- matrix(0, nrow = K, ncol = P)
  is_event     <- matrix(FALSE, nrow = K, ncol = P)
  valid_ind    <- matrix(1, nrow = K, ncol = P)
  is_event_3d  <- array(0, dim = c(K, T_max, P))
  event_at_time <- array(FALSE, dim = c(K, T_max))
  
  for (k in 1:K) {
    for (n in 1:data_list$N[k]) {
      id <- data_list$ind_id[k, n]
      if (data_list$t[k, id] > 0) {
        obs_end_time[k, id] <- data_list$t[k, id]
        is_event[k, id] <- TRUE
        is_event_3d[k, data_list$t[k, id], id] <- 1
        event_at_time[k, data_list$t[k, id]] <- TRUE
      } else {
        valid_ind[k, id] <- 0
      }
    }
    if (data_list$N_c[k] > 0) {
      for (c in 1:data_list$N_c[k]) {
        id <- data_list$ind_id[k, data_list$N[k] + c]
        obs_end_time[k, id] <- data_list$T[k]
        is_event[k, id] <- FALSE
      }
    }
  }
  
  jax <- lapply(data_list, to_jax)
  jax$obs_end_time  <- jnp$array(obs_end_time)
  jax$is_event      <- jnp$array(is_event)
  jax$valid_ind     <- jnp$array(valid_ind)
  jax$is_event_3d   <- jnp$array(is_event_3d)
  jax$event_at_time <- jnp$array(event_at_time)
  
  # Dummy ILV placeholders (no ILVs in base dataset)
  jax$ILV_bool_ILV <- jnp$zeros(shape = tuple(P, 1L))
  jax$ILV_cont_ILV <- jnp$zeros(shape = tuple(P))
  jax$ILV_cat_ILV  <- jnp$zeros(shape = tuple(P, 3L))
  
  # Dummy posterior-edge placeholders
  n_net  <- if (is.null(data_list$N_networks)) 1L else data_list$N_networks
  n_dyad <- if (is.null(data_list$N_dyad))    1L else data_list$N_dyad
  jax$logit_edge_mu  <- jnp$zeros(shape = tuple(n_net, n_dyad))
  cov_arr <- array(0, dim = c(n_net, n_dyad, n_dyad))
  for (i in 1:n_net) cov_arr[i,,] <- diag(n_dyad)
  jax$logit_edge_cov <- jnp$array(cov_arr)
  jax$focal_ID       <- jnp$array(rep(1L, n_dyad))
  jax$other_ID       <- jnp$array(rep(1L, n_dyad))
  
  # Zn needed for dynamic-network and complex-f models
  jax$Zn <- jax$Z
  
  list(raw = data_list, jax = jax)
}

# ---- Master comparison function ----
run_full_comparison <- function(model_name, BF_func, stb_extra_args = list()) {
  cat("\n\n", strrep("=", 60), "\n")
  cat("  MODEL:", model_name, "\n")
  cat(strrep("=", 60), "\n")
  
  dl <- build_data_list()
  m$data_on_model <- list(data = dl$jax)

  # --- BF fit ---
  cat("\n[1] Fitting BF R model (500 warmup + 500 samples, 2 chains)...\n")
  BF_time <- system.time({
    BF_fit <- tryCatch(
      m$fit(BF_func, num_chains = 2L, num_warmup = 500L, num_samples = 500L),
      error = function(e) { cat("  BF ERROR:", conditionMessage(e), "\n"); NULL }
    )
  })
  
  BF_summary <- NULL
  if (!is.null(BF_fit)) {
    BF_summary <- m$summary(BF_fit)
    cat("\n  -- BF Posterior Summary --\n")
    print(BF_summary[, c("mean","sd","hdi_5.5%","hdi_94.5%")])
    cat(sprintf("  >> BF wall time: %.1f s\n", BF_time["elapsed"]))
  }
  
  # --- STbayes fit ---
  cat("\n[2] Fitting STbayes model (500 warmup + 500 samples, 2 chains)...\n")
  stb_time <- system.time({
    stan_code <- tryCatch(
      do.call(generate_STb_model, c(list(dl$raw, gq = FALSE, est_acqTime = FALSE), stb_extra_args)),
      error = function(e) { cat("  STbayes model gen failed:", conditionMessage(e), "\n"); NULL }
    )
    
    stb_fit <- NULL
    if (!is.null(stan_code)) {
      write(stan_code, file = paste0("temp_full_", model_name, ".stan"))
      stb_fit <- tryCatch(
        suppressWarnings(fit_STb(dl$raw,
                                 model_obj   = paste0("temp_full_", model_name, ".stan"),
                                 parallel_chains = 2, chains = 2,
                                 iter = 1000, refresh = 0)),
        error = function(e) { cat("  STbayes fit failed:", conditionMessage(e), "\n"); NULL }
      )
    }
  })
  
  stb_summary <- NULL
  if (!is.null(stb_fit)) {
    stb_summary <- STb_summary(stb_fit, digits = 3)
    cat("\n  -- STbayes Posterior Summary --\n")
    print(stb_summary[, c("Parameter","Median","MAD","CI_Lower","CI_Upper")])
    cat(sprintf("  >> STbayes wall time: %.1f s\n", stb_time["elapsed"]))
  }
  
  # --- Side-by-side comparison for matching parameters ---
  cat("\n  -- Side-by-Side Parameter Comparison --\n")
  if (!is.null(BF_summary) && !is.null(stb_summary)) {
    # Match on common key params
    BF_params <- rownames(BF_summary)
    stb_params <- stb_summary$Parameter
    
    cat(sprintf("  %-28s %10s %10s %10s %10s\n",
                "Parameter", "BF_Mean", "BF_SD", "STb_Median", "STb_MAD"))
    cat("  ", strrep("-", 68), "\n")
    
    for (p in c("log_lambda_0_mean", "log_s_prime_mean")) {
      BF_row  <- BF_summary[grep(p, BF_params, fixed=TRUE), ]
      stb_row <- stb_summary[grep(p, stb_params, fixed=TRUE), ]
      if (nrow(BF_row) > 0 && nrow(stb_row) > 0) {
        cat(sprintf("  %-28s %10.3f %10.3f %10.3f %10.3f\n",
                    p,
                    as.numeric(BF_row[1,"mean"]),
                    as.numeric(BF_row[1,"sd"]),
                    as.numeric(stb_row[1,"Median"]),
                    as.numeric(stb_row[1,"MAD"])))
      }
    }
    cat(sprintf("\n  Speedup factor (STbayes/BF): %.1fx\n",
                as.numeric(stb_time["elapsed"]) / max(as.numeric(BF_time["elapsed"]), 0.01)))
  } else {
    cat("  (incomplete — one or both fits failed)\n")
  }
  
  invisible(list(BF = BF_summary, stb = stb_summary,
                 BF_time = BF_time, stb_time = stb_time))
}

# ============================================================
#  Run all 8 working models
# ============================================================
results <- list()

results$cTADA        <- run_full_comparison("cTADA",        BF_model_cTADA)
results$OADA         <- run_full_comparison("OADA",         BF_model_OADA)
results$OADA_asocial <- run_full_comparison("OADA_Asocial", BF_model_OADA_asocial)
results$ILV          <- run_full_comparison("ILV",          BF_model_ILV)
results$veff         <- run_full_comparison("veff",         BF_model_veff,
                          stb_extra_args = list(veff_params = c("lambda_0", "s_prime")))
results$dynamic_tw   <- run_full_comparison("dynamic_tweights", BF_model_dynamic_networks_dynamic_tweights)
results$complex_f    <- run_full_comparison("complex_f",    BF_model_complex_f)
results$post_edges   <- run_full_comparison("posterior_edges", BF_model_posterior_edges)

# ============================================================
#  Final benchmark summary table
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("  BENCHMARK SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(sprintf("  %-20s %12s %12s %12s\n", "Model", "BF_time(s)", "STb_time(s)", "Speedup"))
cat("  ", strrep("-", 60), "\n")

for (nm in names(results)) {
  r <- results[[nm]]
  BF_t  <- as.numeric(r$BF_time["elapsed"])
  stb_t <- as.numeric(r$stb_time["elapsed"])
  cat(sprintf("  %-20s %12.1f %12.1f %11.1fx\n",
              nm, BF_t, stb_t, stb_t / max(BF_t, 0.01)))
}
cat("\nDone!\n")
