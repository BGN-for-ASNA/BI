# ============================================================
# Density Plot Comparison: BI R vs STbayes
# Produces overlapping density plots for all parameters
# of all models, saved to density_plots/
# ============================================================

library(reticulate)
library(BayesianInference)
library(STbayes)

# Create output directory
dir.create("density_plots", showWarnings = FALSE)

m  <- importBI("cpu")
jnp <- import("jax.numpy")

# Source all BI models
source("BI_Models/model_OADA.R")
source("BI_Models/model_OADA_asocial.R")
source("BI_Models/model_cTADA.R")
source("BI_Models/model_ILV.R")
source("BI_Models/model_veff.R")
source("BI_Models/model_posterior_edges.R")
source("BI_Models/model_dynamic_tweights.R")
source("BI_Models/model_complex_f.R")

# ---- Helpers -------------------------------------------------------

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  jnp$array(x)
}

# Extract raw BI posterior draws as a named list of numeric vectors
bi_draws <- function(fit) {
  raw <- py_to_r(fit$posterior_samples)
  lapply(raw, function(v) as.numeric(py_to_r(v)))
}

# Extract STbayes raw draws for specific parameters via cmdstanr
stb_draws <- function(stb_fit, params) {
  tryCatch({
    d <- stb_fit$draws(format = "draws_matrix")
    cols <- colnames(d)
    out <- list()
    for (p in params) {
      idx <- grep(paste0("^", p, "$"), cols)
      if (length(idx) > 0) out[[p]] <- as.numeric(d[, idx[1]])
    }
    out
  }, error = function(e) { cat("  STb draw extraction failed:", e$message, "\n"); NULL })
}

# Build the shared JAX data list (same as full_comparison.R)
build_data_list <- function() {
  dl <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
  K <- dl$K; P <- dl$P; T_max <- max(dl$T)

  obs_end_time  <- matrix(0, K, P)
  is_event      <- matrix(FALSE, K, P)
  valid_ind     <- matrix(1, K, P)
  is_event_3d   <- array(0, c(K, T_max, P))
  event_at_time <- array(FALSE, c(K, T_max))

  for (k in 1:K) {
    for (n in 1:dl$N[k]) {
      id <- dl$ind_id[k, n]
      if (dl$t[k, id] > 0) {
        obs_end_time[k, id] <- dl$t[k, id]; is_event[k, id] <- TRUE
        is_event_3d[k, dl$t[k, id], id] <- 1; event_at_time[k, dl$t[k, id]] <- TRUE
      } else { valid_ind[k, id] <- 0 }
    }
    if (dl$N_c[k] > 0) for (c in 1:dl$N_c[k]) {
      id <- dl$ind_id[k, dl$N[k] + c]; obs_end_time[k, id] <- dl$T[k]
    }
  }

  jax <- lapply(dl, to_jax)
  jax$obs_end_time  <- jnp$array(obs_end_time)
  jax$is_event      <- jnp$array(is_event)
  jax$valid_ind     <- jnp$array(valid_ind)
  jax$is_event_3d   <- jnp$array(is_event_3d)
  jax$event_at_time <- jnp$array(event_at_time)

  # Dummy ILV placeholders
  jax$ILV_bool_ILV <- jnp$zeros(shape = tuple(P, 1L))
  jax$ILV_cont_ILV <- jnp$zeros(shape = tuple(P))
  jax$ILV_cat_ILV  <- jnp$zeros(shape = tuple(P, 3L))

  n_net  <- if (is.null(dl$N_networks)) 1L else dl$N_networks
  n_dyad <- if (is.null(dl$N_dyad))    1L else dl$N_dyad
  jax$logit_edge_mu  <- jnp$zeros(shape = tuple(n_net, n_dyad))
  cov_arr <- array(0, c(n_net, n_dyad, n_dyad))
  for (i in 1:n_net) cov_arr[i,,] <- diag(n_dyad)
  jax$logit_edge_cov <- jnp$array(cov_arr)
  jax$focal_ID <- jnp$array(rep(1L, n_dyad))
  jax$other_ID <- jnp$array(rep(1L, n_dyad))
  jax$Zn <- jax$Z

  list(raw = dl, jax = jax)
}

# Draw overlapping density plots and save as PNG
plot_densities <- function(model_name, bi_samp, stb_samp) {
  params_bi  <- names(bi_samp)
  params_stb <- names(stb_samp)
  # Only plot scalar parameters (exclude vector ones like v_id, v_sprime[0]...)
  scalar_bi  <- params_bi[!grepl("\\[", params_bi)]
  scalar_stb <- params_stb[!grepl("\\[", params_stb)]

  shared <- intersect(scalar_bi, scalar_stb)
  bi_only <- setdiff(scalar_bi, scalar_stb)
  stb_only <- setdiff(scalar_stb, scalar_bi)

  all_params <- union(union(shared, bi_only), stb_only)
  if (length(all_params) == 0) {
    cat("  No parameters to plot for", model_name, "\n"); return(invisible(NULL))
  }

  n_params <- length(all_params)
  ncols <- min(n_params, 3)
  nrows <- ceiling(n_params / ncols)

  png(file.path("density_plots", paste0(model_name, ".png")),
      width = 380 * ncols, height = 370 * nrows, res = 96)
  par(mfrow = c(nrows, ncols), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0),
      bg = "white")

  for (p in all_params) {
    bi_v  <- if (!is.null(bi_samp[[p]])) bi_samp[[p]] else NULL
    stb_v <- if (!is.null(stb_samp[[p]])) stb_samp[[p]] else NULL

    xlim <- range(c(bi_v, stb_v), na.rm = TRUE)
    xlim <- xlim + c(-1, 1) * diff(xlim) * 0.05  # 5% padding

    ylim_max <- 0
    if (!is.null(bi_v) && length(bi_v) > 4)
      ylim_max <- max(ylim_max, max(density(bi_v)$y))
    if (!is.null(stb_v) && length(stb_v) > 4)
      ylim_max <- max(ylim_max, max(density(stb_v)$y))

    plot(NULL, xlim = xlim, ylim = c(0, ylim_max * 1.1),
         xlab = p, ylab = "Density",
         main = p, cex.main = 0.9, font.main = 1)

    if (!is.null(stb_v) && length(stb_v) > 4) {
      d <- density(stb_v)
      polygon(d$x, d$y, col = adjustcolor("#4E9DC4", alpha.f = 0.5),
              border = "#4E9DC4", lwd = 1.5)
    }
    if (!is.null(bi_v) && length(bi_v) > 4) {
      d <- density(bi_v)
      polygon(d$x, d$y, col = adjustcolor("#F5A623", alpha.f = 0.5),
              border = "#F5A623", lwd = 1.5)
    }

    legend("topright", legend = c("STbayes", "BI"),
           fill = c(adjustcolor("#4E9DC4", 0.5), adjustcolor("#F5A623", 0.5)),
           border = c("#4E9DC4", "#F5A623"), bty = "n", cex = 0.8)
  }

  mtext(paste("Posterior Density:", model_name), outer = TRUE, cex = 1.1, font = 2)
  dev.off()
  cat("  Saved density_plots/", model_name, ".png\n", sep="")
}

# ============================================================
# Main loop: fit each model, extract draws, produce plots
# ============================================================

STB_PARAMS_COMMON <- c("log_lambda_0_mean", "log_s_prime_mean", "lambda_0")

MODEL_CONFIG <- list(
  cTADA        = list(bi = bi_model_cTADA,                             stb_extra = c()),
  OADA         = list(bi = bi_model_OADA,                              stb_extra = c()),
  OADA_asocial = list(bi = bi_model_OADA_asocial,                      stb_extra = c()),
  ILV          = list(bi = bi_model_ILV,
                      stb_extra = c("beta_ILVi_cont_ILV","beta_ILVs_cont_ILV","beta_ILVm_cont_ILV")),
  veff         = list(bi = bi_model_veff,
                      stb_extra = c("sigma_veff[1]","sigma_veff[2]")),
  dynamic_tweights = list(bi = bi_model_dynamic_networks_dynamic_tweights,
                          stb_extra = c("k_shape")),
  complex_f    = list(bi = bi_model_complex_f,   stb_extra = c("log_f")),
  posterior_edges = list(bi = bi_model_posterior_edges, stb_extra = c())
)

dl <- build_data_list()

for (nm in names(MODEL_CONFIG)) {
  cfg <- MODEL_CONFIG[[nm]]
  cat("\n", strrep("=", 50), "\n")
  cat("  Processing:", nm, "\n")
  cat(strrep("=", 50), "\n")

  # --- BI fit ---
  m$data_on_model <- list(data = dl$jax)
  bi_fit <- tryCatch(
    m$fit(cfg$bi, num_chains = 2L, num_warmup = 500L, num_samples = 500L),
    error = function(e) { cat("  BI fit failed:", e$message, "\n"); NULL }
  )
  bi_s <- if (!is.null(bi_fit)) bi_draws(bi_fit) else list()

  # --- STbayes fit (single standard model for comparison) ---
  stan_code <- tryCatch(generate_STb_model(dl$raw, gq = FALSE, est_acqTime = FALSE),
                        error = function(e) NULL)
  stb_fit <- NULL
  if (!is.null(stan_code)) {
    write(stan_code, file = paste0("temp_dens_", nm, ".stan"))
    stb_fit <- tryCatch(
      suppressWarnings(fit_STb(dl$raw,
                               model_obj = paste0("temp_dens_", nm, ".stan"),
                               parallel_chains = 2, chains = 2, iter = 1000, refresh = 0)),
      error = function(e) { cat("  STb fit failed:", e$message, "\n"); NULL }
    )
  }

  params_to_pull <- c(STB_PARAMS_COMMON, cfg$stb_extra)
  stb_s <- if (!is.null(stb_fit)) stb_draws(stb_fit, params_to_pull) else list()

  # --- Plot ---
  plot_densities(nm, bi_s, stb_s)
}

cat("\n\nAll density plots saved to density_plots/\n")
