#%%

# ============================================================
# Density Plot Comparison: BF R vs STbayes
# Produces overlapping density plots for all parameters
# of all models, saved to density_plots/
# ============================================================
# Editable install of BayesForge uses a relative path hook — must set
# PYTHONPATH to project root before Python initializes so 'BF' module resolves.
BF_root <- normalizePath(file.path(getwd(), "../.."), mustWork = FALSE)
if (!nzchar(Sys.getenv("PYTHONPATH"))) Sys.setenv(PYTHONPATH = BF_root)

library(reticulate)
library(BayesForge)
library(STbayes)

# Create output directory
dir.create("density_plots", showWarnings = FALSE)

m  <- importBF("cpu")
jnp <- import("jax.numpy")

# Source all BF models
source("BF_Models/model_OADA.R")
source("BF_Models/model_OADA_asocial.R")
source("BF_Models/model_cTADA.R")
source("BF_Models/model_ILV.R")
source("BF_Models/model_veff.R")
source("BF_Models/model_posterior_edges.R")
source("BF_Models/model_dynamic_tweights.R")
source("BF_Models/model_complex_f.R")

# ---- Helpers -------------------------------------------------------

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  jnp$array(x)
}

#%%
# Extract raw BF posterior draws from m$posteriors (stateful after m$fit())
# m$posteriors values are JAX arrays — must convert via numpy before py_to_r
# BF already flattens chains and samples into the first dimension:
#   - Scalar parameter: 1D array of shape (n_total_samples,)
#   - Vector parameter: 2D array of shape (n_total_samples, vec_length)
BF_draws_from_m <- function() {
  np  <- import("numpy")
  raw <- m$posteriors
  if (is.null(raw)) return(list())
  nms <- names(raw)
  nms <- nms[!grepl("_lik$", nms)]  # drop likelihood tracker entries
  out <- list()
  for (nm in nms) {
    v <- tryCatch({
      jax_arr <- raw[[nm]]
      np_arr  <- np$asarray(jax_arr)
      r_arr <- py_to_r(np_arr)
      
      dims <- dim(r_arr)
      # Scalar param: no dimensions (just a vector) or exactly 1 dimension
      if (is.null(dims) || length(dims) == 1) {
        as.numeric(r_arr)
      } 
      # Vector param (e.g. sigma_veff of length 2): shape is (samples, 2)
      else if (length(dims) == 2) {
        for (i in seq_len(dims[2])) {
           out_name <- paste0(nm, "[", i, "]")
           out[[out_name]] <- as.numeric(r_arr[, i])
        }
        NULL # don't add the root name
      } else {
        NULL # omit higher dimensional matrices for density plots
      }
    }, error = function(e) NULL)
    
    # Only assign if it returned a flat vector (not NULL from the 2D loop)
    if (!is.null(v) && length(v) > 0) out[[nm]] <- v
  }
  
  # Translate BF parameter names to STbayes parameter names for standard overlay plotting
  map_names <- c(
    "v_lambda0" = "log_lambda_0_mean",
    "v_sprime"  = "log_s_prime_mean",
    "k_raw"      = "k_raw", # keep as is initially, will convert to k_shape
    "sigma_lambda0" = "sigma_id[1]",
    "sigma_sprime"  = "sigma_id[2]",
    "edge_weights" = "beta_ILV"
  )
  
  # For 2D vector parameters like sigma_veff, BF makes sigma_veff[1], STbayes uses sigma_veff[1] 
  # wait - BF's scalar values are exactly those. 
  
  # Apply renaming map
  renamed_out <- list()
  for (orig_name in names(out)) {
    new_name <- orig_name
    for (k in names(map_names)) {
      if (orig_name == k) {
        new_name <- map_names[[k]]
      } else if (startsWith(orig_name, paste0(k, "["))) {
        # e.g. v_lambda0[1] -> log_lambda_0_mean[1] (though usually scalar)
        new_name <- sub(k, map_names[[k]], orig_name)
      }
    }
    renamed_out[[new_name]] <- out[[orig_name]]
  }
  
  # Add derived parameters that STbayes computes in generated quantities
  if (!is.null(renamed_out[["log_lambda_0_mean"]])) {
    renamed_out[["lambda_0"]] <- exp(renamed_out[["log_lambda_0_mean"]])
  }
  # k_shape derivation from k_raw
  if (!is.null(renamed_out[["k_raw"]])) {
    renamed_out[["k_shape"]] <- 2.0 * (1.0 / (1.0 + exp(-renamed_out[["k_raw"]]))) - 1.0
    renamed_out[["k_raw"]] <- NULL # Remove k_raw as STbayes outputs k_shape
  }
  
  # Map BF ILV names to STbayes ILV names for overlay
  if (!is.null(renamed_out[["beta_ILVi_bool_ILV[1]"]])) {
    renamed_out[["beta_ILVi_cont_ILV"]] <- renamed_out[["beta_ILVi_bool_ILV[1]"]]
    renamed_out[["beta_ILVi_bool_ILV[1]"]] <- NULL
  }
  if (!is.null(renamed_out[["beta_ILVm_cat_ILV[1]"]])) {
    renamed_out[["beta_ILVm_cont_ILV"]] <- renamed_out[["beta_ILVm_cat_ILV[1]"]]
    renamed_out[["beta_ILVm_cat_ILV[1]"]] <- NULL
    renamed_out[["beta_ILVm_cat_ILV[2]"]] <- NULL # drop the others so they don't plot randomly
    renamed_out[["beta_ILVm_cat_ILV[3]"]] <- NULL
  }
  
  renamed_out
}

# Symmetric KL divergence between two sample sets via KDE on shared grid
kl_div <- function(x, y, n = 512) {
  lo <- min(c(x, y)); hi <- max(c(x, y))
  dp <- density(x, from = lo, to = hi, n = n)$y
  dq <- density(y, from = lo, to = hi, n = n)$y
  eps <- 1e-10
  dp <- pmax(dp, eps); dq <- pmax(dq, eps)
  dp <- dp / sum(dp);  dq <- dq / sum(dq)
  (sum(dp * log(dp / dq)) + sum(dq * log(dq / dp))) / 2
}

# Extract STbayes raw draws for specific parameters via cmdstanr
stb_draws <- function(stb_fit, params) {
  tryCatch({
    d <- stb_fit$draws(format = "draws_matrix")
    cols <- colnames(d)
    out <- list()
    for (p in params) {
      idx <- which(cols == p)
      if (length(idx) > 0) out[[p]] <- as.numeric(d[, idx[1]])
    }
    out
  }, error = function(e) { cat("  STb draw extraction failed:", e$message, "\n"); NULL })
}

# Build the shared JAX data list (same as full_comparison.R)
# Build a generic template structure for all models
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

# Dynamic injection for specific models
inject_model_data <- function(dl, nm) {
  P <- dl$raw$P; K <- dl$raw$K; T_max <- max(dl$raw$T)
  
  if (nm == "ILV" || nm == "complex_f") {
    ilv_vals <- rnorm(P)
    dl$jax$ILV_cont_ILV <- to_jax(ilv_vals)
    # BF model uses ILV_bool_ILV for asocial and ILV_cat_ILV for multiplicative effects.
    # Set both to the same ILV values so all three beta parameters see equivalent data
    # as STbayes (which uses the same cont_ILV vector for ILVi, ILVs, and ILVm).
    dl$jax$ILV_bool_ILV <- jnp$array(matrix(ilv_vals, ncol = 1L))
    cat_mat <- matrix(0, nrow = P, ncol = 3L)
    cat_mat[, 1] <- ilv_vals
    dl$jax$ILV_cat_ILV <- jnp$array(cat_mat)

    if (nm == "ILV") {
      # ILV_c expects a data frame with an 'id' column matching event_data individual IDs
      ids <- sort(unique(STbayes::event_data$id))
      ilv_df <- data.frame(id = ids, cont_ILV = ilv_vals)
      dl$raw <- import_user_STb(STbayes::event_data, STbayes::edge_list,
                                 network_type = "undirected",
                                 ILV_c = ilv_df,
                                 ILVi = "cont_ILV",
                                 ILVs = "cont_ILV",
                                 ILVm = "cont_ILV")
    } else {
      dl$raw$ILV_cont_ILV <- ilv_vals
      dl$raw$ILV_c <- 1L
    }
  }
  
  if (nm == "dynamic_tweights") {
     dl$raw$t_weights <- array(1, dim=c(K, T_max, P, P))
  }
  
  if (nm == "posterior_edges") {
     # Create dummy posteriors 
     net_mu <- array(0, dim=c(1, P*(P-1)/2))
     net_cov <- array(0, dim=c(1, P*(P-1)/2, P*(P-1)/2))
     for(i in 1:dim(net_cov)[2]) net_cov[1,i,i] <- 1
     dl$raw$logit_edge_mu <- net_mu
     dl$raw$logit_edge_cov <- net_cov
     dl$raw$N_networks <- 1
     dl$raw$N_dyad <- P*(P-1)/2
     fID <- rep(1, dl$raw$N_dyad)
     oID <- rep(1, dl$raw$N_dyad)
     k <- 1
     for(i in 1:(P-1)){
       for(j in (i+1):P){
         fID[k] <- i; oID[k] <- j; k <- k+1
       }
     }
     dl$raw$focal_ID <- fID
     dl$raw$other_ID <- oID
     
     dl$jax$logit_edge_mu <- to_jax(net_mu)
     dl$jax$logit_edge_cov <- to_jax(net_cov)
     # Convert to 0-based integers for JAX python indexing!
     dl$jax$focal_ID <- to_jax(fID - 1L)$astype("int32")
     dl$jax$other_ID <- to_jax(oID - 1L)$astype("int32")
  }
  
  if (nm == "veff") {
     dl$raw$N_veff <- 2 # STbayes explicitly requires this in data block when grouping 2 veff parameters
  }
  
  dl
}

# Draw overlapping density plots and save as PNG
plot_densities <- function(model_name, BF_samp, stb_samp) {
  params_BF  <- names(BF_samp)
  params_stb <- names(stb_samp)
  # Only plot scalar parameters, but explicitly allow sigma_veff[#] and beta_ILV[#] 
  # to capture random effect variances and categorical ILV elements.
  scalar_BF  <- params_BF[!grepl("\\[", params_BF) | grepl("sigma_id|beta_ILV", params_BF)]
  scalar_stb <- params_stb[!grepl("\\[", params_stb) | grepl("sigma_id|beta_ILV", params_stb)]

  shared <- intersect(scalar_BF, scalar_stb)
  BF_only <- setdiff(scalar_BF, scalar_stb)
  stb_only <- setdiff(scalar_stb, scalar_BF)

  all_params <- union(union(shared, BF_only), stb_only)
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
    BF_v  <- if (!is.null(BF_samp[[p]])) BF_samp[[p]] else NULL
    stb_v <- if (!is.null(stb_samp[[p]])) stb_samp[[p]] else NULL

    status <- if (p %in% shared) "" else if (p %in% BF_only) " [BF only]" else " [STb only]"

    xlim <- range(c(BF_v, stb_v), na.rm = TRUE)
    xlim <- xlim + c(-1, 1) * diff(xlim) * 0.05

    ylim_max <- 0
    if (!is.null(BF_v) && length(BF_v) > 4)
      ylim_max <- max(ylim_max, max(density(BF_v)$y))
    if (!is.null(stb_v) && length(stb_v) > 4)
      ylim_max <- max(ylim_max, max(density(stb_v)$y))

    plot(NULL, xlim = xlim, ylim = c(0, ylim_max * 1.1),
         xlab = p, ylab = "Density",
         main = paste0(p, status), cex.main = 0.9, font.main = 1,
         col.main = if (status == "") "black" else "gray40")

    if (!is.null(stb_v) && length(stb_v) > 4) {
      d <- density(stb_v)
      polygon(d$x, d$y, col = adjustcolor("#4E9DC4", alpha.f = 0.5),
              border = "#4E9DC4", lwd = 1.5)
    }
    if (!is.null(BF_v) && length(BF_v) > 4) {
      d <- density(BF_v)
      polygon(d$x, d$y, col = adjustcolor("#F5A623", alpha.f = 0.5),
              border = "#F5A623", lwd = 1.5)
    }

    if (p %in% shared && !is.null(BF_v) && !is.null(stb_v) &&
        length(BF_v) > 4 && length(stb_v) > 4) {
      kl <- kl_div(stb_v, BF_v)
      mtext(sprintf("sym-KL = %.4f", kl), side = 3, line = -1.5,
            cex = 0.7, col = "gray40")
    }

    legend("topright", legend = c("STbayes", "BF"),
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
  cTADA        = list(BF = BF_model_cTADA, stb_args=list(), stb_extra = c()),
  OADA         = list(BF = BF_model_OADA, stb_args=list(data_type="order"), stb_extra = c()),
  OADA_asocial = list(BF = BF_model_OADA_asocial, stb_args=list(data_type="order", model_type="asocial"), stb_extra = c()),
  ILV          = list(BF = BF_model_ILV,
                      stb_args=list(), # ILV data injected dynamically
                      stb_extra = c("beta_ILVi_cont_ILV","beta_ILVs_cont_ILV","beta_ILVm_cont_ILV")),
  veff         = list(BF = BF_model_veff,
                      stb_args=list(veff_params = c("lambda_0", "s_prime")),
                      stb_extra = c("sigma_id[1]","sigma_id[2]")),
  dynamic_tweights = list(BF = BF_model_dynamic_networks_dynamic_tweights,
                          stb_file="STAN_example_dynamic_networks_dynamic_tweights.stan",
                          stb_extra = c("k_shape")),
  complex_f    = list(BF = BF_model_complex_f, stb_args=list(transmission_func="freqdep_f"), stb_extra = c("log_f_mean")),
  posterior_edges = list(BF = BF_model_posterior_edges, stb_file="STAN_example_posterior_edges.stan", stb_extra = c())
)

dl <- build_data_list()

for (nm in names(MODEL_CONFIG)) {
  cfg <- MODEL_CONFIG[[nm]]
  cat("\n", strrep("=", 50), "\n")
  cat("  Processing:", nm, "\n")
  cat(strrep("=", 50), "\n")

  model_dl <- inject_model_data(build_data_list(), nm)

  # --- BF fit ---
  m$data_on_model <- list(data = model_dl$jax)
  BF_success <- tryCatch({
    m$fit(cfg$BF, num_chains = 2L, num_warmup = 500L, num_samples = 500L)
    TRUE
  }, error = function(e) { cat("  BF fit failed:", e$message, "\n"); FALSE })
  BF_s <- if (BF_success) BF_draws_from_m() else list()

  # --- STbayes fit (matched model for comparison) ---
  stb_fit <- NULL
  if (!is.null(cfg$stb_file)) {
    stan_file_path <- system.file("extdata", cfg$stb_file, package = "STbayes")
    if (nzchar(stan_file_path) && file.exists(stan_file_path)) {
      cat("  Starting STbayes fitting from file", nm, "...\n")
      stb_fit <- tryCatch(
        suppressWarnings(fit_STb(model_dl$raw,
                                 model_obj = stan_file_path,
                                 parallel_chains = 2, chains = 2, iter = 600, refresh = 0)),
        error = function(e) { cat("  STb fit failed:", e$message, "\n"); NULL }
      )
    } else {
      cat("  STb Stan file not found in STbayes package:", cfg$stb_file, "\n")
    }
  } else {
    # Generate dynamically
    stb_args <- c(list(STb_data = model_dl$raw, gq = FALSE, est_acqTime = FALSE), cfg$stb_args)
    stan_code <- tryCatch(do.call(generate_STb_model, stb_args),
                          error = function(e) { cat("  STb gen failed:", e$message, "\n"); NULL })
    if (!is.null(stan_code)) {
      write(stan_code, file = paste0("temp_dens_", nm, ".stan"))
      cat("  Starting STbayes fitting from generated source", nm, "...\n")
      stb_fit <- tryCatch(
        suppressWarnings(fit_STb(model_dl$raw,
                                 model_obj = paste0("temp_dens_", nm, ".stan"),
                                 parallel_chains = 2, chains = 2, iter = 600, refresh = 0)),
        error = function(e) { cat("  STb fit failed:", e$message, "\n"); NULL }
      )
    }
  }

  params_to_pull <- c(STB_PARAMS_COMMON, cfg$stb_extra)
  # OADA: lambda_0 cancels out of the partial likelihood
  if (grepl("OADA", nm)) {
    params_to_pull <- setdiff(params_to_pull, c("log_lambda_0_mean", "lambda_0"))
  }
  # veff: lambda_0 is individual-specific vector in STbayes (lambda_0[p]), no scalar equivalent
  if (nm == "veff") {
    params_to_pull <- setdiff(params_to_pull, "lambda_0")
  }
  if (nm == "OADA_asocial") {
    # It has neither lambda_0 nor s_prime because it's purely asocial and relative
    params_to_pull <- character(0)
  }
  
  stb_s <- if (!is.null(stb_fit)) stb_draws(stb_fit, params_to_pull) else list()

  # --- Plot ---
  plot_densities(nm, BF_s, stb_s)
}

cat("\n\nAll density plots saved to density_plots/\n")
