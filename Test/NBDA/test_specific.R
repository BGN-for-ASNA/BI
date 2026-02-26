library(reticulate)
library(BayesianInference)
library(STbayes)

m  <- importBI("cpu")
jnp <- import("jax.numpy")

source("BI_Models/model_OADA.R")
source("BI_Models/model_OADA_asocial.R")
source("BI_Models/model_cTADA.R")
source("BI_Models/model_ILV.R")
source("BI_Models/model_veff.R")
source("BI_Models/model_posterior_edges.R")
source("BI_Models/model_dynamic_tweights.R")
source("BI_Models/model_complex_f.R")

# Helpers
to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure" || is.character(x)) return(x)
  jnp$array(x)
}

bi_draws_from_m <- function() {
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
      if (is.null(dims) || length(dims) == 1) {
        as.numeric(r_arr)
      } else if (length(dims) == 2) {
        for (i in seq_len(dims[2])) {
           out_name <- paste0(nm, "[", i, "]")
           out[[out_name]] <- as.numeric(r_arr[, i])
        }
        NULL
      } else { NULL }
    }, error = function(e) NULL)
    if (!is.null(v) && length(v) > 0) out[[nm]] <- v
  }
  
  map_names <- c(
    "v_lambda0" = "log_lambda_0_mean",
    "v_sprime"  = "log_s_prime_mean",
    "log_f_mean" = "log_f",
    "k_raw"      = "k_raw", 
    "sigma_lambda0" = "sigma_veff[1]",
    "sigma_sprime"  = "sigma_veff[2]",
    "edge_weights" = "beta_ILV"
  )
  renamed_out <- list()
  for (orig_name in names(out)) {
    new_name <- orig_name
    for (k in names(map_names)) {
      if (orig_name == k) {
        new_name <- map_names[[k]]
      } else if (startsWith(orig_name, paste0(k, "["))) {
        new_name <- sub(k, map_names[[k]], orig_name)
      }
    }
    renamed_out[[new_name]] <- out[[orig_name]]
  }
  if (!is.null(renamed_out[["log_lambda_0_mean"]])) {
    renamed_out[["lambda_0"]] <- exp(renamed_out[["log_lambda_0_mean"]])
  }
  if (!is.null(renamed_out[["k_raw"]])) {
    renamed_out[["k_shape"]] <- 2.0 * (1.0 / (1.0 + exp(-renamed_out[["k_raw"]]))) - 1.0
    renamed_out[["k_raw"]] <- NULL
  }
  if (!is.null(renamed_out[["beta_ILVi_bool_ILV[1]"]])) {
    renamed_out[["beta_ILVi_cont_ILV"]] <- renamed_out[["beta_ILVi_bool_ILV[1]"]]
    renamed_out[["beta_ILVi_bool_ILV[1]"]] <- NULL
  }
  if (!is.null(renamed_out[["beta_ILVm_cat_ILV[1]"]])) {
    renamed_out[["beta_ILVm_cont_ILV"]] <- renamed_out[["beta_ILVm_cat_ILV[1]"]]
    renamed_out[["beta_ILVm_cat_ILV[1]"]] <- NULL
    renamed_out[["beta_ILVm_cat_ILV[2]"]] <- NULL
    renamed_out[["beta_ILVm_cat_ILV[3]"]] <- NULL
  }
  renamed_out
}

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

inject_model_data <- function(dl, nm) {
  P <- dl$raw$P; K <- dl$raw$K; T_max <- max(dl$raw$T)
  if (nm == "ILV" || nm == "complex_f") {
     dl$raw$ILV_cont_ILV <- rnorm(P)
     dl$raw$ILV_c <- 1
     dl$jax$ILV_cont_ILV <- to_jax(dl$raw$ILV_cont_ILV)
     if (nm == "ILV") {
       dl$raw$ILVi_names <- c("cont_ILV")
       dl$raw$ILVs_names <- c("cont_ILV")
       dl$raw$ILVm_names <- c("cont_ILV")
     }
  }
  if (nm == "veff") { dl$raw$N_veff <- 2 }
  dl
}

MODEL_CONFIG <- list(
  OADA_asocial = list(bi = bi_model_OADA_asocial, stb_args=list(data_type="order", model_type="asocial"), stb_extra = c()),
  complex_f    = list(bi = bi_model_complex_f, stb_args=list(transmission_func="freqdep_f"), stb_extra = c("log_f")),
  ILV          = list(bi = bi_model_ILV,
                      stb_args=list(), 
                      stb_extra = c("beta_ILVi_cont_ILV","beta_ILVs_cont_ILV","beta_ILVm_cont_ILV")),
  veff         = list(bi = bi_model_veff,
                      stb_args=list(veff_params = c("lambda_0", "s_prime")),
                      stb_extra = c("sigma_veff[1]","sigma_veff[2]"))
)

STB_PARAMS_COMMON <- c("log_lambda_0_mean", "log_s_prime_mean", "lambda_0")

for (nm in names(MODEL_CONFIG)) {
  cfg <- MODEL_CONFIG[[nm]]
  cat("\n", strrep("=", 50), "\n")
  cat("  Testing:", nm, "\n")
  cat(strrep("=", 50), "\n")

  model_dl <- inject_model_data(build_data_list(), nm)

  # --- BI fit ---
  m$data_on_model <- list(data = model_dl$jax)
  bi_success <- tryCatch({
    m$fit(cfg$bi, num_chains = 1L, num_warmup = 10L, num_samples = 10L) 
    TRUE
  }, error = function(e) { cat("  BI fit failed:", e$message, "\n"); FALSE })
  bi_s <- if (bi_success) bi_draws_from_m() else list()

  # --- STbayes Model Gen ---
  stb_args <- c(list(STb_data = model_dl$raw, gq = FALSE, est_acqTime = FALSE), cfg$stb_args)
  stan_code <- tryCatch(do.call(generate_STb_model, stb_args),
                        error = function(e) { cat("  STb gen failed:", e$message, "\n"); NULL })
  
  if (!is.null(stan_code)) {
    write(stan_code, file = paste0("debug_test_", nm, ".stan"))
  }

  params_to_pull <- c(STB_PARAMS_COMMON, cfg$stb_extra)
  if (grepl("OADA", nm)) { params_to_pull <- setdiff(params_to_pull, c("log_lambda_0_mean", "lambda_0")) }
  if (nm == "OADA_asocial") { params_to_pull <- character(0) }
  
  # simulate STB names simply by extracting the parameter definition block:
  stb_defined <- c()
  if (!is.null(stan_code)) {
     lines <- strsplit(stan_code, "\n")[[1]]
     for (l in lines) {
       if (grepl("^\\s*(real|vector|row_vector|matrix)[^\\[]*;", l) || grepl("^\\s*(real|vector<[^>]+>)\\s+[a-zA-Z0-9_]+;", l)) {
         stb_defined <- c(stb_defined, l)
       }
     }
  }

  cat("BI Plotted Scalar:", paste(names(bi_s)[!grepl("\\[", names(bi_s))], collapse=", "), "\n")
  cat("STB Set Parameters:", paste(params_to_pull, collapse=", "), "\n")
  cat("STB Definitions in code preview: ", length(stb_defined), " lines\n")
}
