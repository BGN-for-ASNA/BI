setwd("/home/sebastian_sosa/BI")

library(reticulate)
library(BayesianInference)
library(bisonR)
library(cmdstanr)

BASE        <- "Test/Network/Network edges"
RESULTS_DIR <- file.path(BASE, "results")
stan_out_dir <- file.path(BASE, "stan_output")
dir.create(RESULTS_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(stan_out_dir, showWarnings = FALSE, recursive = TRUE)
options(cmdstanr_output_dir = stan_out_dir)

source(file.path(BASE, "Modified simulate_bison_model.R"))
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")
source(file.path(BASE, "bi_model_binary.R"))

m   <- importBI("cpu")
jnp <- import("jax.numpy")

get_bi_draws <- function(raw_post) {
  np <- import("numpy")
  out <- list()
  for (nm in names(raw_post)) {
    if (grepl("_lik$", nm) || nm == "event" || nm == "event_count") next
    r_arr <- py_to_r(np$asarray(raw_post[[nm]]))
    dims   <- dim(r_arr)
    if (!is.null(dims) && length(dims) == 3) {
      r_arr <- matrix(r_arr, nrow = dims[1] * dims[2], ncol = dims[3])
      dims  <- dim(r_arr)
    }
    if (is.null(dims) || length(dims) == 1) {
      out[[nm]] <- as.numeric(r_arr)
    } else if (length(dims) == 2) {
      for (i in seq_len(dims[2]))
        out[[paste0(nm, "[", i, "]")]] <- as.numeric(r_arr[, i])
    }
  }
  out
}

get_stan_draws <- function(fit, param_names) {
  out <- list()
  for (p in param_names) {
    tryCatch({
      draws <- fit$draws(p, format = "matrix")
      if (ncol(draws) == 1) {
        out[[colnames(draws)[1]]] <- as.numeric(draws[, 1])
      } else {
        for (i in seq_len(ncol(draws)))
          out[[colnames(draws)[i]]] <- as.numeric(draws[, i])
      }
    }, error = function(e) NULL)
  }
  out
}

normalize_bi_names <- function(bi_draws, stan_draws) {
  out <- bi_draws
  for (nm in names(bi_draws)) {
    idx_nm <- paste0(nm, "[1]")
    if (idx_nm %in% names(stan_draws) && !(nm %in% names(stan_draws))) {
      out[[idx_nm]] <- out[[nm]]; out[[nm]] <- NULL
    }
  }
  out
}

center_undirected_random_effects <- function(draws, bi_data) {
  if (as.numeric(bi_data$directed) == 1) return(draws)
  if (as.numeric(bi_data$num_random_groups) < 2) return(draws)
  g1_key <- "random_group_mu[1]"
  g2_key <- "random_group_mu[2]"
  if (!all(c(g1_key, g2_key) %in% names(draws))) return(draws)
  offset <- (draws[[g1_key]] - draws[[g2_key]]) / 2
  grp_idx <- bi_data$random_group_index
  for (r in seq_along(grp_idx)) {
    br_k <- paste0("beta_random[", r, "]")
    if (br_k %in% names(draws)) {
      if (grp_idx[r] == 1) draws[[br_k]] <- draws[[br_k]] - offset
      else                  draws[[br_k]] <- draws[[br_k]] + offset
    }
  }
  mu_shared <- (draws[[g1_key]] + draws[[g2_key]]) / 2
  draws[[g1_key]] <- mu_shared
  draws[[g2_key]] <- mu_shared
  draws
}

apply_non_centered_transform <- function(bi_draws, bi_data) {
  if (as.numeric(bi_data$partial_pooling) == 1) {
    sig_key <- if ("edge_sigma" %in% names(bi_draws)) "edge_sigma"
               else if ("edge_sigma[1]" %in% names(bi_draws)) "edge_sigma[1]"
               else NULL
    if (!is.null(sig_key)) {
      sigma_s <- bi_draws[[sig_key]]
      mu_val  <- as.numeric(bi_data$prior_edge_mu)
      for (i in seq_len(as.integer(bi_data$num_edges))) {
        k <- paste0("edge_weight[", i, "]")
        if (k %in% names(bi_draws))
          bi_draws[[k]] <- mu_val + sigma_s * bi_draws[[k]]
      }
    }
  }
  # Undirected: random effects use centered parameterization — no transform needed.
  bi_draws
}

kl_divergence <- function(s, b, n_grid = 512) {
  tryCatch({
    all_v <- c(as.numeric(s), as.numeric(b))
    rng   <- range(all_v)
    pad   <- diff(rng) * 0.15 + 1e-6
    from  <- rng[1] - pad; to <- rng[2] + pad
    d_s   <- density(as.numeric(s), from = from, to = to, n = n_grid)
    d_b   <- density(as.numeric(b), from = from, to = to, n = n_grid)
    eps   <- 1e-10
    p     <- d_s$y + eps;  p <- p / sum(p)
    q     <- d_b$y + eps;  q <- q / sum(q)
    sum(p * log(p / q))
  }, error = function(e) NA_real_)
}

save_multipanel_svg <- function(test_name, stan_draws, bi_draws, out_dir) {
  params <- intersect(names(stan_draws), names(bi_draws))
  params <- params[sapply(params, function(p)
    length(stan_draws[[p]]) > 1 && length(bi_draws[[p]]) > 1)]
  if (!length(params)) return(invisible(NULL))
  ncols <- min(4L, length(params))
  nrows <- ceiling(length(params) / ncols)
  svg_path <- file.path(out_dir, paste0(test_name, ".svg"))
  tryCatch({
    svg(svg_path, width = ncols * 4, height = nrows * 3)
    par(mfrow = c(nrows, ncols), mar = c(3, 3, 2, 1))
    for (param in params) {
      s <- as.numeric(stan_draws[[param]])
      b <- as.numeric(bi_draws[[param]])
      d_s <- density(s); d_b <- density(b)
      xlim <- range(c(d_s$x, d_b$x)); ylim <- range(c(d_s$y, d_b$y))
      plot(d_s, col = "red", lwd = 1.5, main = param,
           xlim = xlim, ylim = ylim, xlab = "", ylab = "", cex.main = 0.7)
      lines(d_b, col = "blue", lwd = 1.5, lty = 2)
    }
    legend("topright", legend = c("Stan", "BI"), col = c("red", "blue"),
           lwd = 1.5, lty = c(1, 2), cex = 0.7, bty = "n")
    dev.off()
    cat("  SVG:", svg_path, "\n")
  }, error = function(e) { try(dev.off(), silent = TRUE) })
}

save_combination_log <- function(test_name, stan_draws, bi_draws, out_dir) {
  all_params <- union(names(stan_draws), names(bi_draws))
  log_path   <- file.path(out_dir, paste0(test_name, "_log.txt"))
  header <- sprintf("%-40s %12s %12s %12s %14s",
                    "Parameter", "Stan_mean", "BI_mean", "Diff", "KL(Stan||BI)")
  sep    <- paste(rep("-", 95), collapse = "")
  rows   <- c(paste0("=== ", test_name, " ==="), header, sep)
  for (param in all_params) {
    s    <- stan_draws[[param]]; b <- bi_draws[[param]]
    s_m  <- if (!is.null(s) && length(s) > 0) mean(as.numeric(s)) else NA
    b_m  <- if (!is.null(b) && length(b) > 0) mean(as.numeric(b)) else NA
    diff <- if (!is.na(s_m) && !is.na(b_m)) s_m - b_m else NA
    kl   <- if (!is.null(s) && !is.null(b) && length(s) > 1 && length(b) > 1)
              kl_divergence(s, b) else NA
    rows <- c(rows, sprintf("%-40s %12.6f %12.6f %12.6f %14.6f",
                            param,
                            ifelse(is.na(s_m), NaN, s_m),
                            ifelse(is.na(b_m), NaN, b_m),
                            ifelse(is.na(diff), NaN, diff),
                            ifelse(is.na(kl),   NaN, kl)))
  }
  # Category-wise mean KL
  categories <- c("edge_weight", "edge_sigma", "beta_fixed", "beta_random",
                  "random_group_mu", "random_group_sigma", "zero_prob")
  rows <- c(rows, "", "Category-wise Mean KL(Stan||BI)")
  for (cat_name in categories) {
    cat_params <- grep(paste0("^", cat_name, "(\\[|$)"), all_params, value = TRUE)
    cat_kls <- sapply(cat_params, function(p) {
      s <- stan_draws[[p]]; b <- bi_draws[[p]]
      if (!is.null(s) && !is.null(b) && length(s) > 1 && length(b) > 1)
        kl_divergence(s, b) else NA
    })
    cat_kls <- cat_kls[!is.na(cat_kls)]
    if (length(cat_kls) > 0)
      rows <- c(rows, sprintf("Mean KL %-25s  %.6f", cat_name, mean(cat_kls)))
  }
  cat(paste(rows, collapse = "\n"), "\n", file = log_path)
  cat("  Log:", log_path, "\n")
}

build_stan_data_bison <- function(stan_data, model_type, partial_pooling, zero_inflated, directed = FALSE) {
  bi <- list(
    num_rows = as.integer(stan_data$num_rows), event = stan_data$event,
    divisor = stan_data$divisor, dyad_ids = stan_data$dyad_ids,
    num_edges = as.integer(stan_data$num_edges), num_fixed = as.integer(stan_data$num_fixed),
    num_random = as.integer(stan_data$num_random),
    num_random_groups = as.integer(stan_data$num_random_groups),
    random_group_index = stan_data$random_group_index,
    design_fixed = stan_data$design_fixed, design_random = stan_data$design_random,
    partial_pooling = as.integer(stan_data$partial_pooling),
    zero_inflated = as.integer(stan_data$zero_inflated),
    directed = as.integer(directed),
    prior_edge_mu = as.numeric(stan_data$prior_edge_mu),
    prior_edge_sigma = as.numeric(stan_data$prior_edge_sigma),
    prior_fixed_mu = as.numeric(stan_data$prior_fixed_mu),
    prior_fixed_sigma = as.numeric(stan_data$prior_fixed_sigma),
    prior_rate_sigma = as.numeric(stan_data$prior_rate_sigma),
    prior_random_mean_mu = as.numeric(stan_data$prior_random_mean_mu),
    prior_random_mean_sigma = as.numeric(stan_data$prior_random_mean_sigma),
    prior_random_std_sigma = as.numeric(stan_data$prior_random_std_sigma),
    prior_zero_prob_alpha = as.numeric(stan_data$prior_zero_prob_alpha),
    prior_zero_prob_beta = as.numeric(stan_data$prior_zero_prob_beta)
  )
  defaults <- list(prior_rate_sigma=1.0, prior_fixed_mu=0.0, prior_fixed_sigma=2.5,
    num_fixed=0L, num_random=0L, num_random_groups=0L, partial_pooling=0L, zero_inflated=0L,
    prior_edge_mu=0.0, prior_edge_sigma=2.5, prior_random_mean_mu=0.0,
    prior_random_mean_sigma=1.0, prior_random_std_sigma=1.0,
    prior_zero_prob_alpha=1.0, prior_zero_prob_beta=1.0)
  for (nm in names(defaults))
    if (length(bi[[nm]]) == 0 || any(is.na(bi[[nm]]))) bi[[nm]] <- defaults[[nm]]
  bi
}

make_jax_data <- function(bi_data, model_type) {
  jd <- bi_data
  jd$dyad_ids <- jnp$array(as.integer(bi_data$dyad_ids), dtype = jnp$int32)
  jd$event    <- jnp$array(as.integer(bi_data$event),    dtype = jnp$int32)
  if (!is.null(bi_data$divisor) && length(bi_data$divisor) > 0)
    jd$divisor <- jnp$array(as.numeric(bi_data$divisor), dtype = jnp$float64)
  if (as.numeric(bi_data$num_fixed) > 0)
    jd$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float64)
  else
    jd$design_fixed <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
  if (as.numeric(bi_data$num_random) > 0) {
    jd$design_random      <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float64)
    jd$random_group_index <- jnp$array(as.integer(bi_data$random_group_index), dtype = jnp$int32)
  } else {
    jd$design_random      <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
    jd$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }
  jd
}

print_priors <- function(bi_data) {
  cat("  Prior values used by BI model:\n")
  cat("    prior_edge_mu          =", bi_data$prior_edge_mu, "\n")
  cat("    prior_edge_sigma       =", bi_data$prior_edge_sigma, "\n")
  cat("    prior_fixed_mu         =", bi_data$prior_fixed_mu, "\n")
  cat("    prior_fixed_sigma      =", bi_data$prior_fixed_sigma, "\n")
  cat("    prior_random_mean_mu   =", bi_data$prior_random_mean_mu, "\n")
  cat("    prior_random_mean_sigma=", bi_data$prior_random_mean_sigma, "\n")
  cat("    prior_random_std_sigma =", bi_data$prior_random_std_sigma, "\n")
}

# ---- main ----
ITER_WARMUP  <- 1000L
ITER_SAMPLES <- 1000L
NUM_CHAINS   <- 8L

test_name <- "binary_full_undirected_zi"
out_dir   <- file.path(RESULTS_DIR, test_name)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n=======================================================\n")
cat("Running:", test_name, "\n")
cat("  Stan + BI: warmup=", ITER_WARMUP, ", samples=", ITER_SAMPLES, ", chains=", NUM_CHAINS, "\n")
cat("=======================================================\n")

set.seed(42)
sim <- simulate_bison_model("binary", aggregated = TRUE,
  location_effect = TRUE, age_diff_effect = TRUE,
  num_nodes = 20, num_locations = 5, max_obs = 10)
df <- sim$df_sim

formula <- as.formula(
  "(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")

# --- Stan fit via bisonR ---
cat("  Fitting Stan model...\n")
fit_bison <- tryCatch(
  bison_model(formula, data = df, model_type = "binary",
    directed = FALSE, partial_pooling = TRUE, zero_inflated = TRUE,
    iter_sampling = 10, iter_warmup = 10, refresh = 0, mc_cores = NUM_CHAINS),
  error = function(e) { cat("  Stan failed:", conditionMessage(e), "\n"); NULL })
if (is.null(fit_bison)) stop("Stan failed")

stan_data_raw <- fit_bison$model_data
bi_data       <- build_stan_data_bison(stan_data_raw, "binary", TRUE, TRUE, directed = FALSE)
print_priors(bi_data)

cat("  Re-running Stan model directly (adapt_delta=0.999)...\n")
t_stan_start <- proc.time()
clean_stan_fit <- fit_bison$stan_model$sample(
  data = stan_data_raw, refresh = 0,
  chains = NUM_CHAINS, parallel_chains = NUM_CHAINS,
  iter_warmup = ITER_WARMUP, iter_sampling = ITER_SAMPLES,
  adapt_delta = 0.999, step_size = 0.05, max_treedepth = 15
)
t_stan_elapsed <- (proc.time() - t_stan_start)[["elapsed"]]

stan_params <- c("edge_weight", "edge_sigma", "beta_fixed", "beta_random",
                 "random_group_mu", "random_group_sigma", "zero_prob")
stan_draws  <- get_stan_draws(clean_stan_fit, stan_params)
stan_draws  <- center_undirected_random_effects(stan_draws, bi_data)

# --- BI fit ---
cat("  Fitting BI model...\n")
bi_draws <- list()
t_bi_start <- proc.time()
tryCatch({
  m$data_on_model <- list(data = make_jax_data(bi_data, "binary"))
  m$fit(bi_model_binary, num_warmup = ITER_WARMUP, num_samples = ITER_SAMPLES,
        num_chains = NUM_CHAINS, target_accept_prob = 0.999)
  raw_bi   <- get_bi_draws(m$posteriors)
  bi_draws <- normalize_bi_names(
    center_undirected_random_effects(apply_non_centered_transform(raw_bi, bi_data), bi_data),
    stan_draws)
}, error = function(e) cat("  BI fit failed:", conditionMessage(e), "\n"))
t_bi_elapsed <- (proc.time() - t_bi_start)[["elapsed"]]

# --- Save outputs ---
save_multipanel_svg(test_name, stan_draws, bi_draws, out_dir)
save_combination_log(test_name, stan_draws, bi_draws, out_dir)

# --- Summary stats ---
cat("\n=== Summary KL stats ===\n")
all_params <- intersect(names(stan_draws), names(bi_draws))
kls <- sapply(all_params, function(p) {
  s <- stan_draws[[p]]; b <- bi_draws[[p]]
  if (length(s) > 1 && length(b) > 1) kl_divergence(s, b) else NA
})
kls <- kls[!is.na(kls)]
cat(sprintf("  N params:  %d\n",   length(kls)))
cat(sprintf("  Mean KL:   %.4f\n", mean(kls)))
cat(sprintf("  Max KL:    %.4f\n", max(kls)))
cat(sprintf("  KL > 0.10: %d\n",   sum(kls > 0.10)))
cat(sprintf("  KL > 0.20: %d\n",   sum(kls > 0.20)))
cat(sprintf("  KL > 0.50: %d\n",   sum(kls > 0.50)))

edge_kls  <- kls[grep("^edge_weight", names(kls))]
rand_kls  <- kls[grep("^beta_random", names(kls))]
other_kls <- kls[!grepl("^edge_weight|^beta_random", names(kls))]
cat(sprintf("\n  edge_weight:   mean=%.4f  max=%.4f\n", mean(edge_kls), max(edge_kls)))
cat(sprintf("  beta_random:   mean=%.4f  max=%.4f\n", mean(rand_kls), max(rand_kls)))
cat(sprintf("  other params:  mean=%.4f  max=%.4f\n", mean(other_kls), max(other_kls)))
cat(sprintf("\n  random_group_mu  Stan: %.4f / %.4f   BI: %.4f / %.4f\n",
  mean(stan_draws[["random_group_mu[1]"]]),
  mean(stan_draws[["random_group_mu[2]"]]),
  mean(bi_draws[["random_group_mu[1]"]]),
  mean(bi_draws[["random_group_mu[2]"]])))
if ("zero_prob[1]" %in% all_params) {
  cat(sprintf("  zero_prob[1]     Stan: %.4f          BI: %.4f\n",
    mean(stan_draws[["zero_prob[1]"]]), mean(bi_draws[["zero_prob[1]"]])))
}


cat("\n=== Timing ===\n")
cat(sprintf("  Stan: %6.1f s  (%4.1f min)\n", t_stan_elapsed, t_stan_elapsed / 60))
cat(sprintf("  BI:   %6.1f s  (%4.1f min)\n", t_bi_elapsed,   t_bi_elapsed   / 60))
cat(sprintf("  Stan/BI ratio: %.1fx\n", t_stan_elapsed / max(t_bi_elapsed, 0.1)))

timing_csv <- file.path(RESULTS_DIR, "timing_summary.csv")
timing_row  <- data.frame(
  model       = test_name,
  num_chains  = NUM_CHAINS,
  iter_warmup = ITER_WARMUP,
  iter_samples= ITER_SAMPLES,
  stan_sec    = round(t_stan_elapsed, 1),
  bi_sec      = round(t_bi_elapsed,   1),
  stan_bi_ratio = round(t_stan_elapsed / max(t_bi_elapsed, 0.1), 1),
  timestamp   = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  stringsAsFactors = FALSE
)
write.table(timing_row, timing_csv,
  append = file.exists(timing_csv), sep = ",",
  row.names = FALSE, col.names = !file.exists(timing_csv), quote = TRUE)

# --- Cleanup Stan CSV output ---
unlink(list.files(stan_out_dir, full.names = TRUE, pattern = "\\.csv$"))

cat("\n  Completed:", test_name, "\n")
