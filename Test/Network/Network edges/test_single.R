setwd("/home/sebastian_sosa/BI")

library(reticulate)
library(BayesianInference)
library(bisonR)
library(cmdstanr)

BASE        <- "/home/sebastian_sosa/BI/Test/Network/Network edges"
RESULTS_DIR <- file.path(BASE, "results")
stan_out_dir <- file.path(BASE, "stan_output")
dir.create(RESULTS_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(stan_out_dir, showWarnings = FALSE, recursive = TRUE)
options(cmdstanr_output_dir = stan_out_dir)

source(file.path(BASE, "Modified simulate_bison_model.R"))
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")
source(file.path(BASE, "bi_model_binary.R"))
source(file.path(BASE, "bi_model_count.R"))
source(file.path(BASE, "bi_model_duration.R"))

m   <- importBI("cpu")
jnp <- import("jax.numpy")

# ---- helpers ----
get_bi_draws <- function(raw_post) {
  np <- import("numpy")
  out <- list()
  for (nm in names(raw_post)) {
    if (grepl("_lik$", nm) || nm == "event" || nm == "event_count") next
    r_arr <- py_to_r(np$asarray(raw_post[[nm]]))
    dims   <- dim(r_arr)
    # 3D (chains, samples, params) -> 2D (total_samples, params)
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

# KL(Stan || BI) via KDE on shared grid
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

# Multi-panel SVG: all parameters in one file
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

# Per-combination log with 4 columns
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
  cat(paste(rows, collapse = "\n"), "\n", file = log_path)
  cat("  Log:", log_path, "\n")
}

build_stan_data_bison <- function(stan_data, model_type, partial_pooling, zero_inflated) {
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

build_stan_data_duration <- function(df, formula, directed, partial_pooling, zero_inflated) {
  ns  <- getNamespace("bisonR")
  gbd <- get("get_bison_model_data", envir = ns)
  mi  <- gbd(formula = formula, observations = df, directed = directed, model_type = "count")
  md  <- mi$model_data
  ne  <- as.integer(md$num_edges); nr <- as.integer(md$num_rows)
  ec  <- integer(ne)
  dfc <- if (!is.null(df$event_count)) as.integer(df$event_count) else rep(1L, nrow(df))
  for (i in seq_len(nr)) ec[as.integer(md$dyad_ids[i])] <- ec[as.integer(md$dyad_ids[i])] + dfc[i]
  list(
    num_rows=nr, num_edges=ne,
    num_fixed=as.integer(md$num_fixed), num_random=as.integer(md$num_random),
    num_random_groups=as.integer(md$num_random_groups),
    event=pmax(1L, as.integer(round(as.numeric(md$event)))),
    event_count=pmax(1L, ec), divisor=as.numeric(md$divisor),
    dyad_ids=as.integer(md$dyad_ids), dyad_ids_receiver=as.integer(md$dyad_ids),
    design_fixed=data.matrix(md$design_fixed), design_random=data.matrix(md$design_random),
    random_group_index=as.integer(md$random_group_index),
    partial_pooling=as.integer(partial_pooling), zero_inflated=as.integer(zero_inflated),
    sender_receiver=0L, priors_only=0L,
    prior_edge_mu=0.0, prior_edge_sigma=2.0, prior_fixed_mu=0.0, prior_fixed_sigma=2.5,
    prior_rate_sigma=1.0, prior_random_mean_mu=0.0, prior_random_mean_sigma=0.1,
    prior_random_std_sigma=1.0, prior_zero_prob_alpha=1.0, prior_zero_prob_beta=1.0
  )
}

make_jax_data <- function(bi_data, model_type) {
  jd <- bi_data
  jd$dyad_ids <- jnp$array(as.integer(bi_data$dyad_ids), dtype = jnp$int32)
  if (model_type == "duration") {
    jd$event       <- jnp$array(as.numeric(bi_data$event),       dtype = jnp$float32)
    jd$event_count <- jnp$array(as.integer(bi_data$event_count), dtype = jnp$int32)
  } else {
    jd$event <- jnp$array(as.integer(bi_data$event), dtype = jnp$int32)
  }
  if (!is.null(bi_data$divisor) && length(bi_data$divisor) > 0)
    jd$divisor <- jnp$array(as.numeric(bi_data$divisor), dtype = jnp$float32)
  if (as.numeric(bi_data$num_fixed) > 0)
    jd$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float32)
  else
    jd$design_fixed <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
  if (as.numeric(bi_data$num_random) > 0) {
    jd$design_random      <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float32)
    jd$random_group_index <- jnp$array(as.integer(bi_data$random_group_index), dtype = jnp$int32)
  } else {
    jd$design_random      <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
    jd$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }
  jd
}

stan_param_names <- function(model_type, bi_data) {
  p <- "edge_weight"
  if (as.numeric(bi_data$partial_pooling) == 1) p <- c(p, "edge_sigma")
  if (as.numeric(bi_data$num_fixed)  > 0)       p <- c(p, "beta_fixed")
  if (as.numeric(bi_data$num_random) > 0)        p <- c(p, "beta_random", "random_group_mu", "random_group_sigma")
  if (as.numeric(bi_data$zero_inflated) == 1)    p <- c(p, "zero_prob")
  if (model_type == "duration")                  p <- c(p, "rate")
  p
}

# ---- main ----
run_focused <- function(model_type, bi_func, directed, zero_inflated,
                        num_nodes = 20, num_locations = 5, max_obs = 10) {

  test_name <- paste0(model_type,
    "_full_", if (directed) "directed" else "undirected",
    "_",      if (zero_inflated) "zi" else "no_zi")

  out_dir <- file.path(RESULTS_DIR, test_name)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat("\n=======================================================\n")
  cat("Running:", test_name, "\n")
  cat("=======================================================\n")

  set.seed(42)
  sim <- simulate_bison_model(model_type, aggregated = TRUE,
    location_effect = TRUE, age_diff_effect = TRUE,
    num_nodes = num_nodes, num_locations = num_locations, max_obs = max_obs)
  df <- sim$df_sim

  formula <- as.formula(
    "(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")

  # --- Stan fit ---
  if (model_type == "duration") {
    stan_data_raw <- build_stan_data_duration(df, formula, directed,
                                              partial_pooling = TRUE, zero_inflated = zero_inflated)
    dur_file  <- system.file("stan", "duration.stan", package = "bisonR")
    dur_model <- cmdstan_model(dur_file, compile = FALSE, stanc_options = list("O1"))
    dur_model$compile(dir = tempdir())
    fit_stan  <- tryCatch({
      stan_data_raw$prior_random_mean_sigma <- 0.1
      dur_model$sample(data = stan_data_raw, chains = 4,
                       iter_sampling = 500, iter_warmup = 500, refresh = 0)
    },
      error = function(e) { cat("  Stan failed:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit_stan)) return(invisible(NULL))
    stan_params <- stan_param_names(model_type, stan_data_raw)
    stan_draws  <- get_stan_draws(fit_stan, stan_params)
    bi_data     <- stan_data_raw
  } else {
    fit_bison <- tryCatch({
      def_edge_sigma <- if (model_type == "binary") 2.5 else 1.0
      bison_model(formula, data = df, model_type = model_type,
        directed = directed, partial_pooling = TRUE, zero_inflated = zero_inflated,
        priors = list(
          prior_edge_mu = 0.0, prior_edge_sigma = def_edge_sigma,
          prior_fixed_mu = 0.0, prior_fixed_sigma = 2.5,
          prior_random_mean_mu = 0.0, prior_random_mean_sigma = 0.1,
          prior_random_std_sigma = 1.0, prior_zero_prob_alpha = 1.0, prior_zero_prob_beta = 1.0,
          prior_rate_sigma = 1.0
        ),
        iter_sampling = 500, iter_warmup = 500, refresh = 0)
    },
      error = function(e) { cat("  bison_model failed:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit_bison)) return(invisible(NULL))
    stan_data_raw <- fit_bison$model_data
    bi_data       <- build_stan_data_bison(stan_data_raw, model_type, TRUE, zero_inflated)
    stan_params   <- stan_param_names(model_type, bi_data)
    stan_draws    <- get_stan_draws(fit_bison$fit, stan_params)
  }

  # --- BI fit ---
  cat("  Fitting BI model...\n")
  bi_draws <- list()
  tryCatch({
    m$data_on_model <- list(data = make_jax_data(bi_data, model_type))
    m$fit(bi_func, num_warmup = 500L, num_samples = 500L, num_chains = 4L)
    bi_draws <- normalize_bi_names(get_bi_draws(m$posteriors), stan_draws)
  }, error = function(e) cat("  BI fit failed:", conditionMessage(e), "\n"))

  # --- Save outputs ---
  save_multipanel_svg(test_name, stan_draws, bi_draws, out_dir)
  save_combination_log(test_name, stan_draws, bi_draws, out_dir)
run_focused("duration", bi_model_duration, directed=FALSE, zero_inflated=FALSE, num_nodes=20)
