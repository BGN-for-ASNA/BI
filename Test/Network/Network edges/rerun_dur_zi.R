setwd("C:/Users/Sosa/Documents/BI")

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
source(file.path(BASE, "bi_model_duration.R"))

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
      s <- as.numeric(stan_draws[[param]]); b <- as.numeric(bi_draws[[param]])
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
  cat(paste(rows, collapse = "\n"), "\n", file = log_path)
  cat("  Log:", log_path, "\n")
}

build_stan_data_duration <- function(df, formula, partial_pooling, zero_inflated) {
  ns  <- getNamespace("bisonR")
  gbd <- get("get_bison_model_data", envir = ns)
  mi  <- gbd(formula = formula, observations = df, directed = FALSE, model_type = "count")
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
    prior_edge_mu=0.0, prior_edge_sigma=2.0, prior_fixed_mu=0.0, prior_fixed_sigma=1.0,
    prior_rate_sigma=1.0, prior_random_mean_mu=0.0, prior_random_mean_sigma=1.0,
    prior_random_std_sigma=1.0, prior_zero_prob_alpha=1.0, prior_zero_prob_beta=1.0
  )
}

make_jax_data <- function(bi_data) {
  jd <- bi_data
  jd$dyad_ids    <- jnp$array(as.integer(bi_data$dyad_ids), dtype = jnp$int32)
  jd$event <- jnp$array(as.numeric(bi_data$event), dtype = jnp$float32)
  jd$event_count <- jnp$array(as.integer(bi_data$event_count), dtype = jnp$int32)
  jd$divisor     <- jnp$array(as.numeric(bi_data$divisor),  dtype = jnp$float32)
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

test_name <- "duration_full_undirected_zi"
out_dir   <- file.path(RESULTS_DIR, test_name)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n=======================================================\n")
cat("Running:", test_name, "\n")
cat("=======================================================\n")

set.seed(42)
sim <- simulate_bison_model("duration", aggregated = TRUE,
  location_effect = TRUE, age_diff_effect = TRUE,
  num_nodes = 20, num_locations = 5, max_obs = 10)
df <- sim$df_sim

formula <- as.formula(
  "(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")

stan_data_raw <- build_stan_data_duration(df, formula,
                                          partial_pooling = TRUE, zero_inflated = TRUE)
dur_file  <- system.file("stan", "duration.stan", package = "bisonR")
dur_model <- cmdstan_model(dur_file, compile = FALSE, stanc_options = list("O1"))
dur_model$compile(dir = tempdir())
fit_stan  <- tryCatch(
  dur_model$sample(data = stan_data_raw, chains = 4,
                   iter_sampling = 500, iter_warmup = 500, refresh = 0),
  error = function(e) { cat("  Stan failed:", conditionMessage(e), "\n"); NULL })
if (is.null(fit_stan)) stop("Stan failed")

param_names <- c("edge_weight", "edge_sigma", "beta_fixed", "beta_random",
                 "random_group_mu", "random_group_sigma", "rate", "zero_prob")
stan_draws  <- get_stan_draws(fit_stan, param_names)

cat("  Fitting BI model...\n")
bi_draws <- list()
tryCatch({
  m$data_on_model <- list(data = make_jax_data(stan_data_raw))
  m$fit(bi_model_duration, num_warmup = 500L, num_samples = 500L, num_chains = 4L)
  bi_draws <- normalize_bi_names(get_bi_draws(m$posteriors), stan_draws)
}, error = function(e) {
  cat("  BI fit failed:", conditionMessage(e), "\n")
  tryCatch(cat("  py_last_error:", py_last_error()$message, "\n"), error=function(e2) NULL)
})

save_multipanel_svg(test_name, stan_draws, bi_draws, out_dir)
save_combination_log(test_name, stan_draws, bi_draws, out_dir)
cat("  Completed:", test_name, "\n")
