# %%
library(reticulate)
library(BayesianInference)
library(bisonR)
library(cmdstanr)

stan_out_dir <- "Test/Network/Network edges/stan_output"
dir.create(stan_out_dir, showWarnings = FALSE, recursive = TRUE)
options(cmdstanr_output_dir = stan_out_dir)

source("Test/Network/Network edges/Modified simulate_bison_model.R")
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")

source("Test/Network/Network edges/bi_model_binary.R")
source("Test/Network/Network edges/bi_model_count.R")
source("Test/Network/Network edges/bi_model_duration.R")

m <- importBI("cpu")
jnp <- import("jax.numpy")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure") return(x)
  if (is.character(x) || is.factor(x)) {
    warning("Found string/factor, coercing to numeric.")
    x_num <- as.numeric(as.character(x))
    if (!is.null(dim(x))) dim(x_num) <- dim(x)
    return(jnp$array(x_num))
  }
  jnp$array(x)
}

# Extract BI posterior draws as named list (vector or indexed vector)
get_bi_draws <- function(raw_post) {
  np <- import("numpy")
  out <- list()
  for (nm in names(raw_post)) {
    if (grepl("_lik$", nm) || nm == "event" || nm == "event_count") next
    r_arr <- py_to_r(np$asarray(raw_post[[nm]]))
    dims <- dim(r_arr)
    if (is.null(dims) || length(dims) == 1) {
      out[[nm]] <- as.numeric(r_arr)
    } else if (length(dims) == 2) {
      for (i in seq_len(dims[2])) {
        out[[paste0(nm, "[", i, "]")]] <- as.numeric(r_arr[, i])
      }
    }
  }
  return(out)
}

# Extract Stan posterior draws from cmdstanr fit for given parameter names
get_stan_draws <- function(fit, param_names) {
  out <- list()
  for (p in param_names) {
    tryCatch({
      draws <- fit$draws(p, format = "matrix")
      if (ncol(draws) == 1) {
        out[[p]] <- as.numeric(draws[, 1])
      } else {
        for (i in seq_len(ncol(draws))) {
          col_nm <- colnames(draws)[i]
          out[[col_nm]] <- as.numeric(draws[, i])
        }
      }
    }, error = function(e) NULL)
  }
  return(out)
}

# Normalize BI draw names to match Stan convention (e.g. "edge_sigma" -> "edge_sigma[1]")
normalize_bi_names <- function(bi_draws, stan_draws) {
  stan_names <- names(stan_draws)
  out <- bi_draws
  for (nm in names(bi_draws)) {
    indexed_nm <- paste0(nm, "[1]")
    if (indexed_nm %in% stan_names && !(nm %in% stan_names)) {
      out[[indexed_nm]] <- out[[nm]]
      out[[nm]] <- NULL
    }
  }
  out
}

# Produce one SVG per parameter
save_density_plots <- function(test_name, stan_draws, bi_draws, plot_dir) {
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

  all_params <- union(names(stan_draws), names(bi_draws))

  for (param in all_params) {
    s <- stan_draws[[param]]
    b <- bi_draws[[param]]
    if (is.null(s) || is.null(b)) next
    if (length(s) < 2 || length(b) < 2) next

    safe_nm <- gsub("[\\[\\]\\.]", "_", param, perl = TRUE)
    svg_path <- file.path(plot_dir, paste0(safe_nm, ".svg"))

    tryCatch({
      svg(svg_path, width = 6, height = 4)
      d_stan <- density(as.numeric(s))
      d_bi   <- density(as.numeric(b))
      xlim <- range(c(d_stan$x, d_bi$x))
      ylim <- range(c(d_stan$y, d_bi$y))
      plot(d_stan, col = "red", lwd = 2,
           main = paste("Density:", param),
           xlim = xlim, ylim = ylim,
           xlab = "Value", ylab = "Density")
      lines(d_bi, col = "blue", lwd = 2, lty = 2)
      legend("topright",
             legend = c("Stan (BisonR)", "BI"),
             col = c("red", "blue"), lwd = 2, lty = c(1, 2), cex = 0.8)
      dev.off()
    }, error = function(e) {
      try(dev.off(), silent = TRUE)
    })
  }
}

# Append comparison rows to log.txt
append_comparison_log <- function(test_name, stan_draws, bi_draws, log_file) {
  all_params <- union(names(stan_draws), names(bi_draws))
  rows <- character(0)
  rows <- c(rows, paste0("\n=== ", test_name, " ==="))
  rows <- c(rows, sprintf("%-35s %12s %12s %12s", "Parameter", "Stan_mean", "BI_mean", "Difference"))
  rows <- c(rows, paste(rep("-", 75), collapse = ""))

  for (param in all_params) {
    s <- stan_draws[[param]]
    b <- bi_draws[[param]]
    s_mean <- if (!is.null(s) && length(s) > 0) mean(as.numeric(s)) else NA
    b_mean <- if (!is.null(b) && length(b) > 0) mean(as.numeric(b)) else NA
    diff   <- if (!is.na(s_mean) && !is.na(b_mean)) s_mean - b_mean else NA
    rows <- c(rows, sprintf("%-35s %12.6f %12.6f %12.6f",
                            param,
                            ifelse(is.na(s_mean), NaN, s_mean),
                            ifelse(is.na(b_mean), NaN, b_mean),
                            ifelse(is.na(diff),   NaN, diff)))
  }

  cat(paste(rows, collapse = "\n"), "\n", file = log_file, append = TRUE)
}

# Build stan_data from bisonR model_info + priors (binary/count)
build_stan_data_bison <- function(stan_data, model_type, partial_pooling, zero_inflated) {
  bi_data <- list(
    num_rows             = as.integer(stan_data$num_rows),
    event                = stan_data$event,
    divisor              = stan_data$divisor,
    dyad_ids             = stan_data$dyad_ids,
    num_edges            = as.integer(stan_data$num_edges),
    num_fixed            = as.integer(stan_data$num_fixed),
    num_random           = as.integer(stan_data$num_random),
    num_random_groups    = as.integer(stan_data$num_random_groups),
    random_group_index   = stan_data$random_group_index,
    design_fixed         = stan_data$design_fixed,
    design_random        = stan_data$design_random,
    partial_pooling      = as.integer(stan_data$partial_pooling),
    zero_inflated        = as.integer(stan_data$zero_inflated),
    prior_edge_mu        = as.numeric(stan_data$prior_edge_mu),
    prior_edge_sigma     = as.numeric(stan_data$prior_edge_sigma),
    prior_fixed_mu       = as.numeric(stan_data$prior_fixed_mu),
    prior_fixed_sigma    = as.numeric(stan_data$prior_fixed_sigma),
    prior_rate_sigma     = as.numeric(stan_data$prior_rate_sigma),
    prior_random_mean_mu    = as.numeric(stan_data$prior_random_mean_mu),
    prior_random_mean_sigma = as.numeric(stan_data$prior_random_mean_sigma),
    prior_random_std_sigma  = as.numeric(stan_data$prior_random_std_sigma),
    prior_zero_prob_alpha   = as.numeric(stan_data$prior_zero_prob_alpha),
    prior_zero_prob_beta    = as.numeric(stan_data$prior_zero_prob_beta)
  )

  defaults <- list(
    prior_rate_sigma = 1.0, prior_fixed_mu = 0.0, prior_fixed_sigma = 2.5,
    num_fixed = 0L, num_random = 0L, num_random_groups = 0L,
    partial_pooling = 0L, zero_inflated = 0L,
    prior_edge_mu = 0.0, prior_edge_sigma = 2.5,
    prior_random_mean_mu = 0.0, prior_random_mean_sigma = 1.0,
    prior_random_std_sigma = 1.0,
    prior_zero_prob_alpha = 1.0, prior_zero_prob_beta = 1.0
  )
  for (nm in names(defaults)) {
    if (length(bi_data[[nm]]) == 0 || any(is.na(bi_data[[nm]]))) {
      bi_data[[nm]] <- defaults[[nm]]
    }
  }

  bi_data
}

# Build stan_data for duration from simulated dataframe (bisonR doesn't support duration)
build_stan_data_duration <- function(df, formula, directed,
                                     partial_pooling, zero_inflated,
                                     age_diff_effect, random_effect) {
  ns <- getNamespace("bisonR")
  gbd <- get("get_bison_model_data", envir = ns)
  model_info <- gbd(formula = formula, observations = df,
                    directed = directed, model_type = "count")  # use count for data parsing

  md <- model_info$model_data
  num_edges <- as.integer(md$num_edges)
  num_rows  <- as.integer(md$num_rows)

  # Build event_count indexed by dyad (aggregated: each row = one dyad)
  # md$dyad_ids[row_i] = dyad index for that row; accumulate event_count per dyad
  event_count_per_dyad <- integer(num_edges)
  df_event_count <- if (!is.null(df$event_count)) as.integer(df$event_count) else rep(1L, nrow(df))
  for (row_i in seq_len(num_rows)) {
    dyad_j <- as.integer(md$dyad_ids[row_i])
    event_count_per_dyad[dyad_j] <- event_count_per_dyad[dyad_j] + df_event_count[row_i]
  }
  event_count_vec <- pmax(1L, event_count_per_dyad)

  list(
    num_rows          = num_rows,
    num_edges         = num_edges,
    num_fixed         = as.integer(md$num_fixed),
    num_random        = as.integer(md$num_random),
    num_random_groups = as.integer(md$num_random_groups),
    event             = pmax(1L, as.integer(round(as.numeric(md$event)))),
    event_count       = event_count_vec,
    divisor           = as.numeric(md$divisor),
    dyad_ids          = as.integer(md$dyad_ids),
    dyad_ids_receiver = as.integer(md$dyad_ids),  # placeholder, no sender/receiver
    design_fixed      = data.matrix(md$design_fixed),
    design_random     = data.matrix(md$design_random),
    random_group_index = as.integer(md$random_group_index),
    partial_pooling   = as.integer(partial_pooling),
    zero_inflated     = as.integer(zero_inflated),
    sender_receiver   = 0L,
    priors_only       = 0L,
    prior_edge_mu     = 0.0,
    prior_edge_sigma  = 2.0,
    prior_fixed_mu    = 0.0,
    prior_fixed_sigma = 1.0,
    prior_rate_sigma  = 1.0,
    prior_random_mean_mu    = 0.0,
    prior_random_mean_sigma = 1.0,
    prior_random_std_sigma  = 1.0,
    prior_zero_prob_alpha   = 1.0,
    prior_zero_prob_beta    = 1.0
  )
}

# Convert bi_data to JAX arrays
make_jax_data <- function(bi_data, model_type) {
  bi_data_jax <- bi_data

  bi_data_jax$dyad_ids <- jnp$array(as.integer(bi_data$dyad_ids), dtype = jnp$int32)

  if (model_type == "duration") {
    bi_data_jax$event <- jnp$array(as.numeric(bi_data$event), dtype = jnp$float32)
    bi_data_jax$event_count <- jnp$array(as.integer(bi_data$event_count), dtype = jnp$int32)
  } else {
    bi_data_jax$event <- jnp$array(as.integer(bi_data$event), dtype = jnp$int32)
  }

  if (!is.null(bi_data$divisor) && length(bi_data$divisor) > 0) {
    bi_data_jax$divisor <- jnp$array(as.numeric(bi_data$divisor), dtype = jnp$float32)
  }

  if (!is.null(bi_data$num_fixed) && as.numeric(bi_data$num_fixed) > 0) {
    bi_data_jax$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float32)
  } else {
    bi_data_jax$design_fixed <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
  }

  if (!is.null(bi_data$num_random) && as.numeric(bi_data$num_random) > 0) {
    bi_data_jax$design_random     <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float32)
    bi_data_jax$random_group_index <- jnp$array(as.integer(bi_data$random_group_index), dtype = jnp$int32)
  } else {
    bi_data_jax$design_random      <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
    bi_data_jax$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }

  bi_data_jax
}

# Parameters present in each Stan model
stan_param_names <- function(model_type, bi_data) {
  params <- "edge_weight"
  if (as.numeric(bi_data$partial_pooling) == 1) params <- c(params, "edge_sigma")
  if (as.numeric(bi_data$num_fixed)  > 0)       params <- c(params, "beta_fixed")
  if (as.numeric(bi_data$num_random) > 0)        params <- c(params, "beta_random", "random_group_mu", "random_group_sigma")
  if (as.numeric(bi_data$zero_inflated) == 1)    params <- c(params, "zero_prob")
  if (model_type == "duration")                  params <- c(params, "rate")
  params
}

run_all_combinations <- function(model_type, bi_func,
                                  num_nodes = 10, num_locations = 5, max_obs = 5) {
  combinations <- expand.grid(
    fixed          = c(FALSE, TRUE),
    random         = c(FALSE, TRUE),
    partial_pooling = c(FALSE, TRUE),
    zero_inflated  = c(FALSE, TRUE),
    directed       = c(FALSE, TRUE)
  )

  log_file <- paste0("Test/Network/Network edges/log_", model_type, ".txt")
  cat(paste0("=== Comparison log: ", model_type, " ===\nDate: ", Sys.time(), "\n"),
      file = log_file, append = FALSE)

  for (i in seq_len(nrow(combinations))) {
    comb <- combinations[i, ]

    test_name <- paste0(
      model_type,
      "_", if (comb$fixed)          "fixed"      else "no_fixed",
      "_", if (comb$directed)       "directed"   else "undirected",
      "_", if (comb$random)         "random"     else "no_random",
      "_", if (comb$partial_pooling) "pooled"    else "unpooled",
      "_", if (comb$zero_inflated)  "zi"         else "no_zi"
    )
    plot_dir <- file.path("Test/Network/Network edges/density_plots", test_name)

    cat("\n=======================================================\n")
    cat("Running:", test_name, "\n")
    cat("=======================================================\n")

    set.seed(42)
    sim_data <- simulate_bison_model(
      model_type, aggregated = TRUE,
      location_effect  = comb$random,
      age_diff_effect  = comb$fixed,
      num_nodes        = num_nodes,
      num_locations    = num_locations,
      max_obs          = max_obs
    )
    df <- sim_data$df_sim

    formula_str <- "(event | duration) ~ dyad(node_1_id, node_2_id)"
    if (comb$fixed)  formula_str <- paste(formula_str, "+ age_diff")
    if (comb$random) formula_str <- paste(formula_str, "+ (1 | node_1_id) + (1 | node_2_id)")
    formula <- as.formula(formula_str)

    # ---------- Fit Stan model ----------
    if (model_type == "duration") {
      # BisonR doesn't support duration — use cmdstanr directly
      cat("  Building duration stan_data...\n")
      stan_data_raw <- build_stan_data_duration(
        df, formula,
        directed        = comb$directed,
        partial_pooling = comb$partial_pooling,
        zero_inflated   = comb$zero_inflated,
        age_diff_effect = comb$fixed,
        random_effect   = comb$random
      )
      cat("  Compiling/running duration.stan...\n")
      dur_stan_file <- system.file("stan", "duration.stan", package = "bisonR")
      dur_model <- cmdstan_model(dur_stan_file, compile = FALSE, stanc_options = list("O1"))
      dur_model$compile(dir = tempdir())
      tryCatch({
        fit_stan <- dur_model$sample(
          data             = stan_data_raw,
          chains           = 1,
          iter_sampling    = 500,
          iter_warmup      = 500,
          refresh          = 0
        )
        stan_params <- stan_param_names(model_type, stan_data_raw)
        stan_draws  <- get_stan_draws(fit_stan, stan_params)
      }, error = function(e) {
        cat("  Stan duration fit failed:", conditionMessage(e), "\n")
        stan_draws <<- list()
      })
      bi_data <- stan_data_raw
    } else {
      cat("  Fitting bisonR model...\n")
      Sys.sleep(0.5)
      fit_bison <- tryCatch(
        bison_model(
          formula,
          data            = df,
          model_type      = model_type,
          directed        = comb$directed,
          partial_pooling = comb$partial_pooling,
          zero_inflated   = comb$zero_inflated,
          iter_sampling   = 500,
          iter_warmup     = 500,
          refresh         = 0
        ),
        error = function(e) {
          cat("  bison_model failed:", conditionMessage(e), "\n")
          NULL
        }
      )
      if (is.null(fit_bison)) {
        cat("  Skipping", test_name, "\n")
        next
      }
      stan_data_raw <- fit_bison$model_data
      bi_data       <- build_stan_data_bison(stan_data_raw, model_type,
                                              comb$partial_pooling, comb$zero_inflated)
      stan_params   <- stan_param_names(model_type, bi_data)
      stan_draws    <- get_stan_draws(fit_bison$fit, stan_params)
    }

    # ---------- Fit BI model ----------
    cat("  Fitting BI model...\n")
    bi_data_jax <- make_jax_data(bi_data, model_type)
    m$data_on_model <- list(data = bi_data_jax)
    tryCatch({
      m$fit(bi_func, num_warmup = 500L, num_samples = 500L, num_chains = 1L)
      bi_draws <- get_bi_draws(m$posteriors)
    }, error = function(e) {
      cat("  BI fit failed:", conditionMessage(e), "\n")
      bi_draws <<- list()
    })

    # ---------- Compare & save ----------
    cat("  Saving density plots and log...\n")
    bi_draws_norm <- normalize_bi_names(bi_draws, stan_draws)
    save_density_plots(test_name, stan_draws, bi_draws_norm, plot_dir)
    append_comparison_log(test_name, stan_draws, bi_draws_norm, log_file)

    cat("  Completed", test_name, "\n")
    Sys.sleep(1)
  }
}


# %%
run_all_combinations(model_type = "binary",   bi_func = bi_model_binary,
                     num_nodes = 50, num_locations = 5, max_obs = 20)
run_all_combinations(model_type = "count",    bi_func = bi_model_count,
                     num_nodes = 50, num_locations = 5, max_obs = 20)
run_all_combinations(model_type = "duration", bi_func = bi_model_duration,
                     num_nodes = 50, num_locations = 5, max_obs = 20)

cat("\nAll tests completed!\n")
cat("Density plots: Test/Network/Network edges/density_plots/\n")
cat("Logs: Test/Network/Network edges/log_binary.txt, log_count.txt, log_duration.txt\n")
