# %%
library(reticulate)
library(BayesianInference)
library(bisonR)
library(brms)
library(cmdstanr)

# Set stable output directory for Stan to avoid /tmp issues in WSL
stan_out_dir <- "Test/Network edges/stan_output"
dir.create(stan_out_dir, showWarnings = FALSE, recursive = TRUE)
options(cmdstanr_output_dir = stan_out_dir)

dir.create("Test/Network edges/density_plots", showWarnings = FALSE)

# Source BI models
source("Test/Network edges/Modified simulate_bison_model.R")
# Overwrite bisonR::simulate_bison_model with the modified version
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")

source("Test/Network edges/bi_model_binary.R")
source("Test/Network edges/bi_model_count.R")
source("Test/Network edges/bi_model_duration.R")

m <- importBI("cpu")
jnp <- import("jax.numpy")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure") {
    return(x)
  }

  # Forcibly convert character or factor to numeric if they sneaked in, keeping structure
  if (is.character(x) || is.factor(x)) {
    warning("Found string/factor, coercing to numeric.")
    x_num <- as.numeric(as.character(x))
    # reshape back if it was an array
    if (!is.null(dim(x))) dim(x_num) <- dim(x)
    return(jnp$array(x_num))
  }

  jnp$array(x)
}

# Extracts BI draws and maps them to Stan names
get_bi_draws <- function(raw_post) {
  np <- import("numpy")
  out <- list()
  for (nm in names(raw_post)) {
    if (grepl("_lik$", nm) || nm == "event") next
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


# Extracts brms draws
get_brms_draws <- function(fit, variables) {
  draws <- as_draws_matrix(fit$fit)
  cols <- colnames(draws)
  out <- list()
  for (v in variables) {
    # Match variables like r_node_1_id[1,Intercept] to edge_weight[1]?
    # Wait: brms renames parameters. If edge weights are modeled as random effects,
    # brms output will use `r_dyad_id[1,Intercept]`.
    # Actually bisonR handles standardizing names? Wait, bisonR uses brms format.
    # To keep it comparable, we just match what brms produces for standard effects.
    idx <- grep(v, cols, fixed = TRUE)
    for (i in idx) {
      # Extract raw draws
      out[[cols[i]]] <- as.numeric(draws[, i])
    }
  }
  return(out)
}

plot_densities <- function(model_name, bi_draws, brms_draws,
                           bi_prefix = "edge_weight",
                           brms_prefix = "edge_weight",
                           true_values = NULL) {
  # Create directory if it doesn't exist
  dir.create("Test/Network edges/density_plots", showWarnings = FALSE, recursive = TRUE)

  num_plots <- length(brms_prefix)
  num_cols <- 4
  num_rows <- ceiling(num_plots / num_cols)

  # Adjust PDF size based on grid
  pdf_width <- num_cols * 4
  pdf_height <- num_rows * 4

  pdf(paste0("Test/Network edges/density_plots/", model_name, "_densities.pdf"),
    width = pdf_width, height = pdf_height
  )
  par(mfrow = c(num_rows, num_cols))

  for (i in seq_len(num_plots)) {
    bi_par <- paste0(bi_prefix, "[", i, "]")
    brms_par <- brms_prefix[i]

    if (!is.null(bi_draws[[bi_par]]) && !is.null(brms_draws[[brms_par]])) {
      d_bi <- density(bi_draws[[bi_par]])
      d_brms <- density(brms_draws[[brms_par]])

      xlim <- range(c(d_bi$x, d_brms$x))
      if (!is.null(true_values) && length(true_values) >= i) {
        xlim <- range(c(xlim, true_values[i]))
      }

      plot(d_bi,
        main = paste("Comparison:", brms_par),
        col = "blue", lwd = 2, xlim = xlim,
        xlab = "Value", ylab = "Density"
      )
      lines(d_brms, col = "red", lwd = 2, lty = 2)

      if (!is.null(true_values) && length(true_values) >= i) {
        abline(v = true_values[i], col = "darkgreen", lwd = 2, lty = 3)
      }

      legend("topright",
        legend = c("BI", "Stan (bisonR)", "True Value"),
        col = c("blue", "red", "darkgreen"), lwd = 2, lty = c(1, 2, 3), cex = 0.8
      )
    }
  }
  dev.off()
}


test_pipeline <- function(model_type, bi_func) {
  cat("\nRunning", model_type, "pipeline...\n")

  # Simulate small network
  set.seed(42)
  sim_data <- simulate_bison_model(model_type, aggregated = TRUE)
  df <- sim_data$df_sim

  # Fit bisonR model
  # For binary_conjugate, bisonR skips Stan. We must use standard model_type
  cat("  Fitting bisonR model...\n")
  fit_bison <- bison_model(
    (event | duration) ~ dyad(node_1_id, node_2_id),
    data = df,
    model_type = model_type,
    iter_sampling = 500, iter_warmup = 500
  )

  # Extract Stan data natively produced by bisonR
  cat("  Extracting Stan data...\n")
  stan_data <- fit_bison$model_data

  cat("  Fitting BI model...\n")
  bi_data <- list(
    num_rows = as.integer(stan_data$num_rows),
    event = stan_data$event,
    divisor = stan_data$divisor,
    dyad_ids = stan_data$dyad_ids,
    num_edges = as.integer(stan_data$num_edges),
    num_fixed = as.integer(stan_data$num_fixed),
    num_random = as.integer(stan_data$num_random),
    num_random_groups = as.integer(stan_data$num_random_groups),
    random_group_index = stan_data$random_group_index,
    design_fixed = stan_data$design_fixed,
    design_random = stan_data$design_random,
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
  if (length(bi_data$prior_rate_sigma) == 0 || is.na(bi_data$prior_rate_sigma)) bi_data$prior_rate_sigma <- 1.0
  if (length(bi_data$prior_fixed_mu) == 0 || is.na(bi_data$prior_fixed_mu)) bi_data$prior_fixed_mu <- 0.0
  if (length(bi_data$prior_fixed_sigma) == 0 || is.na(bi_data$prior_fixed_sigma)) bi_data$prior_fixed_sigma <- 1.0
  if (length(bi_data$num_fixed) == 0 || is.na(bi_data$num_fixed)) bi_data$num_fixed <- 0L
  if (length(bi_data$num_random) == 0 || is.na(bi_data$num_random)) bi_data$num_random <- 0L
  if (length(bi_data$num_random_groups) == 0 || is.na(bi_data$num_random_groups)) bi_data$num_random_groups <- 0L
  if (length(bi_data$partial_pooling) == 0 || is.na(bi_data$partial_pooling)) bi_data$partial_pooling <- 0L
  if (length(bi_data$zero_inflated) == 0 || is.na(bi_data$zero_inflated)) bi_data$zero_inflated <- 0L
  if (length(bi_data$prior_edge_mu) == 0 || is.na(bi_data$prior_edge_mu)) bi_data$prior_edge_mu <- 0.0
  if (length(bi_data$prior_edge_sigma) == 0 || is.na(bi_data$prior_edge_sigma)) bi_data$prior_edge_sigma <- 2.0
  if (length(bi_data$prior_random_mean_mu) == 0 || is.na(bi_data$prior_random_mean_mu)) bi_data$prior_random_mean_mu <- 0.0
  if (length(bi_data$prior_random_mean_sigma) == 0 || is.na(bi_data$prior_random_mean_sigma)) bi_data$prior_random_mean_sigma <- 1.0
  if (length(bi_data$prior_random_std_sigma) == 0 || is.na(bi_data$prior_random_std_sigma)) bi_data$prior_random_std_sigma <- 2.0
  if (length(bi_data$prior_zero_prob_alpha) == 0 || is.na(bi_data$prior_zero_prob_alpha)) bi_data$prior_zero_prob_alpha <- 1.0
  if (length(bi_data$prior_zero_prob_beta) == 0 || is.na(bi_data$prior_zero_prob_beta)) bi_data$prior_zero_prob_beta <- 1.0

  # For duration model, event_count might need to be explicitly set if missing
  if (model_type == "duration") {
    if (!is.null(stan_data$event_count)) {
      bi_data$event_count <- stan_data$event_count
    } else {
      bi_data$event_count <- stan_data$event
    }
  }

  # Only convert actual arrays/data tensors to JAX
  bi_data_jax <- bi_data

  bi_data_jax$dyad_ids <- jnp$array(as.integer(as.character(bi_data$dyad_ids)), dtype = jnp$int32)
  if (model_type == "duration") {
    bi_data_jax$event <- jnp$array(as.numeric(as.character(bi_data$event)), dtype = jnp$float32)
  } else {
    bi_data_jax$event <- jnp$array(as.integer(as.character(bi_data$event)), dtype = jnp$int32)
  }
  if (!is.null(bi_data$event_count)) {
    bi_data_jax$event_count <- jnp$array(as.integer(as.character(bi_data$event_count)), dtype = jnp$int32)
  }

  if (!is.null(bi_data$divisor)) {
    bi_data_jax$divisor <- jnp$array(as.integer(as.character(bi_data$divisor)), dtype = jnp$int32)
  }

  if (bi_data$num_fixed > 0) {
    bi_data_jax$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float32)
  } else {
    bi_data_jax$design_fixed <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
  }

  if (bi_data$num_random > 0) {
    bi_data_jax$design_random <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float32)
    bi_data_jax$random_group_index <- jnp$array(as.integer(as.character(bi_data$random_group_index)), dtype = jnp$int32)
  } else {
    bi_data_jax$design_random <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
    bi_data_jax$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }

  m$data_on_model <- list(data = bi_data_jax)
  m$fit(bi_func, num_warmup = 1000L, num_samples = 1000L, num_chains = 1L)

  cat("  Extracting draws and plotting...\n")

  # bisonR extracts draws into `edge_samples` directly! It's an array of (samples, dyads)
  # We can just use that instead of brms functions to get the exact edge weights!
  # fit_bison$edge_samples is a matrix of (iterations, dyads)
  bison_edge_samples <- fit_bison$edge_samples

  bi_draws <- get_bi_draws(m$posteriors)

  # Format brms draws to match our plot function:
  brms_draws <- list()
  for (i in seq_len(ncol(bison_edge_samples))) {
    brms_draws[[paste0("edge_weight[", i, "]")]] <- as.numeric(bison_edge_samples[, i])
  }

  brms_names <- paste0("edge_weight[", seq_len(ncol(bison_edge_samples)), "]")

  plot_densities(model_type, bi_draws, brms_draws,
    bi_prefix = "edge_weight",
    brms_prefix = brms_names
  )

  cat("  Completed", model_type, "\n")
}

test_complex_pipeline <- function(model_type, bi_func) {
  cat("\nRunning", model_type, "complex pipeline...\n")
  model_name <- paste0(model_type, "_complex")

  set.seed(42)
  # Simulated data has `age_diff`, `age_1`, `age_2` columns automatically
  sim_data <- simulate_bison_model(model_type, aggregated = TRUE)
  df <- sim_data$df_sim

  cat("  Fitting bisonR complex model...\n")
  fit_bison <- bison_model(
    (event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id),
    data = df,
    model_type = model_type,
    directed = TRUE,
    partial_pooling = TRUE,
    zero_inflated = TRUE,
    iter_sampling = 500, iter_warmup = 500
  )

  stan_data <- fit_bison$model_data
  cat("  Fitting BI complex model...\n")

  bi_data <- list(
    num_rows = as.integer(stan_data$num_rows),
    event = stan_data$event,
    divisor = stan_data$divisor,
    dyad_ids = stan_data$dyad_ids,
    num_edges = as.integer(stan_data$num_edges),
    num_fixed = as.integer(stan_data$num_fixed),
    num_random = as.integer(stan_data$num_random),
    num_random_groups = as.integer(stan_data$num_random_groups),
    random_group_index = stan_data$random_group_index,
    design_fixed = stan_data$design_fixed,
    design_random = stan_data$design_random,
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
  if (length(bi_data$prior_rate_sigma) == 0 || is.na(bi_data$prior_rate_sigma)) bi_data$prior_rate_sigma <- 1.0
  if (length(bi_data$prior_fixed_mu) == 0 || is.na(bi_data$prior_fixed_mu)) bi_data$prior_fixed_mu <- 0.0
  if (length(bi_data$prior_fixed_sigma) == 0 || is.na(bi_data$prior_fixed_sigma)) bi_data$prior_fixed_sigma <- 1.0
  if (length(bi_data$num_fixed) == 0 || is.na(bi_data$num_fixed)) bi_data$num_fixed <- 0L
  if (length(bi_data$num_random) == 0 || is.na(bi_data$num_random)) bi_data$num_random <- 0L
  if (length(bi_data$num_random_groups) == 0 || is.na(bi_data$num_random_groups)) bi_data$num_random_groups <- 0L
  if (length(bi_data$partial_pooling) == 0 || is.na(bi_data$partial_pooling)) bi_data$partial_pooling <- 0L
  if (length(bi_data$zero_inflated) == 0 || is.na(bi_data$zero_inflated)) bi_data$zero_inflated <- 0L
  if (length(bi_data$prior_edge_mu) == 0 || is.na(bi_data$prior_edge_mu)) bi_data$prior_edge_mu <- 0.0
  if (length(bi_data$prior_edge_sigma) == 0 || is.na(bi_data$prior_edge_sigma)) bi_data$prior_edge_sigma <- 2.0
  if (length(bi_data$prior_random_mean_mu) == 0 || is.na(bi_data$prior_random_mean_mu)) bi_data$prior_random_mean_mu <- 0.0
  if (length(bi_data$prior_random_mean_sigma) == 0 || is.na(bi_data$prior_random_mean_sigma)) bi_data$prior_random_mean_sigma <- 1.0
  if (length(bi_data$prior_random_std_sigma) == 0 || is.na(bi_data$prior_random_std_sigma)) bi_data$prior_random_std_sigma <- 2.0
  if (length(bi_data$prior_zero_prob_alpha) == 0 || is.na(bi_data$prior_zero_prob_alpha)) bi_data$prior_zero_prob_alpha <- 1.0
  if (length(bi_data$prior_zero_prob_beta) == 0 || is.na(bi_data$prior_zero_prob_beta)) bi_data$prior_zero_prob_beta <- 1.0

  bi_data_jax <- bi_data
  bi_data_jax$dyad_ids <- jnp$array(as.integer(as.character(bi_data$dyad_ids)), dtype = jnp$int32)

  if (model_type == "duration") {
    bi_data_jax$event <- jnp$array(as.numeric(as.character(bi_data$event)), dtype = jnp$float32)
    if (!is.null(stan_data$event_count)) {
      bi_data_jax$event_count <- jnp$array(as.integer(as.character(stan_data$event_count)), dtype = jnp$int32)
    } else {
      bi_data_jax$event_count <- jnp$array(as.integer(as.character(stan_data$event)), dtype = jnp$int32)
    }
  } else {
    bi_data_jax$event <- jnp$array(as.integer(as.character(bi_data$event)), dtype = jnp$int32)
  }
  if (!is.null(bi_data$divisor)) {
    bi_data_jax$divisor <- jnp$array(as.integer(as.character(bi_data$divisor)), dtype = jnp$int32)
  }

  if (bi_data$num_fixed > 0) {
    bi_data_jax$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float32)
  } else {
    bi_data_jax$design_fixed <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
  }

  if (bi_data$num_random > 0) {
    bi_data_jax$design_random <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float32)
    bi_data_jax$random_group_index <- jnp$array(as.integer(as.character(bi_data$random_group_index)), dtype = jnp$int32)
  } else {
    bi_data_jax$design_random <- jnp$zeros(as.integer(c(bi_data$num_rows, 0L)))
    bi_data_jax$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }

  m$data_on_model <- list(data = bi_data_jax)
  m$fit(bi_func, num_warmup = 300L, num_samples = 300L, num_chains = 1L)

  cat("  Extracting draws and plotting...\n")
  bison_edge_samples <- fit_bison$edge_samples
  bi_draws <- get_bi_draws(m$posteriors)
  brms_draws <- list()
  for (i in seq_len(ncol(bison_edge_samples))) {
    brms_draws[[paste0("edge_weight[", i, "]")]] <- as.numeric(bison_edge_samples[, i])
  }
  brms_names <- paste0("edge_weight[", seq_len(ncol(bison_edge_samples)), "]")
  plot_densities(model_name, bi_draws, brms_draws, bi_prefix = "edge_weight", brms_prefix = brms_names)

  cat("  Completed", model_name, "\n")
}

run_all_combinations <- function(model_type, bi_func) {
  combinations <- expand.grid(
    fixed = c(FALSE, TRUE),
    random = c(FALSE, TRUE),
    partial_pooling = c(FALSE, TRUE),
    zero_inflated = c(FALSE, TRUE),
    directed = c(FALSE, TRUE)
  )

  for (i in seq_len(nrow(combinations))) {
    comb <- combinations[i, ]

    # Generate a unique name for this test
    test_name <- paste0(
      model_type,
      "_", if (comb$fixed) "fixed" else "no_fixed",
      "_", if (comb$directed) "directed" else "undirected",
      "_", if (comb$random) "random" else "no_random",
      "_", if (comb$partial_pooling) "pooled" else "unpooled",
      "_", if (comb$zero_inflated) "zi" else "no_zi"
    )

    cat("\n=======================================================\n")
    cat("Running combinatorial pipeline:", test_name, "\n")
    cat("=======================================================\n")

    set.seed(42)
    # The simulate_bison_model outputs a standard dataframe whether we use covariates or not
    sim_data <- simulate_bison_model(
      model_type,
      aggregated = TRUE,
      location_effect = comb$random,
      age_diff_effect = comb$fixed
    )
    df <- sim_data$df_sim

    # Construct formula dynamically
    formula_str <- "(event | duration) ~ dyad(node_1_id, node_2_id)"
    if (comb$fixed) {
      formula_str <- paste(formula_str, "+ age_diff")
    }
    if (comb$random) {
      formula_str <- paste(formula_str, "+ (1 | node_1_id) + (1 | node_2_id)")
    }
    formula <- as.formula(formula_str)

    # Small sleep and log to ensure filesystem stability and clear intent
    cat("  Preparing to fit model (waiting for filesystem stability)...\n")
    Sys.sleep(0.5)

    cat("  Fitting bisonR model...\n")
    fit_bison <- bison_model(
      formula,
      data = df,
      model_type = model_type,
      directed = comb$directed, # We always test directed for combinations
      partial_pooling = comb$partial_pooling,
      zero_inflated = comb$zero_inflated,
      iter_sampling = 1000, iter_warmup = 1000,
    )

    stan_data <- fit_bison$model_data
    cat("  Fitting BI model...\n")

    bi_data <- list(
      num_rows = as.integer(stan_data$num_rows),
      event = stan_data$event,
      divisor = stan_data$divisor,
      dyad_ids = stan_data$dyad_ids,
      num_edges = as.integer(stan_data$num_edges),
      num_fixed = as.integer(stan_data$num_fixed),
      num_random = as.integer(stan_data$num_random),
      num_random_groups = as.integer(stan_data$num_random_groups),
      random_group_index = stan_data$random_group_index,
      design_fixed = stan_data$design_fixed,
      design_random = stan_data$design_random,
      partial_pooling = as.integer(stan_data$partial_pooling),
      zero_inflated = as.integer(stan_data$zero_inflated)
    )
    if (length(bi_data$num_fixed) == 0 || is.na(bi_data$num_fixed)) bi_data$num_fixed <- 0L
    if (length(bi_data$num_random) == 0 || is.na(bi_data$num_random)) bi_data$num_random <- 0L
    if (length(bi_data$num_random_groups) == 0 || is.na(bi_data$num_random_groups)) bi_data$num_random_groups <- 0L
    if (length(bi_data$partial_pooling) == 0 || is.na(bi_data$partial_pooling)) bi_data$partial_pooling <- 0L
    if (length(bi_data$zero_inflated) == 0 || is.na(bi_data$zero_inflated)) bi_data$zero_inflated <- 0L

    bi_data_jax <- bi_data
    bi_data_jax$dyad_ids <- jnp$array(as.integer(as.character(bi_data$dyad_ids)), dtype = jnp$int32)

    if (model_type == "duration") {
      bi_data_jax$event <- jnp$array(as.numeric(as.character(bi_data$event)), dtype = jnp$float32)
      if (!is.null(stan_data$event_count)) {
        bi_data_jax$event_count <- jnp$array(as.integer(as.character(stan_data$event_count)), dtype = jnp$int32)
      } else {
        # Fallback
        bi_data_jax$event_count <- jnp$array(as.integer(as.character(stan_data$event)), dtype = jnp$int32)
      }
    } else {
      bi_data_jax$event <- jnp$array(as.integer(as.character(bi_data$event)), dtype = jnp$int32)
    }
    if (!is.null(bi_data$divisor) && length(bi_data$divisor) > 0) {
      bi_data_jax$divisor <- jnp$array(as.integer(as.character(bi_data$divisor)), dtype = jnp$int32)
    } else {
      # The binomial distribution requires the total_count parameter (divisor) to be >= observed events
      # For standard binary networks where max edge weight is max observed:
      max_event <- max(as.integer(as.character(bi_data$event)), 1L)
      bi_data_jax$divisor <- jnp$full(as.integer(bi_data$num_rows), as.integer(max_event), dtype = jnp$int32)
    }

    if (bi_data$num_fixed > 0) {
      bi_data_jax$design_fixed <- jnp$array(as.matrix(bi_data$design_fixed), dtype = jnp$float32)
    } else {
      bi_data_jax$design_fixed <- NULL
    }

    if (bi_data$num_random > 0) {
      bi_data_jax$design_random <- jnp$array(as.matrix(bi_data$design_random), dtype = jnp$float32)
      bi_data_jax$random_group_index <- jnp$array(as.integer(as.character(bi_data$random_group_index)), dtype = jnp$int32)
    } else {
      bi_data_jax$design_random <- NULL
      bi_data_jax$random_group_index <- NULL
    }

    m$data_on_model <- list(data = bi_data_jax)
    m$fit(bi_func, num_warmup = 1000L, num_samples = 1000L, num_chains = 1L)

    cat("  Extracting draws and plotting...\n")
    bison_edge_samples <- fit_bison$edge_samples
    bi_draws <- get_bi_draws(m$posteriors)

    # Get true values for all dyads
    true_vals <- sim_data$df_true$edge_weight

    brms_draws <- list()
    for (idx in seq_len(ncol(bison_edge_samples))) {
      brms_draws[[paste0("edge_weight[", idx, "]")]] <- as.numeric(bison_edge_samples[, idx])
    }
    brms_names <- paste0("edge_weight[", seq_len(ncol(bison_edge_samples)), "]")
    plot_densities(test_name, bi_draws, brms_draws,
      bi_prefix = "edge_weight",
      brms_prefix = brms_names,
      true_values = true_vals
    )

    cat("  Completed", test_name, "\n")
    # Small sleep to allow filesystem to stabilize and avoid cmdstanr temporary file issues
    Sys.sleep(1)
  }
}


# %%
# Run the combinatoric tests!
run_all_combinations(model_type = "duration", bi_func = bi_model_duration)

cat("\nAll tests completed! Density comparative plots saved in 'Network edges/density_plots'.\n")
