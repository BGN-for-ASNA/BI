library(reticulate)
library(BayesianInference)
library(bisonR)
library(brms)

dir.create("Network edges/density_plots", showWarnings = FALSE)

# Source BI models
source("Network edges/bi_model_binary.R")
source("Network edges/bi_model_count.R")
source("Network edges/bi_model_duration.R")

m <- importBI("cpu")
jnp <- import("jax.numpy")

to_jax <- function(x) {
  if (is.list(x) || typeof(x) == "closure") return(x)
  
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
    idx <- grep(v, cols, fixed=TRUE)
    for (i in idx) {
      # Extract raw draws
      out[[cols[i]]] <- as.numeric(draws[, i])
    }
  }
  return(out)
}

plot_densities <- function(model_name, bi_draws, brms_draws, bi_prefix, brms_prefix, n_plot=3) {
  # We will just plot the first 'n_plot' edge weights for demonstration
  
  png(file.path("Network edges/density_plots", paste0(model_name, "_comparison.png")), width=1000, height=400, res=96)
  par(mfrow=c(1, n_plot))
  
  for (i in 1:n_plot) {
    bi_name <- paste0(bi_prefix, "[", i, "]")
    # brms edge weight random effect pattern: r_dyad_id[i,Intercept] (or similar depending on brms internal naming)
    brms_name <- brms_prefix[i] 
    
    bi_v <- bi_draws[[bi_name]]
    brms_v <- brms_draws[[brms_name]]
    
    if (is.null(bi_v) || is.null(brms_v)) {
      plot(NULL, xlim=c(0,1), ylim=c(0,1), main="Missing Data")
      next
    }
    
    xlim <- range(c(bi_v, brms_v), na.rm=TRUE)
    d_bi <- density(bi_v)
    d_brms <- density(brms_v)
    ylim <- c(0, max(c(d_bi$y, d_brms$y)) * 1.1)
    
    plot(NULL, xlim=xlim, ylim=ylim, main=paste("Edge", i), xlab="Value", ylab="Density")
    polygon(d_brms$x, d_brms$y, col=adjustcolor("#4E9DC4", 0.5), border="#4E9DC4")
    polygon(d_bi$x, d_bi$y, col=adjustcolor("#F5A623", 0.5), border="#F5A623")
    legend("topright", legend=c("bisonR (Stan)", "BI"), fill=c(adjustcolor("#4E9DC4", 0.5), adjustcolor("#F5A623", 0.5)))
  }
  
  dev.off()
}

test_pipeline <- function(model_type, bi_func) {
  cat("\nRunning", model_type, "pipeline...\n")
  
  # Simulate small network
  set.seed(42)
  sim_data <- simulate_bison_model(model_type, aggregated=TRUE)
  df <- sim_data$df_sim
  
  # Fit bisonR model
  # For binary_conjugate, bisonR skips Stan. We must use standard model_type
  cat("  Fitting bisonR model...\n")
  fit_bison <- bison_model(
    (event | duration) ~ dyad(node_1_id, node_2_id),
    data = df,
    model_type = model_type,
    iter_sampling=500, iter_warmup=500
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
    prior_edge_mu = as.numeric(stan_data$prior_edge_mu),
    prior_edge_sigma = as.numeric(stan_data$prior_edge_sigma),
    prior_fixed_mu = as.numeric(stan_data$prior_fixed_mu),
    prior_fixed_sigma = as.numeric(stan_data$prior_fixed_sigma),
    prior_rate_sigma = as.numeric(stan_data$prior_rate_sigma)
  )
  if (length(bi_data$prior_rate_sigma) == 0 || is.na(bi_data$prior_rate_sigma)) bi_data$prior_rate_sigma <- 1.0
  if (length(bi_data$prior_fixed_mu) == 0 || is.na(bi_data$prior_fixed_mu)) bi_data$prior_fixed_mu <- 0.0
  if (length(bi_data$prior_fixed_sigma) == 0 || is.na(bi_data$prior_fixed_sigma)) bi_data$prior_fixed_sigma <- 1.0
  if (length(bi_data$num_fixed) == 0 || is.na(bi_data$num_fixed)) bi_data$num_fixed <- 0L
  if (length(bi_data$prior_edge_mu) == 0 || is.na(bi_data$prior_edge_mu)) bi_data$prior_edge_mu <- 0.0
  if (length(bi_data$prior_edge_sigma) == 0 || is.na(bi_data$prior_edge_sigma)) bi_data$prior_edge_sigma <- 2.0
  
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
  
  # Use exact integer types where required, converting to integer vectors first
  bi_data_jax$dyad_ids <- jnp$array(as.integer(as.character(bi_data$dyad_ids)), dtype=jnp$int32)
  bi_data_jax$event <- jnp$array(as.integer(as.character(bi_data$event)), dtype=jnp$int32)
  if (!is.null(bi_data$event_count)) {
    bi_data_jax$event_count <- jnp$array(as.integer(as.character(bi_data$event_count)), dtype=jnp$int32)
  }
  
  if (!is.null(bi_data$divisor)) {
      bi_data_jax$divisor <- jnp$array(as.integer(as.character(bi_data$divisor)), dtype=jnp$int32)
  }
  
  m$data_on_model <- list(data=bi_data_jax)
  m$fit(bi_func, num_warmup=200L, num_samples=200L, num_chains=1L)
  
  cat("  Extracting draws and plotting...\n")
  
  # bisonR extracts draws into `edge_samples` directly! It's an array of (samples, dyads)
  # We can just use that instead of brms functions to get the exact edge weights!
  # fit_bison$edge_samples is a matrix of (iterations, dyads)
  bison_edge_samples <- fit_bison$edge_samples
  
  bi_draws <- get_bi_draws(m$posteriors)
  
  # Format brms draws to match our plot function:
  brms_draws <- list()
  for (i in 1:ncol(bison_edge_samples)) {
     brms_draws[[paste0("edge_weight[", i, "]")]] <- as.numeric(bison_edge_samples[, i])
  }
  
  brms_names <- paste0("edge_weight[", 1:3, "]")
  
  plot_densities(model_type, bi_draws, brms_draws, 
                 bi_prefix="edge_weight", 
                 brms_prefix=brms_names)
  
  cat("  Completed", model_type, "\n")
}

# Run the tests!
test_pipeline("binary", bi_model_binary)
test_pipeline("count", bi_model_count)
# test_pipeline("duration", bi_model_duration) # bisonR's simulator fails with df_true not found

cat("\nAll tests completed! Density comparative plots saved in 'Network edges/density_plots'.\n")
