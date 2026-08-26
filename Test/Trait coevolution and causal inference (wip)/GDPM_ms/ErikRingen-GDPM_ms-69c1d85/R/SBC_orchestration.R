source("R/SBC_functions.R")
source("R/SBC_coevolve_backend.R")

run_SBC <- function(forward_sim_outputs, iter_warmup = 100,
                    iter_sampling = 200, chains = 1) {
  
  # Report rejection sampling statistics from forward_sim
  if (!is.null(forward_sim_outputs$rejection_stats)) {
    stats <- forward_sim_outputs$rejection_stats
    message("\nRejection sampling summary:")
    message("  - Total generated: ", stats$total_generated)
    message("  - Acceptable: ", stats$n_acceptable)
    message("  - Rejected: ", stats$n_rejected)
  }

  n_sims <- dim(forward_sim_outputs$y_rep)[1]
  
  counter_env <- new.env()
  counter_env$current_index <- 0L
  
  generator_with_counter <- function(...) {
    counter_env$current_index <- counter_env$current_index + 1L
    generator(
      prior_draws = forward_sim_outputs$prior_draws,
      y_rep = forward_sim_outputs$y_rep,
      tree = forward_sim_outputs$tree,
      variables = forward_sim_outputs$variables,
      draw_index = counter_env$current_index
    )
  }

  # Create SBC generator object
  sbc_generator <- SBC::SBC_generator_function(generator_with_counter)

  datasets <- SBC::generate_datasets(sbc_generator, n_sims = n_sims)

  # Set up SBC backend with priors matching Stan generative models
  backend_coevolve <- SBC_backend_coevolve(
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    chains = chains,
    estimate_correlated_drift = FALSE,
    scale = FALSE,
    prior = list(Q_sigma = "normal(2, 1)", A_offdiag = "normal(0, 2.5)", b = "normal(0, 1)", A_diag = "normal(-1.0, 0.5)")
  )

  # 4. Run SBC
  message("Starting compute_SBC with ", length(datasets), " datasets...")
  
  n_datasets <- length(datasets)
  n_cores <- parallel::detectCores()
  n_workers <- ceiling(n_cores)
  chunk_size <- min(n_workers, ceiling(n_datasets / n_workers))
  
  results <- SBC::compute_SBC(
    datasets = datasets,
    backend = backend_coevolve, 
    globals = c("SBC_fit.SBC_backend_coevolve", 
                "SBC_fit_to_draws_matrix.SBC_backend_coevolve_fit"),
    keep_fits = FALSE,
    chunk_size = chunk_size
  )

  return(results)
}
