SBC_backend_coevolve <- function(...) {
   args <- list(...)
   if(any(names(args) == "data")) {
      stop(paste0("Argument 'data' cannot be provided when defining a backend",
                  " as it needs to be set by the SBC package"))
    }
  
    structure(list(args = args), class = "SBC_backend_coevolve")
}

SBC_fit.SBC_backend_coevolve <- function(backend, generated, cores) { 
  args_with_data <- backend$args
  args_with_data$data <- generated$data
  args_with_data$variables <- generated$variables
  args_with_data$id <- generated$id
  args_with_data$tree <- generated$tree
  args_with_data$scale <- generated$scale
  
  # Use effects_mat if provided (from primed priors)
  # This ensures zeros are preserved in the fitted model
  if (!is.null(generated$effects_mat)) {
    args_with_data$effects_mat <- generated$effects_mat
  }
  
  fit_result <- do.call(coevolve::coev_fit, args_with_data)
  # Add class for SBC method dispatch
  class(fit_result) <- c("SBC_backend_coevolve_fit", class(fit_result))
  return(fit_result)
}

SBC_fit_to_draws_matrix.SBC_backend_coevolve_fit <- function(fit) {
  draws_obj <- fit$fit$draws()
  posterior::as_draws_matrix(draws_obj)
}
