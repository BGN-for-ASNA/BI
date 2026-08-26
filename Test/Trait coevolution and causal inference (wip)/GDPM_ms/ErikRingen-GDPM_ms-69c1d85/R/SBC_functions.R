#' Check if a draw has sufficient variance across all traits
#' 
#' @param draw_index Index of the draw to check
#' @param y_rep Array of replicated data [draw, tree, observation, trait]
#' @param variables List of variable types
#' @return TRUE if draw is acceptable, FALSE otherwise
check_draw_variance <- function(draw_index, y_rep, variables) {
  N_traits <- length(variables)
  var_names <- names(variables)
  pass <- TRUE
  
  for (i in 1:N_traits) {
    var_name <- var_names[i]
    trait_type <- variables[[var_name]]
    yrep_values <- y_rep[draw_index, 1, , i]
    
    trait_var <- var(yrep_values, na.rm = TRUE)
    # Check variance based on trait type
    if (trait_type %in% c("bernoulli_logit")) {
      # reject if no variance
      if (length(unique(yrep_values)) == 1) {
        pass <- FALSE
      }    
    }
    else if (trait_type == "ordered_logistic") {
      # reject if not all values are present
      if (length(unique(yrep_values)) != 4) {
        pass <- FALSE
      }
    }
  }
  
  return(pass)
}

forward_sim <- function(sim_config) {
  N <- sim_config$N
  n_sims <- sim_config$n_sims
  traits <- sim_config$traits
  N_traits <- length(traits)
  
  # Check that we have 2 or 3 traits
  if (!(N_traits %in% c(2, 3))) {
    stop("forward_sim currently supports 2 or 3 traits. Got ", N_traits, " traits.")
  }

  tree <- ape::stree(n = N, type = "balanced") |>
    ape::compute.brlen(method = "Grafen")

  d <- data.frame(
    id = tree$tip.label,
    stringsAsFactors = FALSE
  )
  
  # fill in placeholder values for coevolve
  variables <- list()

  for (i in 1:N_traits) {
    trait <- traits[i]
    var_name <- paste0("trait_", i)

    if (trait == "normal") {
      d[[var_name]] <- rnorm(N, 0, 1)
    }
    else if (trait == "poisson_softplus") {
        d[[var_name]] <- rpois(N, 1)
    }
    else if (trait == "ordered_logistic") {
      rep_times <- ceiling(N/4)
      d[[var_name]] <- as.ordered(rep(1:4, each = rep_times, levels = 1:4)[1:N])
    }
    else if (trait == "bernoulli_logit") {
      d[[var_name]] <- rbinom(N, 1, 0.5)
    }
    variables[[var_name]] <- trait
  }
  
  standata <- coevolve::coev_make_standata(
    data = d,
    variables = variables,
    id = "id",
    tree = tree,
    estimate_correlated_drift = FALSE,
    prior_only = TRUE
  )
  
  # Choose the appropriate Stan model based on trait types and number
  if (N_traits == 2) {
    # 2-trait models
    if (all(traits == "normal")) {
      stan_file <- "stan/SBC_gen2traits_double_gaussian.stan"
    } else if (all(traits == "bernoulli_logit")) {
      stan_file <- "stan/SBC_gen2traits_double_bernoulli.stan"
    } else {
      stan_file <- "stan/SBC_gen2traits.stan"
    }
  } else if (N_traits == 3) {
    # 3-trait models
    n_normal <- sum(traits == "normal")
    n_binary <- sum(traits == "bernoulli_logit")
    
    if (n_normal == 2 && n_binary == 1) {
      stan_file <- "stan/SBC_gen3traits_2gaussian_1binary.stan"
    } else if (n_normal == 1 && n_binary == 2) {
      stan_file <- "stan/SBC_gen3traits_2binary_1gaussian.stan"
    } else {
      stop("3-trait models currently only support: 2 normal + 1 binary, or 2 binary + 1 normal. Got: ", 
           paste(traits, collapse = ", "))
    }
  }
  
  # Compile the Stan model
  model <- cmdstanr::cmdstan_model(stan_file, compile = TRUE)
  
  # Run the model with Fixed_param algorithm
  message("Running ", basename(stan_file), " with Fixed_param for ", n_sims, " simulations...")
  stan_fit <- model$sample(
    data = standata,
    chains = 1,
    iter_warmup = 0,
    iter_sampling = n_sims,
    fixed_param = TRUE,
    refresh = 0,
    show_messages = FALSE
  )
  
  # Extract draws
  draws_rvars <- posterior::as_draws_rvars(stan_fit$draws())
  
  # Extract yrep: dimensions should be [draw, tree, observation, trait]
  # Stan generates yrep as array[N_tree, N_obs, J] real yrep
  yrep_draws <- posterior::draws_of(draws_rvars$yrep)  # [draw, tree, observation, trait]
  
  # Verify dimensions
  expected_dims <- c(n_sims, standata$N_tree, standata$N_obs, N_traits)
  if (!all(dim(yrep_draws) == expected_dims)) {
    warning("yrep dimensions don't match expected. Got: ", paste(dim(yrep_draws), collapse = " x "),
            ", Expected: ", paste(expected_dims, collapse = " x "))
  }
  
  y_rep <- yrep_draws
  
  # Extract prior draws: A, Q, b
  # These are generated in the generated quantities block
  prior_draws <- draws_rvars
  
  # Apply variance filtering if needed
  acceptable_indices <- integer(0)
  rejected_indices <- integer(0)
  
  for (draw_idx in 1:n_sims) {
    if (check_draw_variance(draw_idx, y_rep, variables)) {
      acceptable_indices <- c(acceptable_indices, draw_idx)
    } else {
      rejected_indices <- c(rejected_indices, draw_idx)
    }
  }
  
  n_acceptable <- length(acceptable_indices)
  n_rejected <- length(rejected_indices)
  
  if (n_rejected > 0) {
    message("Variance filtering results:")
    message("  - Acceptable draws: ", n_acceptable, " / ", n_sims)
    message("  - Rejected draws: ", n_rejected, " / ", n_sims)
    
    # If we have rejections, subset to acceptable draws
    if (n_acceptable < n_sims) {
      warning("Only ", n_acceptable, " acceptable draws out of ", n_sims, 
              ". Consider increasing n_sims or adjusting priors.")
    }
    
    # Subset to acceptable draws
    y_rep <- y_rep[acceptable_indices, , , , drop = FALSE]
    prior_draws_df <- posterior::as_draws_df(prior_draws)
    prior_draws_filtered_df <- prior_draws_df[acceptable_indices, ]
    prior_draws <- posterior::as_draws_rvars(prior_draws_filtered_df)
  }

  result <- list(
    prior_draws = prior_draws, 
    y_rep = y_rep, 
    tree = tree, 
    variables = variables
  )
  
  return(result)
}

generator <- function(prior_draws, y_rep, tree, variables, draw_index) {
  # Use the provided index directly - SBC will call this n_sims times with indices 1:n_sims
  draw <- draw_index
  
  N_traits <- length(variables)
  var_names <- names(variables)
  
  # Dynamically build variables list for A, b, and Q matrices
  # prior_draws has parameters in dot notation (e.g., "A.1.1.", "A.1.2.", etc.)
  # or bracket notation (e.g., "A[1,1]", "A[1,2]", etc.)
  variables_list <- list()
  
  # Extract parameter names from prior_draws
  param_names <- names(prior_draws)
  
  # Extract parameter values from prior_draws
  # prior_draws has parameters in dot notation (e.g., "A.1.1.", "A.1.2.", etc.)
  # Convert to draws_df for easier access
  prior_draws_df <- posterior::as_draws_df(prior_draws)
  
  # Helper function to find parameter value
  get_param_value <- function(base_name, indices) {
    # Try different naming conventions
    patterns <- c(
      paste0(base_name, "[", paste(indices, collapse = ","), "]"),  # A[1,2]
      paste0(base_name, ".", paste(indices, collapse = "."), "."),  # A.1.2.
      paste0(base_name, paste(indices, collapse = ""))              # A12
    )
    
    for (pattern in patterns) {
      # Find matching column (normalize names for comparison)
      for (col_name in colnames(prior_draws_df)) {
        col_normalized <- gsub("\\[|\\]|\\.", "", col_name)
        pattern_normalized <- gsub("\\[|\\]|\\.", "", pattern)
        if (col_normalized == pattern_normalized) {
          return(as.numeric(prior_draws_df[draw, col_name]))
        }
      }
    }
    return(NA_real_)
  }
  
  # Add A matrix elements (all combinations)
  for (i in 1:N_traits) {
    for (j in 1:N_traits) {
      var_name <- paste0("A[", i, ",", j, "]")
      value <- get_param_value("A", c(i, j))
      if (is.na(value)) {
        # Try accessing as rvar matrix if it exists
        if ("A" %in% param_names) {
          A_rvar <- prior_draws$A
          if (inherits(A_rvar, "rvar")) {
            A_draws <- posterior::draws_of(A_rvar)
            if (length(dim(A_draws)) >= 3) {
              value <- as.numeric(A_draws[draw, i, j])
            }
          }
        }
      }
      variables_list[[var_name]] <- value
    }
  }
  
  # Add b vector elements (include zeros)
  for (i in 1:N_traits) {
    var_name <- paste0("b[", i, "]")
    value <- get_param_value("b", i)
    if (is.na(value) && "b" %in% param_names) {
      b_rvar <- prior_draws$b
      if (inherits(b_rvar, "rvar")) {
        b_draws <- posterior::draws_of(b_rvar)
        if (length(dim(b_draws)) >= 2) {
          value <- as.numeric(b_draws[draw, i])
        }
      }
    }
    variables_list[[var_name]] <- value
  }
  
  # Add Q matrix elements (all combinations, include zeros)
  for (i in 1:N_traits) {
    for (j in 1:N_traits) {
      var_name <- paste0("Q[", i, ",", j, "]")
      value <- get_param_value("Q", c(i, j))
      if (is.na(value) && "Q" %in% param_names) {
        Q_rvar <- prior_draws$Q
        if (inherits(Q_rvar, "rvar")) {
          Q_draws <- posterior::draws_of(Q_rvar)
          if (length(dim(Q_draws)) >= 3) {
            value <- as.numeric(Q_draws[draw, i, j])
          }
        }
      }
      variables_list[[var_name]] <- value
    }
  }
  
  # Dynamically build data.frame columns
  data_list <- list(id = tree$tip.label)
  for (i in 1:N_traits) {
    var_name <- var_names[i]
    trait_type <- variables[[var_name]]

    yrep_values <- y_rep[draw, 1, , i]
    
    if (trait_type == "bernoulli_logit") {
      data_list[[var_name]] <- as.integer(yrep_values)
    } else if (trait_type == "poisson_softplus") {
      data_list[[var_name]] <- as.integer(yrep_values)
    } else if (trait_type == "ordered_logistic") {
      data_list[[var_name]] <- factor(as.ordered(yrep_values), levels = 0:3)
    } else {
      data_list[[var_name]] <- yrep_values
    }
  }
  
  generated_list <- list(
      data = as.data.frame(data_list),
      variables = variables,
      id = "id",
      tree = tree,
      scale = FALSE
    )
  
  return(list(
    variables = variables_list,
    generated = generated_list
  ))
}