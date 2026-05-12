library(reticulate)
library(BayesianInference)
library(bisonR)
source("Modified simulate_bison_model.R")
source("bi_model_binary.R")
source("bi_model_count.R")
source("bi_model_duration.R")

# Update setwd to current dir
# setwd("/home/sebastian_sosa/BI/Test/Network/Network edges")

run_debug <- function(model_type, directed, zero_inflated) {
  num_nodes <- 20
  num_locations <- 5
  max_obs <- 5
  
  sim <- simulate_bison_model(model_type, aggregated = TRUE,
    location_effect = TRUE, age_diff_effect = TRUE,
    num_nodes = num_nodes, num_locations = num_locations, max_obs = max_obs)
  
  df <- sim$df_sim
  if (directed) {
    formula <- as.formula("(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")
  } else {
    formula <- as.formula("(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")
  }
  
  priors <- get_default_priors(model_type)
  if (zero_inflated) {
    priors$zero_prob <- "beta(1, 1)"
  }
  
  # Fit with bisonR to get data structure
  fit_bison <- bison_model(formula, data = df, model_type = model_type, priors = priors, chains = 1, iter = 10)
  stan_data <- fit_bison$model_data
  
  # Build BI data
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
    design_fixed = as.matrix(stan_data$design_fixed),
    design_random = as.matrix(stan_data$design_random),
    partial_pooling = 1,
    zero_inflated = as.integer(zero_inflated),
    prior_edge_mu = 0,
    prior_edge_sigma = 1,
    prior_fixed_mu = 0,
    prior_fixed_sigma = 1,
    prior_random_mean_mu = 0,
    prior_random_mean_sigma = 1,
    prior_random_std_sigma = 1,
    prior_zero_prob_alpha = 1,
    prior_zero_prob_beta = 1
  )
  
  if (model_type == "duration") {
    bi_data$event_count = stan_data$event_count
    bi_data$prior_rate_sigma = 1
  }
  
  print("BI Data summary:")
  print(paste("Num rows:", bi_data$num_rows))
  print(paste("Num fixed:", bi_data$num_fixed))
  print(paste("Num random:", bi_data$num_random))
  print(paste("Num random groups:", bi_data$num_random_groups))
  
  # Run BI model once to check for errors
  if (model_type == "binary") bi_func <- bi_model_binary
  if (model_type == "count") bi_func <- bi_model_count
  if (model_type == "duration") bi_func <- bi_model_duration
  
  # Run model
  m <- importBI("cpu")
  # We can't easily "run" it without fit, but we can check the trace
  print("BI model setup successful")
}

run_debug("binary", FALSE, TRUE)
