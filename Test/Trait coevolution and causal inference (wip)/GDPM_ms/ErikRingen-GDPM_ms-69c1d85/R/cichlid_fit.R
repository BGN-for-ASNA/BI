cichlid_fit <- function(sim_data, chains = 8, iter_warmup = 200,
                        iter_sampling = 500) {

  fit <- coevolve::coev_fit(
    data = sim_data$d_sim,
    variables = list(
      Promiscuity = "normal",
      SpermSize = "normal",
      Predation = "normal"
    ),
    prior = list(A_offdiag = "normal(0, 2)", Q_sigma = "normal(0, 2)"),
    effects_mat = sim_data$effects_mat,
    scale = FALSE,
    estimate_correlated_drift = FALSE,
    id = "species",
    tree = sim_data$tree,
    seed = 42,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 1,
    adapt_delta = 0.98
  )
  
  fit$fit$save_object("fit_synthetic_cichlid.rds")
  
  return(fit)
}