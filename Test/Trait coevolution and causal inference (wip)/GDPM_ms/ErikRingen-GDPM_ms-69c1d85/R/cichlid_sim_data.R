cichlid_sim_data <- function(model, tree_file){
  tree <- phytools::readNexus(tree_file, format = "raxml")
  
  variables <- c("Promiscuity", "SpermSize", "Predation")
  N <- length(tree$tip.label)
  
  d <- data.frame(
    Promiscuity = rnorm(N),
    SpermSize = rnorm(N),
    Predation = rnorm(N),
    species = tree$tip.label
  )
  
  effects_mat <- matrix(TRUE, nrow = 3, ncol = 3, dimnames = list(variables, variables))
  effects_mat[1,2] = FALSE
  effects_mat[3,2] = FALSE

  dat <- coevolve::coev_make_standata(
    data = d,
    variables = list(
      Promiscuity = "normal",
      SpermSize = "normal",
      Predation = "normal"
    ),
    effects_mat = effects_mat,
    estimate_correlated_drift = FALSE,
    id = "species",
    tree = tree,
    prior_only = TRUE
  )
  
  sim <- model$sample(
    data = dat,
    chains = 1,
    seed = 123,
    refresh = 1,
    iter_warmup = 50,
    iter_sampling = 1
  )
  
  draws <- as_draws_rvars(sim)
  
  Promiscuity <- posterior::draws_of(draws$yrep)[1,1,1:N,1]
  SpermSize <- posterior::draws_of(draws$yrep)[1,1,1:N,2]
  Predation <- posterior::draws_of(draws$yrep)[1,1,1:N,3]
  names(Predation) <- tree$tip.label
  names(SpermSize) <- tree$tip.label
  names(Promiscuity) <- tree$tip.label
  
  d_sim <- data.frame(
    Promiscuity = Promiscuity,
    SpermSize = SpermSize,
    Predation = Predation,
    species = tree$tip.label
  )
  
  draws_sim_long <- as_draws_df(sim$draws(variables = c("A", "Q", "b"))) %>% 
    select(-.chain, -.iteration) %>% 
    pivot_longer(-.draw, names_to = "parameter", values_to = "est") %>% 
    filter(!(parameter %in% c( "Q[1,2]", "Q[2,1]", "Q[3,1]", "Q[3,2]", "Q[2,3]", "Q[1,3]", "Q[3,2]", "A[1,2]", "A[3,2]")))
  
  return(
    list(
      d_sim = d_sim,
      tree = tree,
      effects_mat = effects_mat,
      sim_pars = draws_sim_long
    )
  )
}