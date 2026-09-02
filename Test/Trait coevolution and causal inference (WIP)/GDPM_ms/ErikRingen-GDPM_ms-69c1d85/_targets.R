## Quick mode: set GDPM_QUICK=TRUE to run a fast smoke test with fewer
## simulations and shorter MCMC chains. Useful for verifying the pipeline
## runs end-to-end without committing to a multi-hour full run.
quick_mode <- as.logical(Sys.getenv("GDPM_QUICK", "FALSE"))

if (quick_mode) {
  message("*** GDPM_QUICK mode enabled — reduced simulations and iterations ***")
  n_sims_config     <- 10
  sbc_warmup        <- 25
  sbc_sampling      <- 50
  sbc_chains        <- 1
  synth_warmup      <- 50
  synth_sampling    <- 100
  synth_chains      <- 2
  primate_warmup    <- 50
  primate_sampling  <- 100
  primate_chains    <- 2
  n_workers         <- min(4, parallel::detectCores())
} else {
  n_sims_config     <- 500
  sbc_warmup        <- 100
  sbc_sampling      <- 200
  sbc_chains        <- 1
  synth_warmup      <- 200
  synth_sampling    <- 500
  synth_chains      <- 8
  primate_warmup    <- 250
  primate_sampling  <- 1500
  primate_chains    <- 8
  n_workers         <- 12
}

future::plan(future::multisession, workers = n_workers)
targets::tar_source()
library(tarchetypes)

# pipeline
list(
  ### Tree traversal figure
  tar_target(tree_traversal_plot, plot_tree_traversal(seed=456)),
  ### Synthetic example: cichlids
  tar_target(cichlid_tree_file, "data/05_BEAST_RAxML.tre", format = "file"),
  tar_target(synthetic_model_file, "stan/synthetic_model.stan", format = "file"),
  tar_target(synthetic_model, cmdstanr::cmdstan_model(synthetic_model_file)),
  tar_target(synthetic_sim, cichlid_sim_data(synthetic_model, cichlid_tree_file)),
  tar_target(synthetic_fit, cichlid_fit(synthetic_sim,
    chains = synth_chains, iter_warmup = synth_warmup,
    iter_sampling = synth_sampling)),
  tar_target(synthetic_summary, capture.output(summary(synthetic_fit))),
  # Plot cichlids results
  tar_target(synthetic_param_plot, plot_synthetic_params(synthetic_fit, synthetic_sim)),

  ### Simulation-based calibration/power analysis
  # Sim data: N=64, 1 Gaussian + 1 Bernoulli
  tar_target(sim_data_64_2traits, forward_sim(
    sim_config = list(N = 64, n_sims = n_sims_config, traits = c("normal", "bernoulli_logit"))
  )),
  tar_target(sbc_64_2traits, run_SBC(forward_sim_outputs = sim_data_64_2traits, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),

  # Sim data: N=128, 1 Gaussian + 1 Bernoulli
  tar_target(sim_data_128_2traits, forward_sim(
    sim_config = list(N = 128, n_sims = n_sims_config, traits = c("normal", "bernoulli_logit"))
  )),
  tar_target(sbc_128_2traits, run_SBC(forward_sim_outputs = sim_data_128_2traits, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),

  # Sim data: N=256, 1 Gaussian + 1 Bernoulli
  tar_target(sim_data_256_2traits, forward_sim(
    sim_config = list(N = 256, n_sims = n_sims_config, traits = c("normal", "bernoulli_logit"))
  )),
  tar_target(sbc_256_2traits, run_SBC(forward_sim_outputs = sim_data_256_2traits, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  # Double Gaussian variants (2 traits, both normal)
  tar_target(sim_data_64_2traits_double_gaussian, forward_sim(
    sim_config = list(N = 64, n_sims = n_sims_config, traits = c("normal", "normal"))
  )),
  tar_target(sbc_64_2traits_double_gaussian, run_SBC(forward_sim_outputs = sim_data_64_2traits_double_gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_128_2traits_double_gaussian, forward_sim(
    sim_config = list(N = 128, n_sims = n_sims_config, traits = c("normal", "normal"))
  )),
  tar_target(sbc_128_2traits_double_gaussian, run_SBC(forward_sim_outputs = sim_data_128_2traits_double_gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_256_2traits_double_gaussian, forward_sim(
    sim_config = list(N = 256, n_sims = n_sims_config, traits = c("normal", "normal"))
  )),
  tar_target(sbc_256_2traits_double_gaussian, run_SBC(forward_sim_outputs = sim_data_256_2traits_double_gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  # Double Bernoulli-logit variants (2 traits, both bernoulli_logit)
  tar_target(sim_data_64_2traits_double_bernoulli, forward_sim(
    sim_config = list(N = 64, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit"))
  )),
  tar_target(sbc_64_2traits_double_bernoulli, run_SBC(forward_sim_outputs = sim_data_64_2traits_double_bernoulli, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_128_2traits_double_bernoulli, forward_sim(
    sim_config = list(N = 128, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit"))
  )),
  tar_target(sbc_128_2traits_double_bernoulli, run_SBC(forward_sim_outputs = sim_data_128_2traits_double_bernoulli, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_256_2traits_double_bernoulli, forward_sim(
    sim_config = list(N = 256, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit"))
  )),
  tar_target(sbc_256_2traits_double_bernoulli, run_SBC(forward_sim_outputs = sim_data_256_2traits_double_bernoulli, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  # 3-trait models: 2 Gaussian + 1 Binary
  tar_target(sim_data_64_3traits_2gaussian_1binary, forward_sim(
    sim_config = list(N = 64, n_sims = n_sims_config, traits = c("normal", "normal", "bernoulli_logit"))
  )),
  tar_target(sbc_64_3traits_2gaussian_1binary, run_SBC(forward_sim_outputs = sim_data_64_3traits_2gaussian_1binary, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_128_3traits_2gaussian_1binary, forward_sim(
    sim_config = list(N = 128, n_sims = n_sims_config, traits = c("normal", "normal", "bernoulli_logit"))
  )),
  tar_target(sbc_128_3traits_2gaussian_1binary, run_SBC(forward_sim_outputs = sim_data_128_3traits_2gaussian_1binary, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_256_3traits_2gaussian_1binary, forward_sim(
    sim_config = list(N = 256, n_sims = n_sims_config, traits = c("normal", "normal", "bernoulli_logit"))
  )),
  tar_target(sbc_256_3traits_2gaussian_1binary, run_SBC(forward_sim_outputs = sim_data_256_3traits_2gaussian_1binary, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  # 3-trait models: 2 Binary + 1 Gaussian
  tar_target(sim_data_64_3traits_2binary_1gaussian, forward_sim(
    sim_config = list(N = 64, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit", "normal"))
  )),
  tar_target(sbc_64_3traits_2binary_1gaussian, run_SBC(forward_sim_outputs = sim_data_64_3traits_2binary_1gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_128_3traits_2binary_1gaussian, forward_sim(
    sim_config = list(N = 128, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit", "normal"))
  )),
  tar_target(sbc_128_3traits_2binary_1gaussian, run_SBC(forward_sim_outputs = sim_data_128_3traits_2binary_1gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),
  
  tar_target(sim_data_256_3traits_2binary_1gaussian, forward_sim(
    sim_config = list(N = 256, n_sims = n_sims_config, traits = c("bernoulli_logit", "bernoulli_logit", "normal"))
  )),
  tar_target(sbc_256_3traits_2binary_1gaussian, run_SBC(forward_sim_outputs = sim_data_256_3traits_2binary_1gaussian, iter_warmup = sbc_warmup, iter_sampling = sbc_sampling, chains = sbc_chains)),

  ## SBC results
  tar_target(ECDF_diff_plot, sbc_combined_plot(
    sbc_64_2traits,
    sbc_128_2traits,
    sbc_256_2traits,
    sbc_64_2traits_double_gaussian,
    sbc_128_2traits_double_gaussian,
    sbc_256_2traits_double_gaussian,
    sbc_64_2traits_double_bernoulli,
    sbc_128_2traits_double_bernoulli,
    sbc_256_2traits_double_bernoulli,
    sbc_64_3traits_2gaussian_1binary,
    sbc_128_3traits_2gaussian_1binary,
    sbc_256_3traits_2gaussian_1binary,
    sbc_64_3traits_2binary_1gaussian,
    sbc_128_3traits_2binary_1gaussian,
    sbc_256_3traits_2binary_1gaussian
  )),
  tar_target(SBC_recovery_plot, actual_vs_fit_plot(
    sbc_64_2traits,
    sbc_128_2traits,
    sbc_256_2traits,
    sbc_64_2traits_double_gaussian,
    sbc_128_2traits_double_gaussian,
    sbc_256_2traits_double_gaussian,
    sbc_64_2traits_double_bernoulli,
    sbc_128_2traits_double_bernoulli,
    sbc_256_2traits_double_bernoulli,
    sbc_64_3traits_2gaussian_1binary,
    sbc_128_3traits_2gaussian_1binary,
    sbc_256_3traits_2gaussian_1binary,
    sbc_64_3traits_2binary_1gaussian,
    sbc_128_3traits_2binary_1gaussian,
    sbc_256_3traits_2binary_1gaussian
  )),

  ### Primate GDPM
  tar_target(primates_tree_file, "data/primate_consensus_tree.tre", format = "file"),
  tar_target(primates_data_file, "data/primates_data.csv", format = "file"),
  tar_target(primates_GDPM_model_file, "stan/GDPM_primate.stan", format = "file"),
  tar_target(primates_GDPM_model, cmdstanr::cmdstan_model(primates_GDPM_model_file)),
  tar_target(primates_GDPM_standata, prepare_primates_GDPM_data(primates_data_file, primates_tree_file)),
  # descriptive plot (individual - kept for backwards compatibility)
  tar_target(primates_LH_descip_plot, primates_LH_descriptives(primates_GDPM_standata)),
  
  tar_target(primates_GDPM_fit, {primates_GDPM_model$sample(
    data = primates_GDPM_standata |> within({clade = -99}),
    iter_warmup = primate_warmup,
    iter = primate_sampling,
    chains = primate_chains,
    parallel_chains = primate_chains,
    adapt_delta = 0.97,
    seed = 123,
    refresh = 0,
  )$save_object(file = "fit/primate_GDPM.RDS")
    return("fit/primate_GDPM.RDS")}, format = "file"),
  tar_target(primates_GDPM_fit_rds, readRDS(primates_GDPM_fit), format = "rds"),
  # MCMC diagnostics
  tar_target(primates_GDPM_diagnostics, plot_primates_MCMC(primates_GDPM_fit_rds)),
  # Primates GPDM results
  tar_target(primates_results, plot_primates_results(primates_GDPM_fit_rds, primates_GDPM_standata)),
  tar_target(primates_combined_plot, plot_primates_combined(primates_GDPM_fit_rds, primates_GDPM_standata)),
  tar_target(primates_model_check, plot_model_check(primates_GDPM_fit_rds, primates_GDPM_standata)),
  tar_target(primates_posterior_pairs, plot_joint_posterior(primates_GDPM_fit)),
  
  ### Manuscript
  # Depends on upstream targets that produce files read by manuscript.qmd
  tar_target(manuscript, {
    force(primates_combined_plot)
    force(primates_results)
    force(ECDF_diff_plot)
    force(SBC_recovery_plot)
    force(synthetic_param_plot)
    quarto::quarto_render("manuscript.qmd", output_format = "docx")
    "manuscript.docx"
  }, format = "file"),

  ### Session info
  tar_target(
    session_info,
    writeLines(capture.output(sessionInfo()), "session_info.txt")
  )
)
