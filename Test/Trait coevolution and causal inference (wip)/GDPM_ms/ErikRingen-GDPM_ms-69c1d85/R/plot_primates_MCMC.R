plot_primates_MCMC <- function(fit){
  draws <- as_draws_array(fit$draws(variables = c("b", "A", "Q", "alpha", "shape", "lambda_free")))
  
  color_scheme_set("viridisA")
  p_rank <- mcmc_rank_overlay(draws) + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
  
  rhat <- rhat(fit, pars = c("b", "A", "Q", "alpha", "shape", "lambda_free"))
  p_rhat <- bayesplot::mcmc_rhat(rhat)
  
  ggsave("figures/trank_primates.png", p_rank, height = 11, width = 8.5, dpi = 900)
  ggsave("figures/rhat_primates.png", p_rhat, height = 3.5, width = 5, dpi = 900)
  return(p_rank)
}


