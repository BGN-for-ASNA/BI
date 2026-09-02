plot_joint_posterior <- function(fit_path){
  fit <- readRDS(fit_path)
  
  bayesplot::bayesplot_theme_update(
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )
  
  p <- bayesplot::mcmc_pairs(fit$draws(), pars=c("A[1,1]", "A[1,2]", "A[2,1]",  "A[2,2]", "Q[1,1]", "Q[2,1]", "Q[2,2]", "Q[1,2]"), off_diag_fun = "hex")
  
  ggplot2::ggsave("figures/primates_posterior_pairs.png", plot = p, width = 12, height = 12, dpi = 300)
}