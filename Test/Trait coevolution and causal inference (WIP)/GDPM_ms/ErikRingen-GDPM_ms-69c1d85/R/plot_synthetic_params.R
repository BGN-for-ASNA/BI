plot_synthetic_params <- function(fit, synthetic_sim){

  draws_fit <- fit$fit$draws(variables = c("A", "Q", "b"))
  
  post_long <- as_draws_df(draws_fit) %>% 
    select(-.chain, -.iteration) %>% 
    pivot_longer(-.draw, names_to = "parameter", values_to = "est") %>% 
    filter(!(parameter %in% c( "Q[1,2]", "Q[2,1]", "Q[3,1]", "Q[3,2]", "Q[2,3]", "Q[1,3]", "Q[3,2]", "A[1,2]", "A[3,2]")))
  
  draws_sim_long <- synthetic_sim$sim_pars
  
  post_summary <- post_long %>% 
    group_by(parameter) %>% 
    summarize(
      mean = mean(est),
      # Calculate quantiles for different interval widths
      lower_95 = quantile(est, 0.025),
      upper_95 = quantile(est, 0.975),
      lower_50 = quantile(est, 0.25),
      upper_50 = quantile(est, 0.75)
    )
  
  # Convert parameter to factor with proper ordering
  post_summary <- post_summary %>%
    mutate(parameter = factor(parameter, levels = parameter))
  
  # Update draws_sim_long to match factor levels
  draws_sim_long <- draws_sim_long %>%
    mutate(parameter = factor(parameter, levels = levels(post_summary$parameter)))
  
  # Reverse factor levels manually to avoid potential issues with fct_rev
  post_summary <- post_summary %>%
    mutate(parameter = factor(parameter, levels = rev(levels(parameter))))
  draws_sim_long <- draws_sim_long %>%
    mutate(parameter = factor(parameter, levels = levels(post_summary$parameter)))
  
  # Create explicit label mapping to prevent automatic parsing
  # Use identity mapping to avoid any label transformation
  param_levels <- levels(post_summary$parameter)
  label_map <- setNames(as.character(param_levels), param_levels)
  
  p <- ggplot(post_summary, aes(x = mean, y = parameter)) + 
    geom_linerange(aes(xmin = lower_95, xmax = upper_95),
                   size = 1.5, color = "darkgrey") +
    geom_linerange(aes(xmin = lower_50, xmax = upper_50),
                   size = 3, color = "darkgrey") +
    geom_point(size = 2.5, aes(color = "Posterior mean")) + 
    geom_vline(xintercept = 0, linetype = "dashed") + 
    geom_point(data = draws_sim_long, aes(x = est, y = parameter, color = "True value"),  size = 2.2, shape = 17) +
    scale_y_discrete(labels = label_map) +
    scale_color_manual(name = "",  # Remove legend title
                       values = c("Posterior mean" = "black", 
                                  "True value" = "indianred"),
                       breaks = c("Posterior mean", "True value")) +
    theme_classic(base_size = 14) + 
    ylab("") + 
    xlab("Parameter value")
  
  ggsave("figures/synthetic_pars.png", plot = p, width = 6, height = 4, dpi = 900)
  
  return(p)
}
