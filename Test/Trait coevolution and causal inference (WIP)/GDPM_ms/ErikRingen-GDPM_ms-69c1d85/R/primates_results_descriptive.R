primates_LH_descriptives <- function(primates_GDPM_standata){
  
  d <- as.data.frame(primates_GDPM_standata$y)
  d[d == -99] <- NA
  names(d) <- c("body", "brain", "longevity", "maturity")
  
  d <- d %>% 
    mutate(
      `log(body mass)` = log(body),
      `log(longevity)` = log(longevity),
      `log(maturity)` = log(maturity),
      clade = primates_GDPM_standata$clade
      )
  
  p_body_longevity <- ggplot(d, aes(x = `log(body mass)`, y = `log(longevity)`, color = clade)) + 
    geom_point(size = 3, alpha = 0.6) +
    geom_point(shape = 1, size = 3) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(body mass)`, d$`log(longevity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 15)
  
  p_body_maturity <- ggplot(d, aes(x = `log(body mass)`, y = `log(maturity)`, color = clade)) + 
    geom_point(size = 3, alpha = 0.6) +
    geom_point(shape = 1, size = 3) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(body mass)`, d$`log(maturity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 15)
  
  p_longevity_maturity <- ggplot(d, aes(x = `log(longevity)`, y = `log(maturity)`, color = clade)) + 
    geom_point(size = 3, alpha = 0.6) +
    geom_point(shape = 1, size = 3) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(longevity)`, d$`log(maturity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 15) 
  
  p_comb <- p_body_longevity + p_body_maturity + p_longevity_maturity + plot_layout(guides = 'collect')
  
  return(p_comb)
  ggsave("figures/primate_descriptive_scatter.png", p_comb, width = 8, height = 3.5, dpi = 900)
}
