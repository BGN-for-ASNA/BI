plot_synthetic_dynamics <- function(synthetic_fit){
  post <- extract_samples(synthetic_fit)
  
  eta_tip_mean <- post$eta[,,1:synthetic_fit$stan_data$N_tips,] %>% apply(3, mean)
  eta_tip_sd <- post$eta[,,1:synthetic_fit$stan_data$N_tips,] %>% apply(3, sd)
  
  p1 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1] + 2*eta_tip_sd[2], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7)) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Promiscuity +2 SD")
  
  p2 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1] + -2*eta_tip_sd[2], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7))  + labs(title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Promiscuity -2 SD")
  
  p3 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2] + 2*eta_tip_sd[2], Predation = eta_tip_mean[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7))  + labs(title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "SpermSize +2 SD")
  
  p4 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2] + -2*eta_tip_sd[2], Predation = eta_tip_mean[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7))  + labs(title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "SpermSize -2 SD")
  
  p5 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] + 2*eta_tip_sd[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7))  + labs(title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Predation +2 SD")
  
  
  p6 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] + -2*eta_tip_sd[3] ), prob = 0.9) + scale_y_continuous(limits = c(-7, 7))  + labs(title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Predation -2 SD")
  
  p_expected <- wrap_plots(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3, byrow = TRUE) + plot_layout(guides = 'collect', axes = 'collect')
  p_comb <- (p_stochastic + p_expected) + plot_annotation(tag_levels = list(c("a", "b")))

  ggsave("figures/synthetic_dynamics.png", plot = p_comb, dpi = 900, width = 13.5, height = 8.5)
  
  return(p_comb)
}