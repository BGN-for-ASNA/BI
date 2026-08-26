plot_synthetic_intervention <- function(synthetic_fit){
  post <- extract_samples(synthetic_fit)
  
  eta_tip_mean <- post$eta[,,1:synthetic_fit$stan_data$N_tips,] |> apply(3, mean)
  eta_tip_sd <- post$eta[,,1:synthetic_fit$stan_data$N_tips,] |> apply(3, sd)
  
  p1 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1] + 2*eta_tip_sd[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3]), intervention_values = list(Promiscuity = eta_tip_mean[1] + 2*eta_tip_sd[1], SpermSize = NA, Predation = NA), prob = 0.9) + scale_y_continuous(limits = c(-13, 13)) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Promiscuity +2 SD")
  
  p2 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1] + -2*eta_tip_sd[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3]), intervention_values = list(Promiscuity = eta_tip_mean[1] + -2*eta_tip_sd[1], SpermSize = NA, Predation = NA), prob = 0.9) + scale_y_continuous(limits = c(-13, 13)) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Promiscuity -2 SD", title = "")
  
  p3 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] + 2*eta_tip_sd[3]), intervention_values = list(Promiscuity = NA, SpermSize = NA, Predation = eta_tip_mean[3] + 2*eta_tip_sd[3]), prob = 0.9) + scale_y_continuous(limits = c(-13, 13)) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Predation +2 SD", title = "")
  
  p4 <- coev_plot_pred_series(synthetic_fit, stochastic = F, eta_anc = list(Promiscuity = eta_tip_mean[1], SpermSize = eta_tip_mean[2], Predation = eta_tip_mean[3] + -2*eta_tip_sd[3]), intervention_values = list(Promiscuity = NA, SpermSize = NA, Predation = eta_tip_mean[3] + -2*eta_tip_sd[3]), prob = 0.9) + scale_y_continuous(limits = c(-13, 13)) + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + labs(subtitle = "Predation -2 SD", title = "")
  
  p_intervention <- wrap_plots(p1, p2, p3, p4, ncol = 2, nrow = 2, byrow = TRUE) + plot_layout(guides = 'collect', axes = 'collect')

  ggsave("figures/synthetic_intervention.png", plot = p_intervention, dpi = 900, width = 8, height = 6.5)
  
  return(p_comb)
}