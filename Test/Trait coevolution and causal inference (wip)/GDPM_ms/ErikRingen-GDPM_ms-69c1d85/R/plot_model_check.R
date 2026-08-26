plot_model_check <- function(primates_GDPM_fit_rds, primates_GDPM_standata){
  
  post <- extract_samples2(primates_GDPM_fit_rds)
  
  # exclude Tarsiidae as they comprise only 2 species in sample, with low-quality grouped plots
  tarsiidae <- primates_GDPM_standata$clade == "Tarsiidae"
  
  y <- as.data.frame(primates_GDPM_standata$y[!tarsiidae,])
  y[y == -99] <- NA
  
  yrep <- post$yrep[,1,!tarsiidae,]
  yrep_avg <- apply(yrep, 2:3, median)
  
  clade <- primates_GDPM_standata$clade[!tarsiidae]
  
  bayesplot::color_scheme_set("orange")
  
  p_dens_body <- ppc_ecdf_overlay_grouped(y = y$body, yrep = yrep[1:100,,1], group = clade) + facet_wrap(~group,
    scales = "free_x",
    labeller = label_parsed
  )  + theme_classic(base_size = 12) + theme( axis.ticks.x = element_blank(), axis.text.x = element_blank(), strip.background = element_blank()) + 
    labs(x = "Body size", y = "ECDF")
  
  p_dens_longevity <- ppc_ecdf_overlay_grouped(y = y$longevity[!is.na(y$longevity)], yrep = yrep[1:100,!is.na(y$longevity),3], group = clade[!is.na(y$longevity)]) + facet_wrap(~group,scales = "free_x")  + theme_classic(base_size = 12) + theme( axis.ticks.x = element_blank(), axis.text.x = element_blank(), strip.background = element_blank()) + 
    labs(x = "Longevity", y = "ECDF")

  p_dens_maturity <- ppc_ecdf_overlay_grouped(y = y$maturity[!is.na(y$maturity)], yrep = yrep[1:100,!is.na(y$maturity),4], group = clade[!is.na(y$maturity)]) + facet_wrap(~group,scales = "free_x")  + theme_classic(base_size = 12) + theme( axis.ticks.x = element_blank(), axis.text.x = element_blank(), strip.background = element_blank()) + 
    labs(x = "Age at Female Sexual Maturity", y = "ECDF")

  bayesplot::color_scheme_set("blue")
  
  p_dens_brain <- ppc_ecdf_overlay_grouped(y = y$brain, yrep = yrep[1:100,,2], group = clade) + facet_wrap(~group,scales = "free_x")  + theme_classic(base_size = 12) + theme( axis.ticks.x = element_blank(), axis.text.x = element_blank(), strip.background = element_blank(), legend.position = 'none') + 
    labs(x = "Brain size", y = "ECDF")
  
  p_comb <- patchwork::wrap_plots(p_dens_body, p_dens_longevity, p_dens_maturity, p_dens_brain) + plot_layout(guides = 'collect', axis_titles  = "collect") + patchwork::plot_annotation(title = "Posterior predictive checks")
  
  ggsave("figures/primate_model_checks.png", plot = p_comb, dpi = 900, width = 13.5, height = 8.5)
  return(p_comb)
}