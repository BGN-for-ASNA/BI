plot_primates_combined <- function(fit, primates_GDPM_standata) {
  
  # ---- Panel A: Descriptive scatter plots ----
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
    geom_point(size = 2.5, alpha = 0.6) +
    geom_point(shape = 1, size = 2.5) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(body mass)`, d$`log(longevity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 11) +
    theme(legend.position = "none")
  
  p_body_maturity <- ggplot(d, aes(x = `log(body mass)`, y = `log(maturity)`, color = clade)) + 
    geom_point(size = 2.5, alpha = 0.6) +
    geom_point(shape = 1, size = 2.5) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(body mass)`, d$`log(maturity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 11) +
    theme(legend.position = "none")
  
  p_longevity_maturity <- ggplot(d, aes(x = `log(longevity)`, y = `log(maturity)`, color = clade)) + 
    geom_point(size = 2.5, alpha = 0.6) +
    geom_point(shape = 1, size = 2.5) + 
    scale_color_viridis_d(option = "turbo") +
    labs(subtitle = paste("cor =", round(cor(d$`log(longevity)`, d$`log(maturity)`, use = "pairwise.complete.obs"), 2))) +
    theme_classic(base_size = 11)
  
  panel_A <- p_body_longevity + p_body_maturity + p_longevity_maturity + 
    plot_layout(ncol = 3, guides = 'collect') &
    theme(legend.position = "bottom", legend.title = element_blank())
  
  # ---- Panel B: A matrix effects ----
  post <- extract_samples2(fit)
  A <- post$A
  
  N_tips <- dim(post$yrep)[3]
  eta_tips <- post$eta[,1,1:N_tips,]
  eta_mean <- apply(eta_tips, c(1,3), mean)
  eta_sd <- apply(eta_tips, c(1,3), sd)
  
  LH_mean <- apply((eta_tips[,,1] - eta_mean[,1])/eta_sd[1], 2, mean)
  b_mean <- apply(log(1 + exp(eta_tips[,,2])), 2, mean)
  
  var_names <- c("Life-History Pace", "Brain-Body Slope")
  colors <- c("orange", "cornflowerblue")
  
  add_quotes_to_subscripts <- function(label) {
    gsub("\\[(.*?)\\]", "['\\1']", label)
  }
  
  p_list <- list()
  df_A <- data.frame(to = c(), from = c(), est = c(), param_name = c())
  ticker <- 1
  
  for (i in 1:dim(A)[2])
    for (j in 1:dim(A)[2]) {
      A_temp <- A[,i,j] * (eta_sd[,j]/eta_sd[,i])
      
      df_A <- bind_rows(df_A, data.frame(
        to = var_names[i],
        from = var_names[j],
        est = A_temp,
        param_name = paste0("A[", i, ",", j, "]", "~(std)")
      ))
      
      p_temp <- ggplot(data.frame(A = A_temp), aes(x = A)) + 
        stat_halfeye(.width = c(0.5, 0.95), fill = colors[j], alpha = 0.7) +
        labs(y = var_names[j], subtitle = var_names[i], 
             x = parse(text = add_quotes_to_subscripts(paste0("A[", i, ",", j, "]", "~(std)")))) +
        theme_classic(base_size = 10) + 
        theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), 
              plot.subtitle = element_text(size = 10, hjust = 0.5))
      
      if (i != j) p_temp <- p_temp + geom_vline(xintercept = 0, linetype = "dashed")
      p_list[[ticker]] <- p_temp
      ticker <- ticker + 1
    }
  
  panel_B <- wrap_elements(grid::textGrob('From this variable..', rot = 90, gp = grid::gpar(fontsize = 11))) + 
    (wrap_plots(p_list, byrow = TRUE)) +
    plot_annotation(title = "...to this variable", theme = theme(plot.title = element_text(hjust = 0.5, size = 11))) +
    plot_layout(widths = c(0.05, 1))
  
  # ---- Panel C: Posterior scatter ----
  d_mean <- data.frame(
    b_mean = b_mean,
    LH_mean = LH_mean,
    clade = primates_GDPM_standata$clade
  )
  
  panel_C <- ggplot(d_mean, aes(x = LH_mean, y = b_mean, color = clade, fill = clade)) + 
    geom_point(size = 3, alpha = 0.5) + 
    geom_point(shape = 1, size = 3) + 
    scale_color_viridis_d(option = "turbo") +
    xlab("Life-History Pace (z-score)") +
    ylab("Brain-body slope") +
    theme_classic(base_size = 12) +
    theme(legend.position = "none")
  
  # ---- Combine all panels ----
  # Wrap panels so each is treated as a single unit for tagging
  combined <- (wrap_elements(panel_A)) / 
    (wrap_elements(panel_B) | panel_C) +
    plot_layout(heights = c(1, 1.2)) +
    plot_annotation(tag_levels = 'A')
  
  ggsave("figures/primate_combined.png", combined, width = 10, height = 8, dpi = 900)
  
  # Also save results CSV
  primate_results <- bind_rows(df_A, data.frame(est = post$cor_R[,1,2], param_name = "cor_drift[1,2]"))
  write_csv(primate_results, "out/primate_results.csv")
  
  return(combined)
}

