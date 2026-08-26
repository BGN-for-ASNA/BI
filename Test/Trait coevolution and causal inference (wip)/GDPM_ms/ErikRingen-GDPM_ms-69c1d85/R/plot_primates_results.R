plot_primates_results <- function(fit, primates_GDPM_standata){
  post <- extract_samples2(fit)
  A <- post$A
  
  N_tips <- dim(post$yrep)[3]
  eta_tips <- post$eta[,1,1:N_tips,]
  eta_mean <- apply(eta_tips, c(1,3), mean)
  eta_sd <- apply(eta_tips, c(1,3), sd)
  
  LH_mean <- apply((eta_tips[,,1] - eta_mean[,1])/eta_sd[1], 2, mean)
  b_mean <- apply(log(1 + exp(eta_tips[,,2])), 2, mean)
  
  var_names <- c("Slow Life History", "Brain-Body Slope")
  colors <- c("orange", "cornflowerblue")
  
  # Function to add quotes around subscripts
  add_quotes_to_subscripts <- function(label) {
    gsub("\\[(.*?)\\]", "['\\1']", label)
  }
  
  p_list <- list()
  
  df_A <- data.frame(
    to = c(),
    from = c(),
    est = c(),
    param_name = c()
  )
  ticker <- 1
  
  for (i in 1:dim(A)[2])
    for (j in 1:dim(A)[2]) {
      
      # standardize A
      A_temp <- A[,i,j] * (eta_sd[,j]/eta_sd[,i])
      
      df_A <- bind_rows(df_A, data.frame(
        to = var_names[i],
        from = var_names[j],
        est = A_temp,
        param_name = paste0("A[", i, ",", j, "]", "~(std)")
      ))
      
      p_temp <- ggplot(data.frame(A = A_temp), aes(x = A)) + 
      stat_halfeye(.width = c(0.5, 0.95), fill = colors[j], alpha = 0.7) +
      labs(y = var_names[j], subtitle= var_names[i],x = parse(text = add_quotes_to_subscripts(paste0("A[", i, ",", j, "]", "~(std)")))) +
      theme_classic(base_size = 12) + 
      theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), plot.subtitle = element_text(size = 12, hjust = 0.5))
    
    if (i != j) p_temp <- p_temp + geom_vline(xintercept = 0, linetype = "dashed")
    p_list[[ticker]] <- p_temp
    ticker <- ticker + 1
  }
  
  labels = function(x) parse(text = add_quotes_to_subscripts(x))
  
  p_comb <- wrap_elements(grid::textGrob('From this variable..', rot = 90, gp = gpar(fontsize = 14))) + (wrap_plots(p_list, byrow = T)) +
    plot_annotation(title = "...to this variable", theme = theme(plot.title = element_text(hjust = 0.5, size = 14))) +
    plot_layout(widths = c(0.03, 1))
  
  primate_results <- bind_rows(df_A, data.frame(est = post$cor_R[,1,2], param_name = "cor_drift[1,2]"))
  
  ggsave("figures/primate_results.png", p_comb, width = 6, height = 6, dpi = 900)
  write_csv(primate_results, "out/primate_results.csv")
  return(p_comb)
  
  d_mean <- data.frame(
    b_mean = b_mean,
    LH_mean = LH_mean,
    clade = primates_GDPM_standata$clade
  )
  
  p_LH <- ggplot(d_mean, aes(x = LH_mean, y = b_mean, color = clade, fill = clade)) + 
    geom_point(size = 3, alpha = 0.5) + 
    geom_point(shape = 1, size = 3) + 
    scale_color_viridis_d(option = "turbo") +
    xlab("Slow Life History (z-score)") +
    ylab("Brain-body slope") +
    theme_classic(base_size = 16)
  
  ggsave("figures/primate_post_scatter.png", p_LH, width = 6, height = 6, dpi = 900)
}
