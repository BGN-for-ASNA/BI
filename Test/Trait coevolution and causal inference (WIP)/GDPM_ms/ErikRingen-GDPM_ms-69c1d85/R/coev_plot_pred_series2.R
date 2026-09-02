# simplified version of the coevolve function, useable with model for primate model not fit by the package
coev_plot_pred_series2 <- function(preds, fit, standata, prob = 0.95, standardize = T){
  post <- extract_samples2(fit)
  
  eta_tip_mean <- post$eta[,,1:standata$N_tips,] |> apply(3, mean)
  eta_tip_sd <- post$eta[,,1:standata$N_tips,] |> apply(3, sd)
  
  if (standardize == T) {
    for (j in 1:dim(preds)[3]) preds[,,j] <- (preds[,,j] - eta_tip_mean[j])/eta_tip_sd[j]
    ylabel <- "Trait Value z-score (Latent Scale)"
  } else {
    ylabel <- "Trait Value (Latent Scale)"
  }
  
  
  probs <- c((1 - prob)/2, 1 - (1 - prob)/2)
  
  preds_long <- dplyr::mutate(tidyr::pivot_longer(as.data.frame.table(preds, responseName = "est"), cols = -c(samps, time, response), names_to = NULL, values_to = "est"), samps = as.numeric(as.character(samps)), time = as.numeric(as.character(time)))
  
  epreds_summary <- dplyr::summarise(dplyr::group_by(preds_long, .data$response, .data$time), mean = mean(.data$est, na.rm = TRUE), lower_CI = stats::quantile(.data$est, probs[1], na.rm = TRUE), upper_CI = stats::quantile(.data$est, probs[2], na.rm = TRUE), .groups = "drop")
  
  p <- ggplot2::ggplot(epreds_summary, ggplot2::aes(x = .data$time, 
                                                    y = .data$mean, color = .data$response, fill = .data$response, linetype = .data$response)) + ggplot2::geom_line(size = 1) + 
    ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower_CI, 
                                      ymax = .data$upper_CI), alpha = 0.25, color = NA) + 
    ggplot2::theme_classic(base_size = 14) + ggplot2::scale_x_continuous(breaks = c(1, 
                                                                                    max(epreds_summary$time)), labels = c("LCA", "Present")) + 
    ggplot2::ylab(ylabel) + ggplot2::xlab("Time") + 
    ggplot2::theme(legend.title = ggplot2::element_blank(), 
                   strip.background = ggplot2::element_blank(), 
                   strip.text = ggplot2::element_blank(), plot.title = ggplot2::element_text(size = 14))
  
  return(p)
}
