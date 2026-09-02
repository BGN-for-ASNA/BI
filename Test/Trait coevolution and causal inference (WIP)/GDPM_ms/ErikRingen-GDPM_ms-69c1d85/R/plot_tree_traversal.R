plot_tree_traversal <- function(seed=456) {
  # patchwork needs to be loaded for the / operator
  library(patchwork)
  
  # Simple OU process simulation for visualization
  simulate_ou <- function(n, eta0, A_diag = -0.5, sigma = 0.3, dt = 0.01) {
    eta <- numeric(n)
    eta[1] <- eta0
    for (i in 2:n) {
      eta[i] <- eta[i-1] + A_diag * eta[i-1] * dt + sigma * sqrt(dt) * rnorm(1)
    }
    return(eta)
  }
  
  set.seed(seed)
  t_seq <- seq(0, 1, length.out = 80)
  eta1 <- simulate_ou(80, eta0 = 0, A_diag = -0.4, sigma = 0.5)
  eta2 <- simulate_ou(80, eta0 = 0, A_diag = -0.6, sigma = 0.4)
  eta2 <- eta2 + 0.4 * eta1  # add correlation
  
  trait_data <- data.frame(
    t = rep(t_seq, 2),
    eta = c(eta1, eta2),
    trait = factor(rep(c("trait1", "trait2"), each = 80), levels = c("trait1", "trait2"))
  )
  
  p_traits <- ggplot2::ggplot(trait_data, ggplot2::aes(x = t, y = eta, color = trait)) +
    ggplot2::annotate("rect", xmin = -0.05, xmax = 1.05, ymin = -Inf, ymax = Inf, 
             fill = "#9b59b6", alpha = 0.08) +
    ggplot2::geom_line(linewidth = 1.3) +
    ggplot2::geom_point(data = trait_data[trait_data$t == min(trait_data$t), ], size = 4) +
    ggplot2::geom_point(data = trait_data[trait_data$t == max(trait_data$t), ], size = 4) +
    ggplot2::scale_color_manual(values = c("#9b59b6", "#1abc9c"),
                       labels = c(expression(eta[1]), expression(eta[2]))) +
    ggplot2::labs(x = "Time along branch", y = "Latent trait value", color = NULL) +
    ggplot2::scale_x_continuous(breaks = c(0, 1), 
                       labels = c(expression(t[1]~(parent)), expression(t[2]~(child))),
                       expand = c(0.02, 0.02)) +
    ggplot2::scale_y_continuous(labels = NULL) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      legend.position = c(0.92, 0.85),
      legend.background = ggplot2::element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 11),
      axis.ticks.y = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(10, 20, 10, 20)
    )
  
  # ===========================================================================
  # Panel 2: Phylogenetic tree with traversal
  # ===========================================================================
  # Tree topology: ((A,B),(C,(D,E)))
  
  p_tree <- ggplot2::ggplot() +
    # Main trunk: root to first split
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = 0.25, y = 0.5, yend = 0.5), linewidth = 1.2, color = "gray20") +
    # First split - vertical connector
    ggplot2::geom_segment(ggplot2::aes(x = 0.25, xend = 0.25, y = 0.25, yend = 0.75), linewidth = 1.2, color = "gray20") +
    # Upper clade (A, B)
    ggplot2::geom_segment(ggplot2::aes(x = 0.25, xend = 0.5, y = 0.75, yend = 0.75), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 0.5, y = 0.65, yend = 0.85), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 1, y = 0.85, yend = 0.85), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 1, y = 0.65, yend = 0.65), linewidth = 1.2, color = "gray20") +
    # Lower clade (C, (D, E))
    ggplot2::geom_segment(ggplot2::aes(x = 0.25, xend = 0.5, y = 0.25, yend = 0.25), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 0.5, y = 0.15, yend = 0.35), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 1, y = 0.35, yend = 0.35), linewidth = 1.2, color = "gray20") +
    # Sub-clade (D, E)
    ggplot2::geom_segment(ggplot2::aes(x = 0.5, xend = 0.7, y = 0.15, yend = 0.15), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.7, xend = 0.7, y = 0.05, yend = 0.25), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.7, xend = 1, y = 0.25, yend = 0.25), linewidth = 1.2, color = "gray20") +
    ggplot2::geom_segment(ggplot2::aes(x = 0.7, xend = 1, y = 0.05, yend = 0.05), linewidth = 1.2, color = "gray20") +
    # Internal nodes (blue)
    ggplot2::geom_point(ggplot2::aes(x = 0, y = 0.5), size = 7, color = "#3498db") +
    ggplot2::geom_point(ggplot2::aes(x = 0.25, y = 0.5), size = 5, color = "#3498db") +
    ggplot2::geom_point(ggplot2::aes(x = 0.5, y = 0.75), size = 5, color = "#3498db") +
    ggplot2::geom_point(ggplot2::aes(x = 0.5, y = 0.25), size = 5, color = "#3498db") +
    ggplot2::geom_point(ggplot2::aes(x = 0.7, y = 0.15), size = 5, color = "#3498db") +
    # Tip nodes (red)
    ggplot2::geom_point(ggplot2::aes(x = 1, y = 0.85), size = 5, color = "#e74c3c") +
    ggplot2::geom_point(ggplot2::aes(x = 1, y = 0.65), size = 5, color = "#e74c3c") +
    ggplot2::geom_point(ggplot2::aes(x = 1, y = 0.35), size = 5, color = "#e74c3c") +
    ggplot2::geom_point(ggplot2::aes(x = 1, y = 0.25), size = 5, color = "#e74c3c") +
    ggplot2::geom_point(ggplot2::aes(x = 1, y = 0.05), size = 5, color = "#e74c3c") +
    # Node labels (η vector at different time points)
    ggplot2::annotate("text", x = -0.02, y = 0.56, label = expression(bold(eta)(t[0])), size = 3.5, hjust = 1) +
    ggplot2::annotate("text", x = 0.23, y = 0.56, label = expression(bold(eta)(t[1])), size = 3.5, hjust = 1) +
    ggplot2::annotate("text", x = 0.48, y = 0.81, label = expression(bold(eta)(t[2])), size = 3.5, hjust = 1) +
    # Tip labels
    ggplot2::annotate("text", x = 1.03, y = 0.85, label = "A", hjust = 0, size = 4, fontface = "bold") +
    ggplot2::annotate("text", x = 1.03, y = 0.65, label = "B", hjust = 0, size = 4, fontface = "bold") +
    ggplot2::annotate("text", x = 1.03, y = 0.35, label = "C", hjust = 0, size = 4, fontface = "bold") +
    ggplot2::annotate("text", x = 1.03, y = 0.25, label = "D", hjust = 0, size = 4, fontface = "bold") +
    ggplot2::annotate("text", x = 1.03, y = 0.05, label = "E", hjust = 0, size = 4, fontface = "bold") +
    # Highlighted branch (connects to top panel)
    ggplot2::annotate("segment", x = 0.25, xend = 0.5, y = 0.75, yend = 0.75, 
             linewidth = 4, color = "#9b59b6", alpha = 0.5) +
    # Arrow pointing up to top panel
    ggplot2::annotate("curve", x = 0.375, xend = 0.375, y = 0.80, yend = 1.08,
             curvature = 0, linewidth = 1, color = "#9b59b6",
             arrow = grid::arrow(length = grid::unit(0.3, "cm"), type = "closed", ends = "last")) +
    ggplot2::annotate("label", x = 0.375, y = 0.94, label = "see above", 
             size = 3, color = "#9b59b6", fontface = "italic",
             fill = "white", label.padding = grid::unit(0.15, "lines")) +
    # Time axis
    ggplot2::annotate("segment", x = -0.08, xend = 1.08, y = -0.1, yend = -0.1,
             arrow = grid::arrow(length = grid::unit(0.25, "cm"), type = "closed"), 
             linewidth = 0.8, color = "gray30") +
    ggplot2::annotate("text", x = 0.5, y = -0.18, label = "Time", size = 4) +
    ggplot2::annotate("text", x = 0, y = -0.18, label = expression(t[0]), size = 3.5) +
    ggplot2::annotate("text", x = 1, y = -0.18, label = "present", size = 3.5) +
    ggplot2::coord_cartesian(xlim = c(-0.12, 1.15), ylim = c(-0.22, 1.05), clip = "off") +
    ggplot2::theme_void() +
    ggplot2::theme(plot.margin = ggplot2::margin(10, 40, 30, 20))

  
  p_combined <- p_traits / p_tree +
    patchwork::plot_layout(heights = c(1, 1.5))
  
  ggplot2::ggsave("figures/dpm_algorithm_updated.png", p_combined,
         width = 9, height = 9, dpi = 600, bg = "white")
  
  return(p_combined)
}
