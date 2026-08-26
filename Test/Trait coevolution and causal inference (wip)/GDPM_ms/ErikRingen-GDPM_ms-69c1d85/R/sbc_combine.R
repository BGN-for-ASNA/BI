parse_N <- function(name){
  sample_sizes <- stringr::str_extract(name, "(?<=sbc_)\\d+")
  return(sample_sizes)
}

parse_design <- function(name){
  design <- stringr::str_replace(name, "^(?:[^_]*_){2}", "")
  design <- ifelse(design == "2traits", "2traits_normal_bernoulli", design)
  return(design)
}

sbc_combine <- function(sbc_list, n_traits) {

  result <- do.call(SBC::bind_results, sbc_list)

  vars = list(
  A = c("A[1,2]", "A[2,2]", "A[2,1]", "A[1,1]"),
  Q = c("Q[2,2]", "Q[1,1]"),
  b = c("b[1]", "b[2]")
  )
  combine_list <- list(
    "A_diag" = c("A[1,1]", "A[2,2]"),
    "Q_diag" = c("Q[1,1]", "Q[2,2]"),
    "A_offdiag" = c("A[1,2]", "A[2,1]"),
    "b" = c("b[1]", "b[2]")
  )
  if (n_traits == "3"){
    vars$A <- c(vars$A, "A[3,3]", "A[3,1]", "A[1,3]", "A[2,3]", "A[3,2]")
    vars$Q <- c(vars$Q, "Q[3,3]")
    vars$b <- c(vars$b, "b[3]")
    combine_list$A_diag <- c(combine_list$A_diag, "A[3,3]")
    combine_list$A_offdiag <- c(combine_list$A_offdiag, "A[1,3]", "A[3,1]", "A[2,3]", "A[3,2]")
    combine_list$Q_diag <- c(combine_list$Q_diag, "Q[3,3]")
    combine_list$b <- c(combine_list$b, "b[3]")
  }

  p <- SBC::plot_ecdf_diff(result, variables = unlist(vars), combine_variables = combine_list) + 
    ggplot2::theme_classic(base_size = 15) + 
    ggplot2::xlab("Fractional rank") +
    ggplot2::ylab("ECDF difference") +
    ggplot2::ggtitle(paste(n_traits, "co-evolving traits")) + 
    ggplot2::theme(strip.background = ggplot2::element_blank()) + 
    ggplot2::scale_y_continuous(limits = c(-0.15, 0.15)) + 
    ggplot2::scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"))
  
  return(p)
}

sbc_combined_plot <- function(
  sbc_64_2traits,
  sbc_128_2traits,
  sbc_256_2traits,
  sbc_64_2traits_double_gaussian,
  sbc_128_2traits_double_gaussian,
  sbc_256_2traits_double_gaussian,
  sbc_64_2traits_double_bernoulli,
  sbc_128_2traits_double_bernoulli,
  sbc_256_2traits_double_bernoulli,
  sbc_64_3traits_2gaussian_1binary,
  sbc_128_3traits_2gaussian_1binary,
  sbc_256_3traits_2gaussian_1binary,
  sbc_64_3traits_2binary_1gaussian,
  sbc_128_3traits_2binary_1gaussian,
  sbc_256_3traits_2binary_1gaussian
){
  # Bundle 2-trait results
sbc_2traits_list <- list(
    sbc_64_2traits = sbc_64_2traits,
    sbc_128_2traits = sbc_128_2traits,
    sbc_256_2traits = sbc_256_2traits,
    sbc_64_2traits_double_gaussian = sbc_64_2traits_double_gaussian,
    sbc_128_2traits_double_gaussian = sbc_128_2traits_double_gaussian,
    sbc_256_2traits_double_gaussian = sbc_256_2traits_double_gaussian,
    sbc_64_2traits_double_bernoulli = sbc_64_2traits_double_bernoulli,
    sbc_128_2traits_double_bernoulli = sbc_128_2traits_double_bernoulli,
    sbc_256_2traits_double_bernoulli = sbc_256_2traits_double_bernoulli
  )
  
  # Bundle 3-trait results
  sbc_3traits_list <- list(
    sbc_64_3traits_2gaussian_1binary = sbc_64_3traits_2gaussian_1binary,
    sbc_128_3traits_2gaussian_1binary = sbc_128_3traits_2gaussian_1binary,
    sbc_256_3traits_2gaussian_1binary = sbc_256_3traits_2gaussian_1binary,
    sbc_64_3traits_2binary_1gaussian = sbc_64_3traits_2binary_1gaussian,
    sbc_128_3traits_2binary_1gaussian = sbc_128_3traits_2binary_1gaussian,
    sbc_256_3traits_2binary_1gaussian = sbc_256_3traits_2binary_1gaussian
  )
  
  p_2traits <- sbc_combine(sbc_2traits_list, "2")
  p_3traits <- sbc_combine(sbc_3traits_list, "3")

  p_comb <- p_2traits / p_3traits + patchwork::plot_layout(guides = 'collect')
  ggplot2::ggsave("figures/simulation_based_calibration/sbc.png", plot = p_comb, width = 6, height = 7, dpi = 900)
  return(p_comb)
}

actual_vs_fit_plot <- function(
  sbc_64_2traits,
  sbc_128_2traits,
  sbc_256_2traits,
  sbc_64_2traits_double_gaussian,
  sbc_128_2traits_double_gaussian,
  sbc_256_2traits_double_gaussian,
  sbc_64_2traits_double_bernoulli,
  sbc_128_2traits_double_bernoulli,
  sbc_256_2traits_double_bernoulli,
  sbc_64_3traits_2gaussian_1binary,
  sbc_128_3traits_2gaussian_1binary,
  sbc_256_3traits_2gaussian_1binary,
  sbc_64_3traits_2binary_1gaussian,
  sbc_128_3traits_2binary_1gaussian,
  sbc_256_3traits_2binary_1gaussian
){
  # Bundle 2-trait results
  sbc_2traits <- list(
    sbc_64_2traits = sbc_64_2traits,
    sbc_128_2traits = sbc_128_2traits,
    sbc_256_2traits = sbc_256_2traits,
    sbc_64_2traits_double_gaussian = sbc_64_2traits_double_gaussian,
    sbc_128_2traits_double_gaussian = sbc_128_2traits_double_gaussian,
    sbc_256_2traits_double_gaussian = sbc_256_2traits_double_gaussian,
    sbc_64_2traits_double_bernoulli = sbc_64_2traits_double_bernoulli,
    sbc_128_2traits_double_bernoulli = sbc_128_2traits_double_bernoulli,
    sbc_256_2traits_double_bernoulli = sbc_256_2traits_double_bernoulli
  )
  
  # Bundle 3-trait results
  sbc_3traits <- list(
    sbc_64_3traits_2gaussian_1binary = sbc_64_3traits_2gaussian_1binary,
    sbc_128_3traits_2gaussian_1binary = sbc_128_3traits_2gaussian_1binary,
    sbc_256_3traits_2gaussian_1binary = sbc_256_3traits_2gaussian_1binary,
    sbc_64_3traits_2binary_1gaussian = sbc_64_3traits_2binary_1gaussian,
    sbc_128_3traits_2binary_1gaussian = sbc_128_3traits_2binary_1gaussian,
    sbc_256_3traits_2binary_1gaussian = sbc_256_3traits_2binary_1gaussian
  )
  
  stats_2traits <- list()
  for (name in names(sbc_2traits)) {
    sbc_2traits[[name]]$stats$N <- parse_N(name)
    sbc_2traits[[name]]$stats$design <- parse_design(name)
    stats_2traits[[name]] <- sbc_2traits[[name]]$stats
  }
  combined_2traits <- dplyr::bind_rows(stats_2traits, .id = "source_name") |> 
    dplyr::filter(!(variable %in% c("Q[1,2]", "Q[2,1]")))
  
  stats_3traits <- list()
  for (name in names(sbc_3traits)) {
    sbc_3traits[[name]]$stats$N <- parse_N(name)
    sbc_3traits[[name]]$stats$design <- parse_design(name)
    stats_3traits[[name]] <- sbc_3traits[[name]]$stats
  }
  combined_3traits <- dplyr::bind_rows(stats_3traits, .id = "source_name") |> 
    dplyr::filter(!(variable %in% c("Q[1,2]", "Q[2,1]", "Q[3,1]", "Q[3,2]", "Q[2,3]")))
  
  combine_list <- list(
    "A_diag" = c("A[1,1]", "A[2,2]"),
    "Q_diag" = c("Q[1,1]", "Q[2,2]"),
    "A_offdiag" = c("A[1,2]", "A[2,1]"),
    "b" = c("b[1]", "b[2]")
  )
  combine_list3 <- combine_list
  combine_list3$A_diag <- c(combine_list3$A_diag, "A[3,3]")
  combine_list3$A_offdiag <- c(combine_list3$A_offdiag, "A[1,3]", "A[3,1]", "A[2,3]", "A[3,2]")
  combine_list3$Q_diag <- c(combine_list3$Q_diag, "Q[3,3]")
  combine_list3$b <- c(combine_list3$b, "b[3]")
  
  combined_2traits$group <- sapply(combined_2traits$variable, function(p) {
    grp <- names(combine_list)[sapply(combine_list, function(x) p %in% x)]
    if(length(grp) == 0) NA else grp
  })
  combined_3traits$group <- sapply(combined_3traits$variable, function(p) {
    grp <- names(combine_list3)[sapply(combine_list3, function(x) p %in% x)]
    if(length(grp) == 0) NA else grp
  })
  
  # Define consistent factor levels for group (columns) and design (rows)
  group_levels <- c("A_diag", "A_offdiag", "b", "Q_diag")
  
  # Compute z-scores: prior z-score for simulated, posterior z-score for estimate
  combined_2traits <- combined_2traits |> 
    dplyr::filter(!is.na(group)) |>
    dplyr::group_by(group) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      N = factor(N, levels = c("64", "128", "256")),
      group = factor(group, levels = group_levels)
    )
  
  combined_3traits <- combined_3traits |> 
    dplyr::filter(!is.na(group)) |>
    dplyr::group_by(group) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      N = factor(N, levels = c("64", "128", "256")),
      group = factor(group, levels = group_levels)
    )
  
  p1_2traits <- ggplot2::ggplot(combined_2traits, ggplot2::aes(x = simulated_value, y = mean)) +
    ggplot2::facet_wrap(design ~ group, ncol = 4, scales = "free") + 
    ggplot2::geom_point(alpha = 0.2, shape=1) + 
    ggplot2::geom_smooth(method = "lm", se = FALSE, aes(color = N)) +
    ggplot2::xlab("Simulated value") +
    ggplot2::ylab("Posterior mean") +
    ggplot2::ggtitle("2 co-evolving traits") +
    ggplot2::theme_classic(base_size = 10) +
    ggplot2::theme(strip.background = ggplot2::element_blank())
  
  p1_3traits <- ggplot2::ggplot(combined_3traits, ggplot2::aes(x = simulated_value, y = mean)) +
    ggplot2::facet_wrap(design ~ group, ncol = 4, scales = "free") + 
    ggplot2::geom_point(alpha = 0.2, shape=1) + 
    ggplot2::geom_smooth(method = "lm", se = FALSE, aes(color = N)) +
    ggplot2::xlab("Simulated value") +
    ggplot2::ylab("Posterior mean") +
    ggplot2::ggtitle("3 co-evolving traits") +
    ggplot2::theme_classic(base_size = 10) +
    ggplot2::theme(strip.background = ggplot2::element_blank())
  
  p_scatter <- p1_2traits / p1_3traits + patchwork::plot_layout(guides = 'collect')
  
  # Save figures
  ggplot2::ggsave("figures/simulation_based_calibration/sbc_scatter.png", plot = p_scatter, width = 8, height = 10, dpi = 300)
  
  return(p_scatter)
}
