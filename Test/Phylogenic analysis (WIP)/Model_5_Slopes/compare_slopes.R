# %%
library(brms)
library(ape)

# 1. Load data and fit brms benchmark
cat("Loading data and fitting brms benchmark (Model 6)...\n")
data_slopes <- read.table("data_slopes.txt", header = TRUE)
phylo <- read.nexus("phylo_slopes.nex")
A <- vcv.phylo(phylo)

if (!file.exists("model_slopes.rds")) {
  # Formula matches simulation: y ~ x + (1 + x | gr(phylo, cov = A))
  model_slopes <- brm(
    y ~ x + (1 + x | gr(phylo, cov = A)),
    data = data_slopes,
    data2 = list(A = A),
    family = gaussian(),
    prior = c(
      prior(normal(0, 10), class = Intercept),
      prior(normal(0, 10), class = b),
      prior(student_t(3, 0, 10), class = sd),
      prior(lkj(2), class = cor)
    ),
    chains = 2, cores = 1, iter = 2000, warmup = 1000,
    backend = "cmdstanr",
    control = list(adapt_delta = 0.95),
    refresh = 0
  )
  saveRDS(model_slopes, "model_slopes.rds")
} else {
  cat("Loading existing brms model from model_slopes.rds\n")
  model_slopes <- readRDS("model_slopes.rds")
}
# %%
# 2. Check BI results
if (file.exists("bi_post_slopes.csv")) {
  bi_post <- read.csv("bi_post_slopes.csv")

  post_brms <- as.data.frame(model_slopes)

  comparison <- data.frame(
    Parameter = c("Intercept", "b_x", "sigma", "sd_intercept", "sd_slope", "rho"),
    brms_Mean = c(
      if ("b_Intercept" %in% names(post_brms)) mean(post_brms$b_Intercept) else if ("Intercept" %in% names(post_brms)) mean(post_brms$Intercept) else NA,
      if ("b_x" %in% names(post_brms)) mean(post_brms$b_x) else if ("x" %in% names(post_brms)) mean(post_brms$x) else NA,
      mean(post_brms$sigma),
      mean(post_brms$sd_phylo__Intercept),
      mean(post_brms$sd_phylo__x),
      if ("cor_phylo__Intercept__x" %in% names(post_brms)) mean(post_brms$cor_phylo__Intercept__x) else NA
    ),
    BI_Mean = c(
      mean(bi_post$Intercept, na.rm = TRUE),
      mean(bi_post$b_x, na.rm = TRUE),
      mean(bi_post$sigma, na.rm = TRUE),
      mean(bi_post$sd_intercept, na.rm = TRUE),
      mean(bi_post$sd_slope, na.rm = TRUE),
      mean(bi_post$rho, na.rm = TRUE)
    )
  )
  comparison$Diff_Pct <- abs(comparison$brms_Mean - comparison$BI_Mean) / abs(comparison$brms_Mean) * 100

  print(comparison)
  write.csv(comparison, "comparison_slopes.csv", row.names = FALSE)

  cat("\n--- MODEL 6 COMPARISON ---\n", file = "log.txt", append = TRUE)
  write.table(comparison, file = "log.txt", append = TRUE, row.names = FALSE, sep = "\t", quote = FALSE)

  # 3. Density Plots
  cat("\nGenerating density plots...\n")
  dir.create("plots", showWarnings = FALSE)

  plot_density <- function(param_name, brms_samples, bi_samples, file_name) {
    brms_samples <- as.numeric(as.vector(brms_samples))
    bi_samples <- as.numeric(bi_samples)

    # Remove NAs
    brms_samples <- brms_samples[!is.na(brms_samples)]
    bi_samples <- bi_samples[!is.na(bi_samples)]

    if (length(brms_samples) < 2 || length(bi_samples) < 2) {
      cat("Skipping plot for", param_name, "due to insufficient data.\n")
      return()
    }

    svg(paste0("plots/", file_name, ".svg"), width = 6, height = 4)
    d_brms <- density(brms_samples)
    d_bi <- density(bi_samples)
    xlim <- range(c(d_brms$x, d_bi$x))
    ylim <- range(c(d_brms$y, d_bi$y))
    plot(d_brms,
      col = "red", lwd = 2, main = paste("Density: ", param_name),
      xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density"
    )
    lines(d_bi, col = "blue", lwd = 2, lty = 2)
    legend("topright", legend = c("brms", "BI"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
    dev.off()
  }

  plot_density("Intercept", post_brms$b_Intercept, bi_post$Intercept, "slopes_intercept")
  plot_density("b_x", post_brms$b_x, bi_post$b_x, "slopes_b_x")
  plot_density("sigma", post_brms$sigma, bi_post$sigma, "slopes_sigma")
  plot_density("sd_intercept", post_brms$sd_phylo__Intercept, bi_post$sd_intercept, "slopes_sd_intercept")
  plot_density("sd_slope", post_brms$sd_phylo__x, bi_post$sd_slope, "slopes_sd_slope")

  # Correlation name in dataframe might be different
  rho_name <- "cor_phylo__Intercept__x"
  if (rho_name %in% names(post_brms)) {
    plot_density("rho", post_brms[[rho_name]], bi_post$rho, "slopes_rho")
  }
} else {
  cat("BI posterior file not found. Run fit_bi_slopes.py first.\n")
}
