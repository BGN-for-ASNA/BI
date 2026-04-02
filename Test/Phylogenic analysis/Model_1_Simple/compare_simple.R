# %%
library(brms)
library(ape)
library(reticulate)
library(BayesianInference)
library(jsonlite)

# 1. Load data and brms results
phylo <- read.nexus("phylo.nex")
data_simple <- read.table("data_simple.txt", header = TRUE)
A <- vcv.phylo(phylo)
# De-compose A for BI
L <- t(chol(A))

# Load brms model for comparison
if (file.exists("model_simple.rds")) {
  model_simple <- readRDS("model_simple.rds")
} else {
  stop("brms model results not found.")
}

# 2. Check BI results
if (file.exists("bi_post_simple.csv")) {
  bi_post <- read.csv("bi_post_simple.csv")

  # 3. Compare Results
  cat("\n--- Comparison of Posterior Means ---\n")
  brms_sum <- summary(model_simple)
  brms_fixef <- brms_sum$fixed
  brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
  brms_sigma <- brms_sum$spec_pars["sigma", "Estimate"]

  # BI summaries
  bi_intercept_mean <- mean(bi_post$b_Intercept)
  bi_cofactor_mean <- mean(bi_post$b_cofactor)
  bi_sd_phylo_mean <- mean(bi_post$sd_phylo)
  bi_sigma_mean <- mean(bi_post$sigma)

  # Create comparison table
  comparison <- data.frame(
    Parameter = c("Intercept (uncentered)", "cofactor", "sd_phylo", "sigma"),
    brms_Mean = c(
      brms_fixef["Intercept", "Estimate"],
      brms_fixef["cofactor", "Estimate"],
      brms_sd_phylo,
      brms_sigma
    ),
    BI_Mean = c(
      bi_intercept_mean,
      bi_cofactor_mean,
      bi_sd_phylo_mean,
      bi_sigma_mean
    )
  )

  print(comparison)
  write.csv(comparison, "comparison_simple.csv", row.names = FALSE)

  # Save to log.txt
  cat("\n--- SIMPLE MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
  write.table(comparison, file = "log.txt", append = TRUE, row.names = FALSE, sep = "\t", quote = FALSE)

  # 4. Density Plots
  cat("\nGenerating density plots...\n")
  dir.create("plots", showWarnings = FALSE)

  # Extract draws
  post_brms <- as.data.frame(model_simple)

  plot_density <- function(param_name, brms_samples, bi_samples, file_name) {
    svg(paste0("plots/", file_name, ".svg"), width = 6, height = 4)
    brms_samples <- as.numeric(as.vector(brms_samples))
    bi_samples <- as.numeric(bi_samples)

    d_brms <- density(brms_samples)
    d_bi <- density(bi_samples)

    xlim <- range(c(d_brms$x, d_bi$x))
    ylim <- range(c(d_brms$y, d_bi$y))

    plot(d_brms,
      col = "red", lwd = 2, main = paste("Density Comparison:", param_name),
      xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density"
    )
    lines(d_bi, col = "blue", lwd = 2, lty = 2)
    legend("topright", legend = c("brms", "BI"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
    dev.off()
  }

  plot_density("Intercept (uncentered)", post_brms$b_Intercept, bi_post$b_Intercept, "simple_intercept")
  plot_density("cofactor", post_brms$b_cofactor, bi_post$b_cofactor, "simple_cofactor")
  plot_density("sd_phylo", post_brms$sd_phylo__Intercept, bi_post$sd_phylo, "simple_sd_phylo")
  plot_density("sigma", post_brms$sigma, bi_post$sigma, "simple_sigma")
} else {
  cat("BI posterior file not found. Run fit_bi_simple.py first.\n")
}
# %%
