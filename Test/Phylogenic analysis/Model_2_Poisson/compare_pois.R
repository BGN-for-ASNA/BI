# %%
library(brms)
library(ape)
library(reticulate)
library(BayesianInference)
library(jsonlite)

# 1. Load data and fit brms
phylo <- read.nexus("phylo.nex")
data_pois <- read.table("data_pois.txt", header = TRUE)
data_pois$obs <- 1:nrow(data_pois)
A <- vcv.phylo(phylo)
L <- t(chol(A))

cat("\nFitting brms Poisson model...\n")
if (file.exists("model_pois.rds")) {
  model_pois <- readRDS("model_pois.rds")
} else {
  model_pois <- brm(
    phen_pois ~ cofactor + (1 | gr(phylo, cov = A)) + (1 | obs),
    data = data_pois, family = poisson("log"),
    data2 = list(A = A),
    chains = 2, cores = 2, iter = 4000, warmup = 2000,
    control = list(adapt_delta = 0.95),
    refresh = 0
  )
  saveRDS(model_pois, "model_pois.rds")
}

# 2. Check BI results
if (file.exists("bi_post_pois.csv")) {
  bi_post <- read.csv("bi_post_pois.csv")
  
  cat("\n--- Comparison of Poisson Model Results ---\n")
  brms_sum <- summary(model_pois)
  brms_fixef <- brms_sum$fixed
  brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
  brms_sd_obs <- brms_sum$random$obs["sd(Intercept)", "Estimate"]
  
  # BI summaries
  bi_intercept_mean <- mean(bi_post$b_Intercept)
  bi_cofactor_mean <- mean(bi_post$b_cofactor)
  bi_sd_phylo_mean <- mean(bi_post$sd_phylo)
  bi_sd_obs_mean <- mean(bi_post$sd_obs)
  
  comparison <- data.frame(
    Parameter = c("Intercept (uncentered)", "cofactor", "sd_phylo", "sd_obs"),
    brms_Mean = c(
      brms_fixef["Intercept", "Estimate"],
      brms_fixef["cofactor", "Estimate"],
      brms_sd_phylo,
      brms_sd_obs
    ),
    BI_Mean = c(
      bi_intercept_mean,
      bi_cofactor_mean,
      bi_sd_phylo_mean,
      bi_sd_obs_mean
    )
  )
  
  print(comparison)
  write.csv(comparison, "comparison_pois.csv", row.names = FALSE)
  
  # Save to log.txt
  cat("\n--- POISSON MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
  write.table(comparison, file = "log.txt", append = TRUE, row.names = FALSE, sep = "\t", quote = FALSE)
  
  # 4. Density Plots
  cat("\nGenerating density plots...\n")
  dir.create("plots", showWarnings = FALSE)
  
  # Extract draws
  post_brms <- as.data.frame(model_pois)
  
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
  
  plot_density("Intercept (uncentered)", post_brms$b_Intercept, bi_post$b_Intercept, "pois_intercept")
  plot_density("cofactor", post_brms$b_cofactor, bi_post$b_cofactor, "pois_cofactor")
  plot_density("sd_phylo", post_brms$sd_phylo__Intercept, bi_post$sd_phylo, "pois_sd_phylo")
  plot_density("sd_obs", post_brms$sd_obs__Intercept, bi_post$sd_obs, "pois_sd_obs")
} else {
  cat("BI posterior file not found. Run fit_bi_pois.py first.\n")
}
