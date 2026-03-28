library(brms)
library(ape)

# 1. Load data and fit brms
phylo <- read.nexus("phylo.nex")
data_fisher <- read.table("data_effect.txt", header = TRUE)
data_fisher$obs <- 1:nrow(data_fisher)
data_fisher$se <- sqrt(1 / (data_fisher$N - 3))
A <- vcv.phylo(phylo)

cat("\nFitting brms meta-analysis model...\n")
model_meta <- brm(
  Zr | se(se) ~ 1 + (1|gr(phylo, cov = A)) + (1|obs),
  data = data_fisher, family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), "Intercept"),
    prior(student_t(3, 0, 10), "sd")
  ),
  chains = 2, cores = 2, iter = 4000, warmup = 2000,
  refresh = 0
)

# 2. Load BI results
bi_post <- read.csv("bi_post_meta.csv")

# 3. Compare Results
cat("\n--- Comparison of Posterior Means (Meta-Analysis) ---\n")
brms_sum <- summary(model_meta)
brms_fixef <- brms_sum$fixed
brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
brms_sd_obs <- brms_sum$random$obs["sd(Intercept)", "Estimate"]

comparison <- data.frame(
  Parameter = c("Intercept", "sd_phylo", "sd_obs"),
  brms_Mean = c(
    brms_fixef["Intercept", "Estimate"],
    brms_sd_phylo,
    brms_sd_obs
  ),
  BI_Mean = c(
    mean(bi_post$Intercept),
    mean(bi_post$sd_phylo),
    mean(bi_post$sd_obs)
  )
)

print(comparison)
write.csv(comparison, "comparison_meta.csv", row.names = FALSE)

# Save to log.txt
cat("\n--- META-ANALYSIS MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
write.table(comparison, file = "log.txt", append = TRUE, row.names = FALSE, sep = "\t", quote = FALSE)

# 4. Density Plots
cat("\nGenerating density plots...\n")
dir.create("plots", showWarnings = FALSE)

post_brms <- as.data.frame(model_meta)

plot_density <- function(param_name, brms_samples, bi_samples, file_name) {
  svg(paste0("plots/", file_name, ".svg"), width = 6, height = 4)
  brms_samples <- as.numeric(as.vector(brms_samples))
  bi_samples <- as.numeric(bi_samples)
  
  d_brms <- density(brms_samples)
  d_bi <- density(bi_samples)
  
  xlim <- range(c(d_brms$x, d_bi$x))
  ylim <- range(c(d_brms$y, d_bi$y))
  
  plot(d_brms, col = "red", lwd = 2, main = paste("Density: ", param_name), 
       xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density")
  lines(d_bi, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("brms", "BI"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
  dev.off()
}

plot_density("Intercept", post_brms$b_Intercept, bi_post$Intercept, "meta_intercept")
plot_density("sd_phylo", post_brms$sd_phylo__Intercept, bi_post$sd_phylo, "meta_sd_phylo")
plot_density("sd_obs", post_brms$sd_obs__Intercept, bi_post$sd_obs, "meta_sd_obs")
