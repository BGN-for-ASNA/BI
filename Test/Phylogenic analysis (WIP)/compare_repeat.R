library(brms)
library(ape)
library(reticulate)

# 1. Load data and fit brms
phylo <- read.nexus("phylo.nex")
data_repeat <- read.table("data_repeat.txt", header = TRUE)
data_repeat$spec_mean_cf <- with(data_repeat, tapply(cofactor, species, mean)[species])
A <- vcv.phylo(phylo)

cat("\nFitting brms model...\n")
model_repeat <- brm(
  phen ~ spec_mean_cf + (1 | gr(phylo, cov = A)) + (1 | species),
  data = data_repeat, family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), "b"),
    prior(normal(0, 50), "Intercept"),
    prior(student_t(3, 0, 20), "sd"),
    prior(student_t(3, 0, 20), "sigma")
  ),
  chains = 2, cores = 2, iter = 4000, warmup = 2000,
  refresh = 0
)

# 2. Load BI results
bi_post <- read.csv("bi_post_repeat.csv")

# 3. Compare Results
cat("\n--- Comparison of Posterior Means ---\n")
brms_sum <- summary(model_repeat)
brms_fixef <- brms_sum$fixed
brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
brms_sd_spec <- brms_sum$random$species["sd(Intercept)", "Estimate"]
brms_sigma <- brms_sum$spec_pars["sigma", "Estimate"]

comparison <- data.frame(
  Parameter = c("Intercept (uncentered)", "spec_mean_cf", "sd_phylo", "sd_species", "sigma"),
  brms_Mean = c(
    brms_fixef["Intercept", "Estimate"],
    brms_fixef["spec_mean_cf", "Estimate"],
    brms_sd_phylo,
    brms_sd_spec,
    brms_sigma
  ),
  BI_Mean = c(
    mean(bi_post$b_Intercept),
    mean(bi_post$b_spec_mean_cf),
    mean(bi_post$sd_phylo),
    mean(bi_post$sd_species),
    mean(bi_post$sigma)
  )
)

print(comparison)
write.csv(comparison, "comparison_repeat.csv", row.names = FALSE)

# Save to log.txt
cat("\n--- REPEAT MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
write.table(comparison, file = "log.txt", append = TRUE, row.names = FALSE, sep = "\t", quote = FALSE)

# 4. Density Plots
cat("\nGenerating density plots...\n")
dir.create("plots", showWarnings = FALSE)

post_brms <- as.data.frame(model_repeat)

plot_density <- function(param_name, brms_samples, bi_samples, file_name) {
  svg(paste0("plots/", file_name, ".svg"), width = 6, height = 4)
  brms_samples <- as.numeric(as.vector(brms_samples))
  bi_samples <- as.numeric(bi_samples)
  
  d_brms <- density(brms_samples)
  d_bi <- density(bi_samples)
  
  xlim <- range(c(d_brms$x, d_bi$x))
  ylim <- range(c(d_brms$y, d_bi$y))
  
  plot(d_brms, col = "red", lwd = 2, main = paste("Density Comparison:", param_name), 
       xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density")
  lines(d_bi, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("brms", "BI"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
  dev.off()
}

plot_density("Intercept (uncentered)", post_brms$b_Intercept, bi_post$b_Intercept, "repeat_intercept")
plot_density("spec_mean_cf", post_brms$b_spec_mean_cf, bi_post$b_spec_mean_cf, "repeat_spec_mean_cf")
plot_density("sd_phylo", post_brms$sd_phylo__Intercept, bi_post$sd_phylo, "repeat_sd_phylo")
plot_density("sd_species", post_brms$sd_species__Intercept, bi_post$sd_species, "repeat_sd_species")
plot_density("sigma", post_brms$sigma, bi_post$sigma, "repeat_sigma")
