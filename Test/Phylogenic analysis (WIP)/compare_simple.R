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
# %%
# Load brms model for comparison
if (file.exists("model_simple.rds")) {
  model_simple <- readRDS("model_simple.rds")
} else {
  stop("brms model results not found.")
}

# 2. Fit BI Model
m <- importBI("cpu")
jnp <- import("jax.numpy")
jax <- import("jax")

# Map species factors to indices
data_simple$phylo_idx <- as.integer(as.factor(data_simple$phylo)) - 1L
mean_cofactor_simple <- mean(data_simple$cofactor)

# Initialize data on model
m$data_on_model <- list()
m$data_on_model$phen <- jnp$array(data_simple$phen, dtype = jnp$float32)
m$data_on_model$cofactor <- jnp$array(data_simple$cofactor - mean_cofactor_simple, dtype = jnp$float32)
m$data_on_model$phylo_idx <- jnp$array(data_simple$phylo_idx, dtype = jnp$int32)
m$data_on_model$A_cholesky <- jnp$array(L, dtype = jnp$float32)

model <- function(phen, cofactor, phylo_idx, A_cholesky) {
  # Priors - Aligned with brms
  intercept <- m$dist$normal(0, 50, name = "Intercept")
  beta_cofactor <- m$dist$normal(0, 10, name = "b_cofactor")

  # sd_phylo and sigma - Aligned with brms student_t(3, 0, 20)
  sd_phylo <- m$dist$left_truncated_distribution(
    m$dist$student_t(3, 0, 20, create_obj = TRUE),
    low = 0.0, name = "sd_phylo"
  )
  sigma <- m$dist$left_truncated_distribution(
    m$dist$student_t(3, 0, 20, create_obj = TRUE),
    low = 0.0, name = "sigma"
  )

  # Non-centered parameterization for species effects
  # Use rep(0L, 200L) to ensure integer type for JAX
  z_phylo <- m$dist$normal(jnp$array(rep(0L, 200L)), 1.0, name = "z_phylo")
  u_phylo <- jnp$matmul(A_cholesky, z_phylo) * sd_phylo

  # Mean
  mu <- intercept + beta_cofactor * cofactor + reticulate::py_get_item(u_phylo, phylo_idx)

  # Likelihood
  m$dist$normal(mu, sigma, name = "obs", obs = phen)
}


cat("\nFitting BI model...\n")
# %%
tryCatch({
  m$fit(model, num_samples = 2000L, num_warmup = 1000L, num_chains = 1L)
}, error = function(e) {
  cat("\nBI Fit failed. Python Error:\n")
  print(py_last_error())
  stop(e)
})

# 3. Compare Results
cat("\n--- Comparison of Posterior Means ---\n")
brms_sum <- summary(model_simple)
brms_fixef <- brms_sum$fixed
brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
brms_sigma <- brms_sum$spec_pars["sigma", "Estimate"]

# Recover uncentered intercept (b_Intercept)
post_bi <- m$posteriors
post_bi$b_Intercept <- post_bi$Intercept - mean_cofactor_simple * post_bi$b_cofactor

# BI summaries
sum_bi <- m$summary()
bi_intercept_mean <- mean(as.numeric(np$array(post_bi$b_Intercept)))

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
    sum_bi["b_cofactor", "mean"],
    sum_bi["sd_phylo", "mean"],
    sum_bi["sigma", "mean"]
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
post_bi <- m$posteriors

plot_density <- function(param_name, brms_samples, bi_samples, file_name) {
  svg(paste0("plots/", file_name, ".svg"), width = 6, height = 4)
  # Ensure samples are numeric vectors
  np <- import("numpy")
  brms_samples <- as.numeric(as.vector(brms_samples))
  bi_samples <- as.numeric(np$array(bi_samples))
  
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

plot_density("Intercept (uncentered)", post_brms$b_Intercept, post_bi$b_Intercept, "simple_intercept")
plot_density("cofactor", post_brms$b_cofactor, post_bi$b_cofactor, "simple_cofactor")
plot_density("sd_phylo", post_brms$sd_phylo__Intercept, post_bi$sd_phylo, "simple_sd_phylo")
plot_density("sigma", post_brms$sigma, post_bi$sigma, "simple_sigma")
