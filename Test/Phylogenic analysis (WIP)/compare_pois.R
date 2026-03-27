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

# 2. Fit BI Model
m <- importBI("cpu")
jnp <- import("jax.numpy")

# Mapping
data_pois$obs_idx <- as.integer(as.factor(data_pois$obs)) - 1L
data_pois$phylo_idx <- as.integer(as.factor(data_pois$phylo)) - 1L
mean_cofactor <- mean(data_pois$cofactor)

m$data_on_model <- list()
m$data_on_model$phen <- jnp$array(data_pois$phen_pois, dtype = jnp$int32)
m$data_on_model$cofactor <- jnp$array(data_pois$cofactor - mean_cofactor, dtype = jnp$float32)
m$data_on_model$phylo_idx <- jnp$array(data_pois$phylo_idx, dtype = jnp$int32)
m$data_on_model$obs_idx <- jnp$array(data_pois$obs_idx, dtype = jnp$int32)
m$data_on_model$A_cholesky <- jnp$array(L, dtype = jnp$float32)

model <- function(phen, cofactor, phylo_idx, obs_idx, A_cholesky) {
  # Priors - Aligned with exact brms Stan code
  intercept <- m$dist$student_t(3, 0.3, 2.6, name = "Intercept") # Centered intercept
  b_cofactor <- m$dist$normal(0, 10, name = "b_cofactor")

  # Hyperparameters for random effects - Aligned with brms student_t(3, 0, 2.6)
  sd_phylo <- m$dist$left_truncated_distribution(
    m$dist$student_t(3, 0, 2.6, create_obj = TRUE),
    low = 0.0, name = "sd_phylo"
  )
  sd_obs <- m$dist$left_truncated_distribution(
    m$dist$student_t(3, 0, 2.6, create_obj = TRUE),
    low = 0.0, name = "sd_obs"
  )

  # Species effects (Phylogenetic)
  # Hardcode 200L to avoid Tracer/Type issues from arguments
  z_phylo <- m$dist$normal(jnp$array(rep(0L, 200L)), 1.0, name = "z_phylo")
  u_phylo <- jnp$matmul(A_cholesky, z_phylo) * sd_phylo

  # Observation-level random effects (Overdispersion)
  # Hardcode 200L to avoid Tracer/Type issues from arguments
  z_obs <- m$dist$normal(jnp$array(rep(0L, 200L)), 1.0, name = "z_obs")
  u_obs <- z_obs * sd_obs

  # Linear Predictor (centered)
  mu <- intercept + b_cofactor * cofactor +
    py_get_item(u_phylo, phylo_idx) +
    py_get_item(u_obs, obs_idx)

  # Likelihood
  m$dist$poisson(jnp$exp(mu), name = "obs", obs = phen)
}
# %%
cat("\nFitting BI Poisson model...\n")
tryCatch(
  {
    m$fit(model, num_samples = 2000L, num_warmup = 1000L, num_chains = 1L)
  },
  error = function(e) {
    cat("\nBI Fit failed. Python Error:\n")
    print(reticulate::py_last_error())
    stop(e)
  }
)

# Recover uncentered intercept (b_Intercept) to match brms output
post_bi <- m$posteriors
post_bi$b_Intercept <- post_bi$Intercept - mean_cofactor * post_bi$b_cofactor

cat("\n--- Comparison of Poisson Model Results ---\n")
brms_sum <- summary(model_pois)
brms_fixef <- brms_sum$fixed
brms_sd_phylo <- brms_sum$random$phylo["sd(Intercept)", "Estimate"]
brms_sd_obs <- brms_sum$random$obs["sd(Intercept)", "Estimate"]

# Calculate BI means including the recovered b_Intercept
sum_bi <- m$summary()
bi_intercept_mean <- mean(as.numeric(np$array(post_bi$b_Intercept)))

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
    sum_bi["b_cofactor", "mean"],
    sum_bi["sd_phylo", "mean"],
    sum_bi["sd_obs", "mean"]
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

  plot(d_brms,
    col = "red", lwd = 2, main = paste("Density Comparison:", param_name),
    xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density"
  )
  lines(d_bi, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("brms", "BI"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
  dev.off()
}

plot_density("Intercept (uncentered)", post_brms$b_Intercept, post_bi$b_Intercept, "pois_intercept")
plot_density("cofactor", post_brms$b_cofactor, post_bi$b_cofactor, "pois_cofactor")
plot_density("sd_phylo", post_brms$sd_phylo__Intercept, post_bi$sd_phylo, "pois_sd_phylo")
plot_density("sd_obs", post_brms$sd_obs__Intercept, post_bi$sd_obs, "pois_sd_obs")
