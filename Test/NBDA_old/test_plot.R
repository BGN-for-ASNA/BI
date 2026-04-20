library(reticulate)
library(BayesianInference)
library(STbayes)

m <- importBI("cpu")
jnp <- import("jax.numpy")
np  <- import("numpy")

source("BI_Models/model_cTADA.R")
source("density_plots.R")

dl <- build_data_list()
m$data_on_model <- list(data = dl$jax)

cat("Fitting...\n")
m$fit(bi_model_cTADA, num_chains=2L, num_warmup=50L, num_samples=50L)

bi_s <- bi_draws_from_m()
cat("Keys in bi_s:", paste(names(bi_s), collapse=", "), "\n")
cat("Length of log_lambda_0_mean:", length(bi_s[["log_lambda_0_mean"]]), "\n")
cat("Any NA in log_lambda_0_mean:", any(is.na(bi_s[["log_lambda_0_mean"]])), "\n")

# Load STbayes draws for cTADA
stan_code <- generate_STb_model(dl$raw, gq = FALSE, est_acqTime = FALSE)
write(stan_code, file = "temp_dens_cTADA.stan")
stb_fit <- suppressWarnings(fit_STb(dl$raw, model_obj = "temp_dens_cTADA.stan",
                         parallel_chains = 2, chains = 2, iter = 100, refresh = 0))

stb_s <- stb_draws(stb_fit, STB_PARAMS_COMMON)

cat("Keys in stb_s:", paste(names(stb_s), collapse=", "), "\n")

# Trace inside plot_densities
params_bi  <- names(bi_s)
params_stb <- names(stb_s)
scalar_bi  <- params_bi[!grepl("\\[", params_bi)]
scalar_stb <- params_stb[!grepl("\\[", params_stb)]
shared <- intersect(scalar_bi, scalar_stb)
cat("Shared keys:", paste(shared, collapse=", "), "\n")

# Let's see the means to make sure they are somewhat overlapping
cat("BI log_lambda_0_mean mean:", mean(bi_s[["log_lambda_0_mean"]]), "\n")
cat("STb log_lambda_0_mean mean:", mean(stb_s[["log_lambda_0_mean"]]), "\n")

plot_densities("cTADA_test", bi_s, stb_s)
