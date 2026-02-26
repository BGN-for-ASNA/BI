library(reticulate)
library(BayesianInference)

m <- importBI("cpu")
jnp <- import("jax.numpy")
np  <- import("numpy")

source("BI_Models/model_cTADA.R")
source("density_plots.R") # Load the functions including bi_draws_from_m

dl <- build_data_list()
m$data_on_model <- list(data = dl$jax)

cat("Fitting model cTADA...\n")
m$fit(bi_model_cTADA, num_chains=2L, num_warmup=10L, num_samples=20L)

cat("Extracting posteriors directly from m$posteriors...\n")
raw <- m$posteriors
cat("Raw names:", paste(names(raw), collapse=", "), "\n")

cat("\nTesting bi_draws_from_m extraction...\n")
res <- bi_draws_from_m()
cat("Final renamed keys:", paste(names(res), collapse=", "), "\n")
