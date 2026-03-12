# Verification script for BI Binary Model Fix

library(reticulate)
library(BayesianInference)
library(bisonR)
library(cmdstanr)

# Source the fixed model
source("Test/Network edges/bi_model_binary.R")
source("Test/Network edges/test_all_models.R") # For load/plot utilities

m <- importBI("cpu")
jnp <- import("jax.numpy")

# Run only the simplest binary test
model_type <- "binary"
set.seed(42)
sim_data <- simulate_bison_model(model_type, aggregated = TRUE)
df <- sim_data$df_sim

cat("Fitting bisonR model...\n")
fit_bison <- bison_model(
    (event | duration) ~ dyad(node_1_id, node_2_id),
    data = df,
    model_type = model_type,
    iter_sampling = 500, iter_warmup = 500
)

stan_data <- fit_bison$model_data
bi_data <- list(
    num_rows = as.integer(stan_data$num_rows),
    event = stan_data$event,
    divisor = stan_data$divisor,
    dyad_ids = stan_data$dyad_ids,
    num_edges = as.integer(stan_data$num_edges),
    num_fixed = 0L,
    num_random = 0L,
    num_random_groups = 0L,
    partial_pooling = 0L,
    zero_inflated = 0L
)

bi_data_jax <- bi_data
bi_data_jax$dyad_ids <- jnp$array(as.integer(as.character(bi_data$dyad_ids)), dtype = jnp$int32)
bi_data_jax$event <- jnp$array(as.integer(as.character(bi_data$event)), dtype = jnp$int32)
bi_data_jax$divisor <- jnp$array(as.integer(as.character(bi_data$divisor)), dtype = jnp$int32)

m$data_on_model <- list(data = bi_data_jax)
m$fit(bi_model_binary, num_warmup = 1000L, num_samples = 1000L, num_chains = 1L)

cat("Extracting draws and plotting...\n")
bison_edge_samples <- fit_bison$edge_samples
bi_draws <- get_bi_draws(m$posteriors)
true_vals <- sim_data$df_true$edge_weight[1:3]

brms_draws <- list()
for (idx in seq_len(min(3, ncol(bison_edge_samples)))) {
    brms_draws[[paste0("edge_weight[", idx, "]")]] <- as.numeric(bison_edge_samples[, idx])
}
brms_names <- paste0("edge_weight[", 1:3, "]")

# This will save to Test/Network edges/density_plots/verification_binary_densities.pdf
plot_densities("verification_binary", bi_draws, brms_draws,
    bi_prefix = "edge_weight",
    brms_prefix = brms_names,
    true_values = true_vals
)
cat("Verification complete. Check the plot at Test/Network edges/density_plots/verification_binary_densities.pdf\n")
