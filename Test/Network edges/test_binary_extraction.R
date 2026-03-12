library(bisonR)
library(brms)

cat("Simulating data...\n")
set.seed(42)
sim_data <- simulate_bison_model("binary", num_nodes=10)
df <- sim_data$df_sim

cat("Fitting brms model...\n")
# Using binary (not binary_conjugate) since we wrote the explicit binary Stan model
fit_edge <- bison_model(
  (event | duration) ~ dyad(node_1_id, node_2_id),
  data=df,
  model_type="binary"
)

cat("Extracting Stan data list...\n")
# bisonR uses brms underneath, but we can also use rstan/cmdstan extraction
stan_data <- brms::make_standata(
  fit_edge$formula,
  data = fit_edge$data,
  family = fit_edge$family,
  prior = fit_edge$prior,
  data2 = fit_edge$data2
)

str(stan_data)
saveRDS(stan_data, "Network edges/binary_stan_data.rds")
cat("Saved mapped data list to RDS.\n")
