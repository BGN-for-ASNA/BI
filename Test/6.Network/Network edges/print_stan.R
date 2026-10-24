library(bisonR)
set.seed(42)
sim <- simulate_bison_model("count", aggregated = TRUE, location_effect = TRUE, age_diff_effect = TRUE, num_nodes = 20, num_locations = 5, max_obs = 10)
df <- sim$df_sim
formula <- as.formula("(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")
fit <- bison_model(formula, data = df, model_type = "count", directed = TRUE, partial_pooling = TRUE, zero_inflated = TRUE, iter_sampling = 10, iter_warmup = 10, refresh = 0)
cat(fit$stan_model$code())
