library(bisonR)
df <- data.frame(
    event = rexp(10, 1),
    duration = runif(10, 1, 5),
    node_1_id = factor(sample(1:5, 10, replace = TRUE), levels = 1:5),
    node_2_id = factor(sample(1:5, 10, replace = TRUE), levels = 1:5)
)
fit <- bison_model(
    (event | duration) ~ dyad(node_1_id, node_2_id),
    data = df, model_type = "duration", iter_sampling = 10, iter_warmup = 10
)
print(names(fit$model_data))
print(head(fit$model_data$event))
print(head(fit$model_data$divisor))
print(head(fit$model_data$event_count))
