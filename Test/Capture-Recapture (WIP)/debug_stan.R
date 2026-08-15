library(cmdstanr)
library(readr)

repo_root <- "c:/Users/Sosa/Documents/BF/Test/Capture-Recapture/cr-in-stan"
stan_file <- file.path(repo_root, "stan/cjs-ms.stan")
data_file <- file.path(repo_root, "case-studies/data/fleayi-stan-data.rds")

full_data <- readRDS(data_file)
y_raw <- full_data$y[1, 1:400, , ]
stan_data <- list(
    I = 400,
    J = dim(y_raw)[2],
    S = 3,
    tau = full_data$tau[1:(dim(y_raw)[2]-1), 1],
    y = apply(y_raw, c(1, 2), function(x) {
        if (all(x == 0)) return(0)
        which(x == 1)
    }),
    ind = 0,
    grainsize = 0
)

mod <- cmdstan_model(stan_file)
fit <- mod$sample(data = stan_data, chains = 1, iter_warmup = 100, iter_sampling = 100, refresh = 10)
draws <- fit$draws(format = "matrix")
cat("Column names (subset):", head(colnames(draws), 20), "\n")
h_means <- colMeans(draws[, grep("^h\\[", colnames(draws))])
q_means <- colMeans(draws[, grep("^q\\[", colnames(draws))])
print(h_means)
print(q_means)
