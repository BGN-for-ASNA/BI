library(cmdstanr)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
n_ind <- if(length(args) > 0) as.integer(args[1]) else 400

# Use concatenated standalone Stan file
stan_file <- "cjs-ms-all.stan"
data_file <- "c:/Users/Sosa/Documents/BF/Test/Capture-Recapture (WIP)/cr-in-stan/case-studies/data/fleayi-stan-data.rds"

full_data <- readRDS(data_file)
orig_N <- dim(full_data$y)[2]
idx <- rep(1:orig_N, length.out = n_ind)
y_raw <- full_data$y[1, idx, , ]

# Fix y extraction
y_mat <- apply(y_raw, c(1, 2), function(x) {
    found <- which(x == 1)
    if (length(found) == 0) return(0L)
    as.integer(found[1])
})

stan_data <- list(
    I = as.integer(n_ind),
    J = as.integer(dim(y_raw)[2]),
    S = 3L,
    tau = as.double(full_data$tau[1:20, 1]),
    y = matrix(as.integer(y_mat), nrow=as.integer(n_ind)),
    ind = 0L,
    grainsize = 0L
)

mod <- cmdstan_model(stan_file)
start_time <- Sys.time()
fit <- mod$sample(
    data = stan_data,
    chains = 1,
    iter_warmup = 150,
    iter_sampling = 150,
    refresh = 1,
    show_exceptions = TRUE
)
end_time <- Sys.time()

exec_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("STAN_TIME:", exec_time, "\n")

# Extract means
draws <- fit$draws(format = "matrix")
h_means <- colMeans(draws[, grep("^h\\[", colnames(draws))])
q_means <- colMeans(draws[, grep("^q\\[", colnames(draws))])
p_means <- colMeans(draws[, grep("^p\\[", colnames(draws))])

results <- list(
    time = exec_time,
    h = h_means,
    q = q_means,
    p = p_means
)

library(jsonlite)
write_json(results, "stan_results.json", auto_unbox = TRUE)
cat("Results saved to stan_results.json\n")
