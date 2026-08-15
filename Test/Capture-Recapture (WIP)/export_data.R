library(jsonlite)
full_data <- readRDS('c:/Users/Sosa/Documents/BF/Test/Capture-Recapture (WIP)/cr-in-stan/case-studies/data/fleayi-stan-data.rds')

n_ind <- 400
idx <- 1:n_ind # Use first 400 for consistency
y_raw <- full_data$y[1, idx, , ]

# Fix y extraction (same as run_stan_standalone.R)
y_mat <- apply(y_raw, c(1, 2), function(x) {
    found <- which(x == 1)
    if (length(found) == 0) return(0L)
    as.integer(found[1])
})

# Recalculate f, l for safety
first_last <- function(y) {
    I <- nrow(y)
    J <- ncol(y)
    f_l <- matrix(0, nrow=I, ncol=2)
    for (i in 1:I) {
        seen <- which(y[i, ] > 0)
        if (length(seen) > 0) {
            f_l[i, 1] <- seen[1]
            f_l[i, 2] <- seen[length(seen)]
        }
    }
    return(f_l)
}
f_l <- first_last(y_mat)

tau <- as.double(full_data$tau[1:20, 1])

data_export <- list(
    y = y_mat,
    f = f_l[, 1],
    l = f_l[, 2],
    tau = tau,
    n_individuals = as.integer(n_ind),
    n_surveys = as.integer(ncol(y_mat)),
    n_states = 3L
)

write_json(data_export, "data_BF.json", auto_unbox = TRUE)
cat("Data exported to data_BF.json\n")
