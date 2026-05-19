library(rethinking)
source("cg_vocal_repertoires/simulation.r")

# Set seed for reproducibility
set.seed(42)

# Simulate data as done in simulation.r line 125
dat <- sim_repertoire(N=50, M=50, NR=10)
dat$a <- NULL

# Save data to a JSON file for cmdstanpy and a CSV for Python/BI
# Stan data needs: N, M, J, d, id, Y
# We'll also save true L and p for comparison if needed, though the task asks for Stan vs BI comparison.

# Prepare Stan data list
stan_data <- list(
    N = dat$N,
    M = dat$M,
    J = dat$J,
    d = dat$d,
    id = dat$id,
    Y = dat$Y
)

# Use jsonlite to save as JSON
if (!require("jsonlite")) install.packages("jsonlite", repos="https://cloud.r-project.org")
library(jsonlite)
write_json(stan_data, "stan_data.json", auto_unbox = TRUE)

# Also save Y, d, id as CSVs for easier loading in Python if JSON is tricky
write.table(dat$Y, "data_Y.csv", row.names=FALSE, col.names=FALSE, sep=",")
write.table(dat$d, "data_d.csv", row.names=FALSE, col.names=FALSE, sep=",")
write.table(dat$id, "data_id.csv", row.names=FALSE, col.names=FALSE, sep=",")

# Save ground truth for reference
write.table(dat$L, "true_L.csv", row.names=FALSE, col.names=FALSE, sep=",")
write.table(dat$p, "true_p.csv", row.names=FALSE, col.names=FALSE, sep=",")

cat("Data generation complete. Saved to stan_data.json and CSV files.\n")
