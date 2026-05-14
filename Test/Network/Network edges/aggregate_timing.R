#!/usr/bin/env Rscript
# Print timing comparison table from results/timing_summary.csv
# Usage: Rscript aggregate_timing.R [results_dir]

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) > 0) args[1] else "Test/Network/Network edges/results"

csv_path <- file.path(results_dir, "timing_summary.csv")
if (!file.exists(csv_path)) {
  cat("No timing_summary.csv found at:", csv_path, "\n")
  quit(status = 1)
}

d <- read.csv(csv_path, stringsAsFactors = FALSE)

# Keep latest run per model
d <- d[order(d$timestamp), ]
d <- d[!duplicated(d$model, fromLast = TRUE), ]
d <- d[order(d$model), ]

cat(sprintf("\n%-35s  %8s  %8s  %7s  %s\n",
    "Model", "Stan(s)", "BI(s)", "Ratio", "Timestamp"))
cat(paste(rep("-", 75), collapse = ""), "\n")
for (i in seq_len(nrow(d))) {
  cat(sprintf("%-35s  %8.1f  %8.1f  %6.1fx  %s\n",
    d$model[i], d$stan_sec[i], d$bi_sec[i], d$stan_bi_ratio[i], d$timestamp[i]))
}
cat(paste(rep("-", 75), collapse = ""), "\n")
cat(sprintf("%-35s  %8.1f  %8.1f  %6.1fx\n",
    "MEAN", mean(d$stan_sec), mean(d$bi_sec), mean(d$stan_bi_ratio)))
cat("\n")
cat(sprintf("Settings: %d chains, %d warmup + %d samples\n",
    d$num_chains[1], d$iter_warmup[1], d$iter_samples[1]))
