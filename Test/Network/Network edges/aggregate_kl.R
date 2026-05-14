#!/usr/bin/env Rscript

# Script to aggregate KL divergence results by parameter category
# Usage: ./aggregate_kl.R <results_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  results_dir <- "results"
} else {
  results_dir <- args[1]
}

categories <- c("edge_weight", "beta_random", "edge_sigma", "beta_fixed", 
                "random_group_mu", "random_group_sigma", "zero_prob", "rate")

cat(sprintf("Aggregating KL divergence results from: %s\n", results_dir))
cat(sprintf("%-25s | %-10s\n", "Category", "Mean KL"))
cat(paste0(rep("-", 40), collapse = ""), "\n")

all_files <- list.files(results_dir, pattern = "_log\\.txt$", recursive = TRUE, full.names = TRUE)

if (length(all_files) == 0) {
  cat("No log files found.\n")
  quit(status = 0)
}

# Data structure to hold KL values for each category
kl_data <- list()
for (cat_name in categories) {
  kl_data[[cat_name]] <- numeric()
}

for (f in all_files) {
  lines <- readLines(f)
  # Find the table start (header contains "KL(Stan||BI)")
  header_idx <- grep("KL\\(Stan\\|\\|BI\\)", lines)
  if (length(header_idx) == 0) next
  
  # Process lines after the separator
  data_lines <- lines[(header_idx + 2):length(lines)]
  
  for (line in data_lines) {
    if (trimws(line) == "" || grepl("^-+$", trimws(line))) next
    
    parts <- strsplit(trimws(line), "\\s+")[[1]]
    if (length(parts) < 5) next
    
    param_name <- parts[1]
    kl_val <- as.numeric(parts[5])
    
    if (is.na(kl_val)) next
    
    # Identify category
    # Extract prefix (everything before '[' if present)
    base_name <- gsub("\\[.*\\]", "", param_name)
    
    if (base_name %in% categories) {
      kl_data[[base_name]] <- c(kl_data[[base_name]], kl_val)
    }
  }
}

for (cat_name in categories) {
  vals <- kl_data[[cat_name]]
  if (length(vals) > 0) {
    mean_kl <- mean(vals, na.rm = TRUE)
    cat(sprintf("%-25s | %-10.4f (n=%d)\n", cat_name, mean_kl, length(vals)))
  } else {
    cat(sprintf("%-25s | %-10s\n", cat_name, "N/A"))
  }
}
