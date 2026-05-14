#!/usr/bin/env Rscript

RESULTS_DIR <- "Test/Network/Network edges/results"

# Categories requested by user
CATEGORIES <- c("edge_weight", "beta_random", "edge_sigma", "beta_fixed", 
                "random_group_mu", "random_group_sigma", "zero_prob", "rate")

log_files <- list.files(RESULTS_DIR, pattern = "_log.txt$", recursive = TRUE, full.names = TRUE)

if (length(log_files) == 0) {
  stop("No log files found in ", RESULTS_DIR)
}

cat("Found", length(log_files), "log files.\n")

global_kl <- list()

for (f in log_files) {
  lines <- readLines(f)
  # Look for the parameter table
  # Format: Parameter   Stan_mean   BI_mean   Diff   KL(Stan||BI)
  
  start_idx <- grep("^Parameter", lines)
  if (length(start_idx) == 0) next
  
  sep_lines <- grep("^---", lines)
  header_sep <- sep_lines[sep_lines > start_idx][1]
  if (is.na(header_sep)) next
  
  table_end <- sep_lines[sep_lines > header_sep][1]
  if (is.na(table_end)) table_end <- length(lines)
  
  table_lines <- lines[(header_sep + 1):(table_end - 1)]
  
  for (l in table_lines) {
    parts <- strsplit(trimws(l), "\\s+")[[1]]
    if (length(parts) < 5) next
    
    param_name <- parts[1]
    kl_val     <- as.numeric(parts[5])
    
    if (is.na(kl_val) || is.nan(kl_val)) next
    
    # Categorize
    cat_name <- gsub("\\[[0-9]+\\]", "", param_name)
    if (cat_name %in% CATEGORIES) {
      if (is.null(global_kl[[cat_name]])) global_kl[[cat_name]] <- c()
      global_kl[[cat_name]] <- c(global_kl[[cat_name]], kl_val)
    }
  }
}

cat("\n=======================================================\n")
cat(sprintf("%-25s %12s %12s\n", "Category", "Mean KL", "Count"))
cat(paste(rep("-", 52), collapse = ""), "\n")

for (cat_name in CATEGORIES) {
  vals <- global_kl[[cat_name]]
  if (is.null(vals)) {
    cat(sprintf("%-25s %12s %12d\n", cat_name, "NA", 0))
  } else {
    cat(sprintf("%-25s %12.6f %12d\n", cat_name, mean(vals), length(vals)))
  }
}
cat("=======================================================\n")
