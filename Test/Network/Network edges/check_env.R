cat("--- R Environment Check ---\n")
cat("R version:", as.character(getRversion()), "\n")
cat(".libPaths():\n")
print(.libPaths())

pkgs <- c("reticulate", "BayesianInference", "bisonR", "cmdstanr")
for (p in pkgs) {
  cat("\nChecking package:", p, "\n")
  tryCatch({
    library(p, character.only = TRUE)
    cat("  Loaded successfully. Version:", as.character(packageVersion(p)), "\n")
  }, error = function(e) {
    cat("  FAILED to load:", conditionMessage(e), "\n")
  })
}

cat("\n--- Python Check via reticulate ---\n")
tryCatch({
  library(reticulate)
  cat("Python path:", py_config()$python, "\n")
  cat("Python version:", py_config()$version, "\n")
  bi <- import("BI")
  cat("BI python package imported successfully.\n")
}, error = function(e) {
  cat("Python check failed:", conditionMessage(e), "\n")
})
