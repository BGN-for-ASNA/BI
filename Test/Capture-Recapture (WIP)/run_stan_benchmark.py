import os
import sys
import time
import psutil
import pandas as pd
import numpy as np

# Setup R environment to avoid DLL issues on Windows
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ["R_HOME"] = R_HOME
os.environ["PATH"] = os.path.join(R_HOME, "bin", "x64") + ";" + os.environ["PATH"]

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    print("rpy2 not found in current environment.")
    sys.exit(1)

def run_stan_benchmark(n_individuals=400):
    print(f"--- Running Stan CJS-MS Benchmark (N={n_individuals}) ---")
    
    r_code = f"""
    library(cmdstanr)
    library(posterior)
    library(readr)
    
    repo_root <- "c:/Users/Sosa/Documents/BI/Test/Capture-Recapture/cr-in-stan"
    stan_file <- file.path(repo_root, "stan/cjs-ms.stan")
    data_file <- file.path(repo_root, "case-studies/data/fleayi-stan-data.rds")
    
    # Load and simplify data for CJS-MS
    full_data <- readRDS(data_file)
    
    # Expand by tiling if n_individuals > original N
    orig_N <- dim(full_data$y)[2]
    idx <- rep(1:orig_N, length.out = {n_individuals})
    y_raw <- full_data$y[1, idx, , ] 
    
    stan_data <- list(
        I = {n_individuals},
        J = full_data$J[1],
        S = 3,
        tau = full_data$tau[1:(full_data$J[1]-1), 1],
        y = apply(y_raw, c(1, 2), function(x) {{
            if (all(x == 0)) return(0)
            which(x == 1)
        }}),
        ind = 0,
        grainsize = 0
    )
    
    # Compile and Sample
    mod <- cmdstan_model(stan_file)
    start_time <- Sys.time()
    fit <- mod$sample(
        data = stan_data,
        chains = 1,
        iter_warmup = 150,
        iter_sampling = 150,
        refresh = 0,
        show_exceptions = FALSE
    )
    end_time <- Sys.time()
    
    # Get means directly from draws
    draws <- fit$draws(format = "matrix")
    h_idx <- grep("^h", colnames(draws))
    q_idx <- grep("^q", colnames(draws))
    
    h_means <- colMeans(draws[, h_idx, drop=FALSE])
    q_means <- colMeans(draws[, q_idx, drop=FALSE])
    
    list(
        time = as.numeric(difftime(end_time, start_time, units = "secs")),
        ess = 0,
        h_mean = as.numeric(h_means),
        q_mean = as.numeric(q_means)
    )
    """
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    try:
        res = ro.r(r_code)
        exec_time = res.rx2("time")[0]
        h_mean = np.array(res.rx2("h_mean"))
        q_mean = np.array(res.rx2("q_mean"))
        
        end_mem = process.memory_info().rss / (1024 * 1024)
        
        print(f"Execution Time: {exec_time:.2f} seconds")
        print(f"Posterior mean h: {h_mean}")
        print(f"Posterior mean q: {q_mean}")
        print(f"RAM Usage Increase: {end_mem - start_mem:.2f} MB")
        
        return {
            "n": n_individuals,
            "time": exec_time,
            "h_mean": h_mean,
            "q_mean": q_mean,
            "mem": end_mem - start_mem
        }
    except Exception as e:
        print(f"Error during Stan execution: {e}")
        return None

if __name__ == "__main__":
    n = 400
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    results = run_stan_benchmark(n)
