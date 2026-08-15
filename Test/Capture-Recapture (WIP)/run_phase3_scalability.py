import subprocess
import time
import os
import jax.numpy as jnp
from BayesForge import BayesForge
import sys

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from model_BF import run_bi_benchmark
from run_stan_benchmark import run_stan_benchmark as run_stan

def scalability_test():
    scales = [500, 5000, 50000]
    results = {}

    for n in scales:
        print(f"\n--- Testing Scalability: N = {n} ---\n")
        
        # BF Test
        print(f"Running BF Benchmark (N={n})...")
        try:
            # We modify run_bi_benchmark to return results
            # For simplicity, we just run and record from output
            run_bi_benchmark(n)
        except Exception as e:
            print(f"BF failed at N={n}: {e}")

        # Stan Test (only for small N if it takes too long)
        if n <= 5000:
            print(f"Running Stan Benchmark (N={n})...")
            try:
                run_stan(n)
            except Exception as e:
                print(f"Stan failed at N={n}: {e}")
        else:
            print(f"Skipping Stan for N={n} (Expected Timeout/Crash)")

if __name__ == "__main__":
    scalability_test()
