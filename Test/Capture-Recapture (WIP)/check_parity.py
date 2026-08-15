import sys
import os
import numpy as np

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_stan_benchmark import run_stan_benchmark
from model_BF import run_bi_benchmark

def check_parity():
    print("\n=== Parity Check: Stan vs. BF (N=400) ===\n")
    
    # 1. Run Stan
    print("Running Stan...")
    stan_res = run_stan_benchmark(400)
    
    # 2. Run BF
    print("\nRunning BF...")
    BF_res = run_bi_benchmark(400)
    
    # Comparison logic would follow here normally, 
    # but run_bi_benchmark currently prints means.
    # We will manually inspect output for now.
    print("\nParity Check Finished. Please compare the means above.")

if __name__ == "__main__":
    check_parity()
