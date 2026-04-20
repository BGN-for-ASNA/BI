import pandas as pd
import numpy as np
import json
import os
import sys

# Add BI to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from BI import bi

def main():
    with open('stan_data.json', 'r') as f:
        data = json.load(f)
    Y_int = np.array(data['Y'], dtype=np.int32)
    bi_data = {'N': data['N'], 'M': data['M'], 'J': data['J'], 'd': np.array(data['d']), 'Y': Y_int}
    
    stan_samples = pd.read_csv('stan_reference_samples.csv')
    
    m = bi('cpu')
    def bi_model(**d):
        L = m.dist.exponential(1.0, shape=(d['M'],), name='L')
        p = m.dist.beta(2.0, 2.0, shape=(d['M'],), name='p')
        rate = L[None, :] * d['d'][:, None]
        gate = 1.0 - p[None, :]
        m.dist.zero_inflated_poisson(gate=gate, rate=rate, obs=d['Y'], name='Y')

    m.fit(bi_model, obs=bi_data, num_samples=1000, num_warmup=1000, num_chains=2, seed=42)
    bi_post = m.posteriors
    
    print(f"{'Param':<10} | {'Mean (S/BI)':<20} | {'SD (S/BI)':<20} | {'Diff Mean':<10}")
    print("-" * 70)
    
    for i in range(5):
        stan_m = stan_samples[f'p[{i+1}]'].mean()
        bi_m = bi_post['p'][:, i].mean()
        stan_s = stan_samples[f'p[{i+1}]'].std()
        bi_s = bi_post['p'][:, i].std()
        print(f"p[{i}] | {stan_m:.4f}/{bi_m:.4f} | {stan_s:.4f}/{bi_s:.4f} | {stan_m - bi_m:.4f}")

    for i in range(5):
        stan_m = stan_samples[f'L[{i+1}]'].mean()
        bi_m = bi_post['L'][:, i].mean()
        stan_s = stan_samples[f'L[{i+1}]'].std()
        bi_s = bi_post['L'][:, i].std()
        print(f"L[{i}] | {stan_m:.4f}/{bi_m:.4f} | {stan_s:.4f}/{bi_s:.4f} | {stan_m - bi_m:.4f}")

if __name__ == "__main__":
    main()
