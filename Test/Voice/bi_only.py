import os
import sys
import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az

# Add BI to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from BI import bi

def main():
    # 1. Load Data
    with open('stan_data.json', 'r') as f:
        data = json.load(f)
    
    # Ensure Y is integer
    Y_int = jnp.array(data['Y'], dtype=jnp.int32)
    
    bi_data = {
        'N': data['N'],
        'M': data['M'],
        'J': data['J'],
        'd': jnp.array(data['d']),
        'id': jnp.array(data['id']),
        'Y': Y_int
    }

    # 2. Load Stan Reference
    stan_samples = pd.read_csv('stan_reference_samples.csv')
    
    # 3. Fit BI
    print("Running BI model...")
    m = bi('cpu')
    
    def bi_model(**d):
        # L: rates of each token (shape M)
        L = m.dist.exponential(1.0, shape=(d['M'],), name='L')
        # p: population probabilities of each token (shape M)
        p = m.dist.beta(2.0, 2.0, shape=(d['M'],), name='p')
        # rate: J x M (L is M, d is J)
        rate = L[None, :] * d['d'][:, None]
        # gate: probability of structural zero (1-p)
        gate = 1.0 - p[None, :]
        
        # Ensure rate is positive and broadcasted
        m.dist.zero_inflated_poisson(gate=gate, rate=rate, obs=d['Y'], name='Y')

    m.fit(bi_model, obs=bi_data, num_samples=1000, num_warmup=1000, num_chains=2, seed=42)
    bi_post = m.posteriors

    # 4. Compare densities
    print("Generating comparison plots...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i in range(4):
        # L
        stan_l = stan_samples[f'L[{i+1}]'].values
        bi_l = np.array(bi_post['L'][:, i])
        az.plot_density([stan_l, bi_l], data_labels=['Stan', 'BI'], ax=axes[0, i], colors=['blue', 'orange'])
        axes[0, i].set_title(f'L[{i}] Comparison')
        axes[0, i].set_xlim(0, max(stan_l.max(), bi_l.max()) * 1.1)

        # p
        idx_stan = i + 1
        stan_p = stan_samples[f'p[{idx_stan}]'].values
        bi_p = np.array(bi_post['p'][:, i])
        az.plot_density([stan_p, bi_p], data_labels=['Stan', 'BI'], ax=axes[1, i], colors=['blue', 'orange'])
        axes[1, i].set_title(f'p[{i}] Comparison')
        axes[1, i].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('debug_plot_v2.png')
    print("Plots saved to debug_plot_v2.png")

if __name__ == "__main__":
    main()
