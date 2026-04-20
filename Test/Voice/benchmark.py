import os
import sys
import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az

# Add BI to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from BI import bi

def main():
    # 1. Load Data
    with open('stan_data.json', 'r') as f:
        data = json.load(f)
    
    # Convert data for BI (BI expects jnp arrays)
    bi_data = {
        'N': data['N'],
        'M': data['M'],
        'J': data['J'],
        'd': jnp.array(data['d']),
        'id': jnp.array(data['id']),
        'Y': jnp.array(data['Y'])
    }

    # 2. Run Stan
    print("Running Stan model...", flush=True)
    stan_file = 'cg_vocal_repertoires/model0.stan'
    model_stan = CmdStanModel(stan_file=stan_file)
    # Use more iterations for better comparison
    fit_stan = model_stan.sample(data='stan_data.json', seed=42, iter_sampling=1000, iter_warmup=1000, chains=2)
    
    # Extract Stan estimates
    stan_summary = fit_stan.summary()
    # L parameters are L[1] to L[M]
    L_stan_means = [stan_summary.loc[f'L[{i+1}]', 'Mean'] for i in range(data['M'])]
    p_stan_means = [stan_summary.loc[f'p[{i+1}]', 'Mean'] for i in range(data['M'])]

    # 3. Run BI
    print("Running BI model...", flush=True)
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
        # Obs Y is (J, M)
        m.dist.zero_inflated_poisson(gate=gate, rate=rate, obs=d['Y'], name='Y')

    m.fit(bi_model, obs=bi_data, num_samples=1000, num_warmup=1000, num_chains=2, seed=42)
    bi_summary = m.summary()

    # Extract BI estimates
    # L and p in BI summary might be arrays (L[0], L[1]...)
    L_bi_means = [bi_summary.loc[f'L[{i}]', 'mean'] for i in range(data['M'])]
    p_bi_means = [bi_summary.loc[f'p[{i}]', 'mean'] for i in range(data['M'])]

    # 4. Save results to log.txt
    print("Generating log.txt...")
    with open('log.txt', 'w') as f:
        f.write("Parameter | Stan Mean | BI Mean | Difference\n")
        f.write("-" * 50 + "\n")
        
        diff_L = []
        for i in range(data['M']):
            diff = L_stan_means[i] - L_bi_means[i]
            diff_L.append(diff)
            f.write(f"L[{i}] | {L_stan_means[i]:.4f} | {L_bi_means[i]:.4f} | {diff:.4f}\n")
            
        diff_p = []
        for i in range(data['M']):
            diff = p_stan_means[i] - p_bi_means[i]
            diff_p.append(diff)
            f.write(f"p[{i}] | {p_stan_means[i]:.4f} | {p_bi_means[i]:.4f} | {diff:.4f}\n")

    # 5. Generate density plots
    print("Generating density plots...")
    # Plot first 4 rates L and first 4 probabilities p
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Extract samples for plotting
    stan_post = fit_stan.draws_pd()
    bi_post = m.posteriors

    for i in range(4):
        # L
        stan_l = stan_post[f'L[{i+1}]']
        bi_l = bi_post['L'][:, i]
        az.plot_dist(stan_l, ax=axes[0, i], label='Stan', color='blue')
        az.plot_dist(bi_l, ax=axes[0, i], label='BI', color='orange')
        axes[0, i].set_title(f'Density Comparison: L[{i}]')
        axes[0, i].legend()

        # p
        stan_p = stan_post[f'p[{i+1}]']
        bi_p = bi_post['p'][:, i]
        az.plot_dist(stan_p, ax=axes[1, i], label='Stan', color='blue')
        az.plot_dist(bi_p, ax=axes[1, i], label='BI', color='orange')
        axes[1, i].set_title(f'Density Comparison: p[{i}]')
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    print("Plots saved to comparison_plot.png")

if __name__ == "__main__":
    main()
