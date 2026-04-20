import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import json
import numpy as np

def plot():
    # Load BI samples
    bi_data = jnp.load('bi_samples.npz')
    bi_h = bi_data['h']
    bi_q = bi_data['q']
    
    # Load Stan results (for means)
    with open('stan_results.json', 'r') as f:
        stan_results = json.load(f)
    
    stan_h = stan_results['h']
    stan_q = stan_results['q']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CJS-MS: BI (Density) vs Stan (Mean Baseline)', fontsize=16)
    
    # Plot h
    for i in range(3):
        sns.kdeplot(bi_h[:, i], ax=axes[0, i], fill=True, label='BI Posterior')
        axes[0, i].axvline(stan_h[i], color='red', linestyle='--', label='Stan Mean')
        axes[0, i].set_title(f'Posterior h[{i+1}]')
        axes[0, i].legend()
        
    # Plot q (first 3)
    for i in range(3):
        sns.kdeplot(bi_q[:, i], ax=axes[1, i], fill=True, label='BI Posterior')
        axes[1, i].axvline(stan_q[i], color='red', linestyle='--', label='Stan Mean')
        axes[1, i].set_title(f'Posterior q[{i+1}]')
        axes[1, i].legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison_plots.png')
    print("Plots saved to comparison_plots.png")

if __name__ == "__main__":
    plot()
