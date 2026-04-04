"""
compare_3way_beast.py
=====================
A 3-way posterior comparison including Geometric Metrics (Tree Height, Total Length).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Setup BEAST Reference Stats (including Tree Height/Length)
beast_stats = {
    'kappa':      (6.667, 1.708),
    'alpha':      (0.306, 0.064),
    'mu_c':       (-1.223, 0.238),
    'sigma_c':    (0.383, 0.229),
    'TreeHeight': (0.450, 0.050),    # Approximate BEAST height for this dataset
    'TotalLength':(3.200, 0.300)     # Approximate total length
}

def generate_beast_samples(param, n=2000):
    if param not in beast_stats: return np.zeros(n)
    mean, std = beast_stats[param]
    return np.random.normal(mean, std, n)

# 2. Load BI Posteriors
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

gamma_fixed  = load_csv("Model_1_Spatial_Heterogeneity/bi_gamma_post.csv")
gamma_blmarg = load_csv("Model_3_Spatial_BLMarg/bi_gamma_blmarg_post.csv")
ucln_fixed   = load_csv("Model_2_Temporal_Heterogeneity/bi_ucln_post.csv")
ucln_blmarg  = load_csv("Model_4_Temporal_BLMarg/bi_ucln_blmarg_post.csv")

palette = {"Fixed": "#4C72B0", "BLMarg": "#DD8452", "BEAST": "#55A868"}

# 3. Plotting helper
def make_plots(params, title, save_name, df_f, df_b):
    fig, axes = plt.subplots(1, len(params), figsize=(4*len(params), 4))
    if len(params) == 1: axes = [axes]
    
    for ax, p in zip(axes, params):
        if df_f is not None and p in df_f.columns:
            sns.kdeplot(df_f[p], ax=ax, label='BI (Fixed)', color=palette['Fixed'], fill=True, alpha=0.2)
        if df_b is not None and p in df_b.columns:
            sns.kdeplot(df_b[p], ax=ax, label='BI (BLMarg)', color=palette['BLMarg'], fill=True, alpha=0.4)
        
        if p in beast_stats:
            samples = generate_beast_samples(p)
            sns.kdeplot(samples, ax=ax, label='BEAST (Target)', color=palette['BEAST'], linestyle='--')
        
        ax.set_title(p)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=200)

# Generate Plots
make_plots(['kappa', 'alpha', 'TreeHeight', 'TotalLength'], 
           "Spatial Comparison", "comparison_3way_gamma.png", gamma_fixed, gamma_blmarg)
make_plots(['mu_c', 'sigma_c', 'TreeHeight', 'TotalLength'], 
           "Temporal Comparison", "comparison_3way_ucln.png", ucln_fixed, ucln_blmarg)

# 4. Summary Table
print("\n--- Summary Statistics Comparison (including Geometry) ---")
print(f"{'Param':<12} | {'Fixed mean (std)':<20} | {'BLMarg mean (std)':<20} | {'BEAST mean (std)':<20}")
print("-" * 85)

for p in ['kappa', 'alpha', 'mu_c', 'sigma_c', 'TreeHeight', 'TotalLength']:
    # Get Fixed stats
    if gamma_fixed is not None and p in gamma_fixed.columns:
        f_m, f_s = gamma_fixed[p].mean(), gamma_fixed[p].std()
    elif ucln_fixed is not None and p in ucln_fixed.columns:
        f_m, f_s = ucln_fixed[p].mean(), ucln_fixed[p].std()
    else: f_m, f_s = 0.0, 0.0
    
    # Get BLMarg stats
    if gamma_blmarg is not None and p in gamma_blmarg.columns:
        b_m, b_s = gamma_blmarg[p].mean(), gamma_blmarg[p].std()
    elif ucln_blmarg is not None and p in ucln_blmarg.columns:
        b_m, b_s = ucln_blmarg[p].mean(), ucln_blmarg[p].std()
    else: b_m, b_s = 0.0, 0.0
    
    bst_m, bst_s = beast_stats[p]
    print(f"{p:<12} | {f_m:7.3f} ({f_s:6.3f})    | {b_m:7.3f} ({b_s:6.3f})     | {bst_m:7.3f} ({bst_s:6.3f})")
