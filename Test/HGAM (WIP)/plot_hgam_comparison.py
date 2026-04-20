import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparison():
    models = ['G', 'GS', 'GI', 'S', 'I']
    all_logs = []
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        data_dir = f"bi_data/{model}"
        r_file = f"{data_dir}/results_r.csv"
        bi_file = f"{data_dir}/results_bi.csv"
        
        if not (os.path.exists(r_file) and os.path.exists(bi_file)):
            print(f"Results for {model} not found. Skipping plot.")
            continue
            
        r_res = pd.read_csv(r_file)
        bi_res = pd.read_csv(bi_file)
        
        # Combine
        comp = pd.DataFrame({
            'Model': model,
            'Parameter': [f"b_{j}" for j in range(len(r_res))],
            'R_Mean': r_res['coef'],
            'BI_Mean': bi_res['bi_mean'],
            'Difference': bi_res['bi_mean'] - r_res['coef']
        })
        
        all_logs.append(comp)
        
        # Parity Plot
        ax = axes[i]
        sns.regplot(data=comp, x='R_Mean', y='BI_Mean', ax=ax, scatter_kws={'alpha':0.5})
        ax.plot([comp['R_Mean'].min(), comp['R_Mean'].max()], 
                [comp['R_Mean'].min(), comp['R_Mean'].max()], 'r--')
        ax.set_title(f"Model {model}: R vs BI Parity")
        ax.set_xlabel("R (REML) Estimate")
        ax.set_ylabel("BI (MCMC) Mean")
        
    # Final log
    full_log = pd.concat(all_logs)
    full_log.to_csv("log.txt", index=False, sep="\t")
    
    # Summary of differences
    summary = full_log.groupby('Model')['Difference'].agg(['mean', 'std', 'max'])
    print("\nBenchmark Summary (Differences):")
    print(summary)
    
    plt.tight_layout()
    plt.savefig("plots/hgam_parity_comparison.png")
    print("\nPlots saved to plots/hgam_parity_comparison.png")

if __name__ == "__main__":
    generate_comparison()
