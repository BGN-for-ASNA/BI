import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(orig_path, vec_path, title, out_path):
    if not os.path.exists(orig_path) or not os.path.exists(vec_path):
        print(f"Skipping {title}: Files not found.")
        return
    
    df_orig = pd.read_csv(orig_path)
    df_vec = pd.read_csv(vec_path)
    
    cols = [c for c in df_orig.columns if c in df_vec.columns]
    n_cols = len(cols)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
        
    for i, col in enumerate(cols):
        df_orig[col].plot.kde(ax=axes[i], label='Original', color='blue', alpha=0.6, linewidth=3)
        df_vec[col].plot.kde(ax=axes[i], label='Vectorized', color='orange', alpha=0.6, linestyle='--')
        axes[i].set_title(f"Parameter: {col}")
        axes[i].legend()
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

# Model 1
plot_comparison(
    "Model_1_Spatial_Heterogeneity/bi_gamma_post.csv",
    "Model_1_Spatial_Heterogeneity/bi_gamma_vec_post.csv",
    "Model 1: Spatial Heterogeneity (+Gamma) Comparison",
    "comparison_gamma.png"
)

# Model 2
plot_comparison(
    "Model_2_Temporal_Heterogeneity/bi_ucln_post.csv",
    "Model_2_Temporal_Heterogeneity/bi_ucln_vec_post.csv",
    "Model 2: Temporal Heterogeneity (UCLN) Comparison",
    "comparison_ucln.png"
)
