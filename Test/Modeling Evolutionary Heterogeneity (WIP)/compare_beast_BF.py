import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_beast_log(filepath):
    if not os.path.exists(filepath): return None
    # BEAST 2 logs have some comment lines at start starting with # usually
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_lines = [l for l in lines if not l.startswith('#')]
    
    with open("temp_log.csv", 'w') as f:
        f.writelines(data_lines)
        
    df = pd.read_csv("temp_log.csv", sep='\t')
    # Burn-in 10%
    burnin = int(0.1 * len(df))
    return df.iloc[burnin:]

def compare_model1():
    print("Comparing Model 1...")
    df_BF = pd.read_csv("Model_1_Spatial_Heterogeneity/BF_gamma_vec_post.csv")
    df_beast = load_beast_log("Model_1_Spatial_Heterogeneity/beast_model1_gamma.log")
    
    if df_beast is None:
        print("Model 1 BEAST output not found.")
        return
    
    # Map params:
    # BF: kappa, alpha
    # BEAST: hky.kappa, gammaShape
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.kdeplot(df_BF['kappa'], ax=axes[0], label='BF (Native)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(df_beast['hky.kappa'], ax=axes[0], label='BEAST 2', color='green', linestyle='--')
    axes[0].set_title('Kappa (ts/tv)')
    axes[0].legend()
    
    sns.kdeplot(df_BF['alpha'], ax=axes[1], label='BF (Native)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(df_beast['gammaShape'], ax=axes[1], label='BEAST 2', color='green', linestyle='--')
    axes[1].set_title('Gamma Shape (Alpha)')
    axes[1].legend()
    
    plt.suptitle("Model 1: Spatial Heterogeneity Comparison")
    plt.tight_layout()
    plt.savefig("Model_1_Spatial_Heterogeneity/comparison_model1_kde.png", dpi=200)
    print("Saved comparison plot to Model_1_Spatial_Heterogeneity/comparison_model1_kde.png")

def compare_model2():
    print("Comparing Model 2...")
    df_BF = pd.read_csv("Model_2_Temporal_Heterogeneity/BF_ucln_vec_post.csv")
    df_beast = load_beast_log("Model_2_Temporal_Heterogeneity/beast_model2_ucln.log")
    
    if df_beast is None:
        print("Model 2 BEAST output not found.")
        return
        
    # Map params:
    # BF: kappa, alpha, mu_c, sigma_c
    # BEAST: hky.kappa, gammaShape, ucld.mean, ucld.stdev
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    sns.kdeplot(df_BF['kappa'], ax=axes[0], label='BF (Native)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(df_beast['hky.kappa'], ax=axes[0], label='BEAST 2', color='green', linestyle='--')
    axes[0].set_title('Kappa (ts/tv)')
    
    sns.kdeplot(df_BF['alpha'], ax=axes[1], label='BF (Native)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(df_beast['gammaShape'], ax=axes[1], label='BEAST 2', color='green', linestyle='--')
    axes[1].set_title('Gamma Shape (Alpha)')
    
    # Actually, BEAST parameterizes ucld.mean normally (mean in real space) or log space. 
    # Let's plot ucld.mean vs exp(mu_c). Wait, in BF, branch lengths scaled by c_j = exp(mu_c + sigma_c * z). 
    # The mean in real space is exp(mu_c + sigma_c^2 / 2). Let's plot raw distributions since mu_c is log-mean.
    # Usually BEAST lognormal mean parameter M corresponds to the mode if meanInRealSpace is false, or real mean if true.
    # Let's just plot the ucld.mean and ucld.stdev directly.
    # Often BF mu_c corresponds to log(ucld.mean) - 0.5*sigma_c^2 in BEAST if parameterizing real mean.
    
    sns.kdeplot(df_BF['mu_c'], ax=axes[2], label='BF (Native) mu_c', color='blue', fill=True, alpha=0.3)
    # BEAST ucld.mean is usually 1.0 (fixed) for strict clock calibrations, if it is sampled it's the real mean. 
    # Let's log it if we want to compare to mu_c or leave. In primate.xml, ucld.mean is set to 1.0 and is NOT sampled (no operator for it). 
    # So we'll skip mu_c vs ucld.mean and just plot sigma_c.
    
    sns.kdeplot(df_BF['sigma_c'], ax=axes[3], label='BF (Native)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(df_beast['ucld.stdev'], ax=axes[3], label='BEAST 2', color='green', linestyle='--')
    axes[3].set_title('UCLN Sigma (Clock SD)')
    
    for ax in axes: ax.legend()
    
    plt.suptitle("Model 2: Temporal Heterogeneity (UCLN) Comparison")
    plt.tight_layout()
    plt.savefig("Model_2_Temporal_Heterogeneity/comparison_model2_kde.png", dpi=200)
    print("Saved comparison plot to Model_2_Temporal_Heterogeneity/comparison_model2_kde.png")

if __name__ == "__main__":
    compare_model1()
    compare_model2()
