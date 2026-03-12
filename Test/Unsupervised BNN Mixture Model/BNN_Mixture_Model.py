#%%
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from BI import bi
import jax.numpy as jnp
import jax
import numpyro
import jax.scipy.stats as stats
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import matplotlib.patches as mpl_patches
import matplotlib.transforms as mpl_transforms

# Setup device------------------------------------------------
m = bi(platform='cpu')

# Generate Synthetic Data ------------------------------------
# 4 clusters
data, true_labels = make_blobs(
    n_samples=500, centers=9, cluster_std=0.8,
    center_box=(-10,10), random_state=101
)
N, D = data.shape

print(f"Data shape: {data.shape}")
print(f"True labels shape: {true_labels.shape}")

plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('Synthetic Data (True Labels)')
plt.savefig('synthetic_data.png')
plt.close()

# DPMM Baseline-----------------------------------------------
print("\n--- Running DPMM Baseline ---")
m.data_on_model = dict(data=data, T=11)
print("Fitting DPMM..."); m.fit(m.models.dpmm, num_chains=1)

dpmm_clusters = m.models.dpmm.proportion_of_data_assigned_to_cluster()

print("\n--- Extracting DPMM Cluster Assignments ---")
w_samps_dpmm, mu_samps_dpmm, sigma_samps_dpmm, Lcorr_samps_dpmm, dpmm_labels = m.models.dpmm.predict(data, m.sampler)
from sklearn.metrics import adjusted_rand_score
dpmm_ari = adjusted_rand_score(true_labels, dpmm_labels)
print(f"DPMM Adjusted Rand Index: {dpmm_ari:.4f}")

# BNN Mixture Model---------------------------------------------
print("\n--- Running BNN Mixture Model ---")
m.data_on_model = dict(data=data, K=11, D_H1=10)

print("Fitting BNN..."); m.fit(m.models.bnnc, num_chains=1)

print("\n--- Extracting BNN Mixture Model Cluster Assignments ---")
theta_samps_bnn, mu_samps_bnn, sigma_samps_bnn, bnn_labels = m.models.bnnc.predict(data, m.sampler)

bnn_ari = adjusted_rand_score(true_labels, bnn_labels)
print(f"BNN Mixture Model Adjusted Rand Index: {bnn_ari:.4f}")

# Sklearn DPMM & GMM Baseline-----------------------------------
print("\n--- Running Sklearn Baselines ---")
sk_dpmm = BayesianGaussianMixture(
    n_components=11, weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.05, max_iter=1000, random_state=42
)
sk_dpmm_labels = sk_dpmm.fit_predict(data)
sk_dpmm_ari = adjusted_rand_score(true_labels, sk_dpmm_labels)
print(f"Sklearn DPMM Adjusted Rand Index: {sk_dpmm_ari:.4f}")

sk_gmm = GaussianMixture(n_components=4, max_iter=1000, random_state=42)
sk_gmm_labels = sk_gmm.fit_predict(data)
sk_gmm_ari = adjusted_rand_score(true_labels, sk_gmm_labels)
print(f"Sklearn GMM Adjusted Rand Index: {sk_gmm_ari:.4f}")

# Plotting Function with Ellipses
def draw_stylized_ellipses(ax, data, labels, title, cmap_name='viridis'):
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    
    # Plot points
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap_name, s=15, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Draw 2-Sigma confidence ellipses for each discovered cluster
    for i, label in enumerate(unique_labels):
        cluster_data = data[labels == label]
        if len(cluster_data) < 3:
            continue # Not enough points for covariance
            
        mean = np.mean(cluster_data, axis=0)
        cov = np.cov(cluster_data, rowvar=False)
        
        # Eigen decomposition to find ellipse properties
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        
        ellipse = mpl_patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                                      facecolor=colors[i], alpha=0.2, edgecolor=colors[i], linewidth=2)
        
        # Scale and translate the ellipse
        scale_x = np.sqrt(cov[0, 0]) * 2 # 2 standard deviations
        scale_y = np.sqrt(cov[1, 1]) * 2
        
        transf = mpl_transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean[0], mean[1])
            
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        
        # Mark center
        ax.scatter(mean[0], mean[1], marker='X', color='black', s=50, edgecolor='white')

    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

# Final Comparison Plot
fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.suptitle('Mixture Model Clustering Comparison (Empirical 2σ Confidence Ellipses)', fontsize=16)

draw_stylized_ellipses(axes[0], data, true_labels, 'True Labels')
draw_stylized_ellipses(axes[1], data, sk_gmm_labels, f'Sklearn GMM (ARI: {sk_gmm_ari:.3f})')
draw_stylized_ellipses(axes[2], data, sk_dpmm_labels, f'Sklearn DPMM (ARI: {sk_dpmm_ari:.3f})')
draw_stylized_ellipses(axes[3], data, dpmm_labels, f'BI DPMM (ARI: {dpmm_ari:.3f})')
draw_stylized_ellipses(axes[4], data, bnn_labels, f'BI BNNC (ARI: {bnn_ari:.3f})')

plt.tight_layout()
plt.savefig('cluster_comparison.png', dpi=150)
plt.close()

print("\n--- Process Complete ---")

# %%
# -----------------------------------------------------------------------------
# Systematic Evaluation (100 Iterations)
# -----------------------------------------------------------------------------
import pandas as pd
import time

print("\n\n" + "="*60)
print("STARTING 100-ITERATION SYSTEMATIC EVALUATION")
print("="*60)

n_iters = 100
results = []

for i in range(n_iters):
    # Randomize dataset parameters
    n_samples = np.random.randint(200, 800)
    true_k = np.random.randint(3, 11)
    cluster_std = np.random.uniform(0.5, 2.0)
    
    print(f"\nIteration {i+1}/{n_iters} | N={n_samples}, True K={true_k}, Std={cluster_std:.2f}")
    
    X_sim, y_sim = make_blobs(n_samples=n_samples, centers=true_k, cluster_std=cluster_std, center_box=(-15, 15))
    
    # 1. Sklearn GMM
    print("\nFitting Sklearn GMM...")
    sk_gmm = GaussianMixture(n_components=true_k, max_iter=500, random_state=i)
    preds_sk_gmm = sk_gmm.fit_predict(X_sim)
    ari_sk_gmm = adjusted_rand_score(y_sim, preds_sk_gmm)
    n_sk_gmm = len(np.unique(preds_sk_gmm))
    
    # 2. Sklearn DPMM
    print("\nFitting Sklearn DPMM...")
    sk_dpmm = BayesianGaussianMixture(n_components=12, weight_concentration_prior=0.05, max_iter=500, random_state=i)
    preds_sk_dpmm = sk_dpmm.fit_predict(X_sim)
    ari_sk_dpmm = adjusted_rand_score(y_sim, preds_sk_dpmm)
    n_sk_dpmm = len(np.unique(preds_sk_dpmm))
    
    # 3. BI DPMM
    print("\nFitting BI DPMM...")
    try:
        m.data_on_model = dict(data=X_sim, T=12, empirical_bayes=True)
        m.fit(m.models.dpmm, num_chains=1, num_samples=300, num_warmup=300, progress_bar=False)
        _, _, _, _, preds_bi_dpmm = m.models.dpmm.predict(X_sim, m.sampler)
        ari_bi_dpmm = adjusted_rand_score(y_sim, preds_bi_dpmm)
        n_bi_dpmm = len(np.unique(preds_bi_dpmm))
    except Exception as e:
        print(f"BI DPMM failed: {e}")
        ari_bi_dpmm = np.nan
        n_bi_dpmm = np.nan
        
    # 4. BI BNNC
    print("\nFitting BI BNNC...")
    try:
        m.data_on_model = dict(data=X_sim, K=12, D_H1=10, empirical_bayes=True)
        m.fit(m.models.bnnc, num_chains=1, num_samples=300, num_warmup=300, progress_bar=False)
        _, _, _, preds_bi_bnnc = m.models.bnnc.predict(X_sim, m.sampler)
        ari_bi_bnnc = adjusted_rand_score(y_sim, preds_bi_bnnc)
        n_bi_bnnc = len(np.unique(preds_bi_bnnc))
    except Exception as e:
        print(f"BI BNNC failed: {e}")
        ari_bi_bnnc = np.nan
        n_bi_bnnc = np.nan

    iteration_res = {
        'Iteration': i + 1,
        'N_Samples': n_samples,
        'True_K': true_k,
        'Cluster_Std': cluster_std,
        'ARI_Sklearn_GMM': ari_sk_gmm,
        'N_Sklearn_GMM': n_sk_gmm,
        'ARI_Sklearn_DPMM': ari_sk_dpmm,
        'N_Sklearn_DPMM': n_sk_dpmm,
        'ARI_BI_DPMM': ari_bi_dpmm,
        'N_BI_DPMM': n_bi_dpmm,
        'ARI_BI_BNNC': ari_bi_bnnc,
        'N_BI_BNNC': n_bi_bnnc
    }
    results.append(iteration_res)

# Save and summarize results
df_results = pd.DataFrame(results)
df_results.to_csv("simulation_100_results.csv", index=False)

print("\n" + "="*60)
print("EVALUATION SUMMARY (Means)")
print("="*60)
print(df_results[['ARI_Sklearn_GMM', 'ARI_Sklearn_DPMM', 'ARI_BI_DPMM', 'ARI_BI_BNNC']].mean().to_string())
print("-" * 60)
print(df_results[['N_Sklearn_GMM', 'N_Sklearn_DPMM', 'N_BI_DPMM', 'N_BI_BNNC']].mean().to_string())

# Optional Error/Difference metric against True_K
df_results['Err_Sklearn_GMM'] = np.abs(df_results['N_Sklearn_GMM'] - df_results['True_K'])
df_results['Err_Sklearn_DPMM'] = np.abs(df_results['N_Sklearn_DPMM'] - df_results['True_K'])
df_results['Err_BI_DPMM'] = np.abs(df_results['N_BI_DPMM'] - df_results['True_K'])
df_results['Err_BI_BNNC'] = np.abs(df_results['N_BI_BNNC'] - df_results['True_K'])

print("-" * 60)
print("Average Absolute True_K Error (Lower is better):")
print(df_results[['Err_Sklearn_GMM', 'Err_Sklearn_DPMM', 'Err_BI_DPMM', 'Err_BI_BNNC']].mean().to_string())
print("="*60)

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df_results = pd.read_csv("simulation_100_results.csv")
# Visualizing results of the systematic evaluation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Adjusted Rand Index (ARI) Distribution
ari_cols = [ 'ARI_Sklearn_DPMM', 'ARI_BI_DPMM', 'ARI_BI_BNNC']
sns.boxplot(data=df_results[ari_cols], ax=axes[0], palette='viridis')
axes[0].set_title('Clustering Performance (ARI)', fontsize=14)
axes[0].set_xticklabels(['Sk GMM', 'Sk DPMM', 'BI DPMM', 'BI BNNC'])
axes[0].set_ylabel('Adjusted Rand Index')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Plot 2: Absolute Error in Cluster Count (K)
err_cols = ['N_Sklearn_DPMM', 'N_BI_DPMM', 'N_BI_BNNC']
sns.boxplot(data=df_results[err_cols], ax=axes[1], palette='magma')
axes[1].set_title('Cluster Count Accuracy (|Predicted K - True K|)', fontsize=14)
axes[1].set_xticklabels(['Sk GMM', 'Sk DPMM', 'BI DPMM', 'BI BNNC'])
axes[1].set_ylabel('Absolute Error')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.png', dpi=150)
plt.show()
