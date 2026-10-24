import sys
from unittest.mock import MagicMock

# Mock IPython for environments where it is not installed
mock_ipython = MagicMock()
sys.modules["IPython"] = mock_ipython
sys.modules["IPython.display"] = mock_ipython

import numpy as np
import jax.numpy as jnp
from BayesForge import bf
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

import os

def run_similarity_test():
    log_path = os.path.join(os.path.dirname(__file__), "log.txt")
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log_msg(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log_file.write("=== MGM (GMM) & DPMM Similarity Test Results ===\n")
        print("=== MGM (GMM) & DPMM Similarity Test ===")
        
        # 1. Data Simulation
        n_samples = 500
        n_centers = 4
        data, true_labels = make_blobs(
            n_samples=n_samples, centers=n_centers, cluster_std=0.8,
            center_box=(-10, 10), random_state=101
        )
        print(f"Generated {n_samples} samples with {n_centers} centers.")

        # Initialize BF
        m = bf(platform='cpu')

        # --- DPMM Section ---
        print("\n--- Testing DPMM Similarity ---")
        T = 11
        m.data_on_model = dict(data=data, T=T)
        
        # BF DPMM Fit
        print("Fitting BF DPMM...")
        m.fit(m.models.dpmm, num_chains=1, num_samples=1000, num_warmup=1000)
        
        # Extract labels from BayesForge DPMM
        _, _, _, _, BF_dpmm_labels = m.models.dpmm.predict(data, m.sampler)
        n_clusters_BF = len(np.unique(BF_dpmm_labels))
        log_msg(f"BF DPMM found {n_clusters_BF} clusters.")

        # Sklearn DPMM (BayesianGaussianMixture)
        print("Fitting Sklearn BayesianGaussianMixture...")
        bgm = BayesianGaussianMixture(
            n_components=T, weight_concentration_prior_type='dirichlet_process',
            random_state=101
        )
        sklearn_dpmm_labels = bgm.fit_predict(data)
        n_clusters_sklearn = len(np.unique(sklearn_dpmm_labels))
        log_msg(f"Sklearn DPMM found {n_clusters_sklearn} clusters.")

        # Compare labels
        ari_dpmm = adjusted_rand_score(BF_dpmm_labels, sklearn_dpmm_labels)
        log_msg(f"DPMM Adjusted Rand Index (ARI): {ari_dpmm:.4f}")

        # --- GMM Section ---
        print("\n--- Testing GMM Similarity ---")
        K = n_centers
        # Use K-means to initialize means for BF GMM
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=K, random_state=101).fit(data)
        initial_means = kmeans.cluster_centers_

        # BF GMM Fit
        print("Fitting BF GMM...")
        # Reset model data for GMM
        m.data_on_model = dict(data=data, K=K, initial_means=initial_means)
        m.fit(m.models.available['gmm'], num_chains=1, num_samples=1000, num_warmup=1000)
        
        # Extract labels from BayesForge GMM
        from BayesForge.Models.GMM import predict_gmm
        _, _, _, BF_gmm_labels = predict_gmm(data, m.sampler)
        
        # Sklearn GMM
        print("Fitting Sklearn GaussianMixture...")
        gmm = GaussianMixture(n_components=K, random_state=101)
        sklearn_gmm_labels = gmm.fit_predict(data)

        # Compare labels
        ari_gmm = adjusted_rand_score(BF_gmm_labels, sklearn_gmm_labels)
        log_msg(f"GMM Adjusted Rand Index (ARI): {ari_gmm:.4f}")

        # --- Summary & Assertions ---
        log_msg("\n=== Test Result Summary ===")
        log_msg(f"DPMM ARI: {ari_dpmm:.4f}")
        log_msg(f"GMM ARI:  {ari_gmm:.4f}")

        if ari_dpmm > 0.8 and ari_gmm > 0.8:
            log_msg("SUCCESS: High similarity detected between BF and Sklearn!")
        else:
            log_msg("WARNING: Low similarity detected. Please check model hyperparameters.")

if __name__ == "__main__":
    run_similarity_test()
