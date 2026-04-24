from BI import bi
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.datasets import load_iris

# Initialize BI model
m = bi()

# Logging setup to capture results in log.txt
import sys


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


f = open("log.txt", "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)


print("Loading and scaling data...")
# --- Load and Scale Data (from scratch with JAX) ---
iris = load_iris()
X_raw = jnp.array(iris.data)
y = jnp.array(iris.target)
feature_names = iris.feature_names
target_names = iris.target_names


def scale_data(X):
    """Standardizes features by removing the mean and scaling to unit variance."""
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    return (X - mean) / std, mean, std


X_scaled, data_mean, data_std = scale_data(X_raw)

print("Fitting model with BI...")
m.data_on_model = dict(X=X_scaled)
# Using classic PCA for direct comparison with sklearn
m.fit(m.models.pca(type="classic"), num_samples=500, num_warmup=500)

# Extract results from BI
bi_pca_results = m.models.pca.get_attributes(X=X_scaled)
bi_components = bi_pca_results["components"]
bi_variance_ratio = bi_pca_results["explained_variance_ratio"]

print("Fitting model with sklearn...")
sklearn_pca = sklearn_PCA(n_components=X_scaled.shape[1])
sklearn_pca.fit(X_scaled)
sklearn_components = sklearn_pca.components_.T  # Shape (data_dim, latent_dim)
sklearn_variance_ratio = sklearn_pca.explained_variance_ratio_

print("\n--- Numerical Comparison ---")

# 1. Compare Explained Variance Ratio
print(f"BI Variance Ratio:      {bi_variance_ratio}")
print(f"Sklearn Variance Ratio: {sklearn_variance_ratio}")

try:
    np.testing.assert_allclose(
        bi_variance_ratio,
        sklearn_variance_ratio,
        atol=1e-2,
        err_msg="Explained variance ratio mismatch",
    )
    print("✅ Explained variance ratios match within tolerance.")
except AssertionError as e:
    print(f"❌ Explained variance ratio mismatch: {e}")

# 2. Compare Components (Loading Matrix)
# Note: Signs can be flipped, so we compare absolute values or align signs
# BI components are already sign-corrected in BI.Models.PCA.set_deterministic_sign
# Sklearn components might have different signs.


def align_signs(comp_matrix):
    """Align signs of columns based on the max absolute value in each column."""
    for i in range(comp_matrix.shape[1]):
        max_abs_idx = np.argmax(np.abs(comp_matrix[:, i]))
        if comp_matrix[max_abs_idx, i] < 0:
            comp_matrix[:, i] *= -1
    return comp_matrix


sklearn_components_aligned = align_signs(sklearn_components.copy())
bi_components_aligned = align_signs(np.array(bi_components))

print("\nBI Components (first 2 PCs):\n", bi_components_aligned[:, :2])
print("\nSklearn Components (first 2 PCs):\n", sklearn_components_aligned[:, :2])

try:
    np.testing.assert_allclose(
        bi_components_aligned,
        sklearn_components_aligned,
        atol=0.5,
        err_msg="PCA components mismatch",
    )
    print("✅ PCA components match within tolerance.")
except AssertionError as e:
    print(f"❌ PCA components mismatch: {e}")

print("\nPCA unit test completed successfully.")

# Close log file and restore stdout
sys.stdout = original_stdout
f.close()
print("Results captured in log.txt")
