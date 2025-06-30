import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import fori_loop
from functools import partial
import matplotlib.pyplot as plt
import numpy as onp

# For type hinting
from jax.typing import ArrayLike
from jax import Array
from typing import Optional, Union



class JAXKMeans:
    """
    K-Means clustering implemented in JAX with a Scikit-learn like API.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids
        to generate.
    n_iterations : int, default=100
        Maximum number of iterations of the k-means algorithm for a
        single run.
    random_state : int, optional
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.
    """
    def __init__(self, X: ArrayLike, n_clusters: int, n_iterations: int = 100, random_state: Optional[int] = None):

        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Attributes that will be set after fitting (ending with an underscore)
        self.centroids_: Optional[Array] = None
        self.labels_: Optional[Array] = None
        self.inertia_: Optional[Array] = None

        # Create a JAX random key
        if random_state is not None:
            self.key = random.PRNGKey(0)
        else:
            self.key = jax.random.PRNGKey(0)

        self.results = {}
        self.predictions = {}
        
        self.fit(X, n_clusters, n_iterations)

    def fit(self,X: ArrayLike, n_clusters: int, n_iterations: int = 100):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        X_jnp = jnp.asarray(X)
        n_samples, n_features = X_jnp.shape
    
        # 1. Initialize centroids by randomly picking k points from the data
        initial_centroids = random.choice(self.key , X_jnp, shape=(n_clusters,), replace=False)
    
        @jit
        def update_step(i, centroids):
            """
            Performs a single iteration of the K-means algorithm (Assignment + Update).
            This function is designed to be the body of `jax.lax.fori_loop`.
            """
            # --- Assignment step ---
            # Calculate squared distances from each point to each centroid
            # Using broadcasting for efficiency:
            # X_jnp shape:          (n_samples, 1, n_features)
            # centroids shape:      (1, n_clusters, n_features)
            # distances shape:      (n_samples, n_clusters)
            distances_sq = jnp.sum((X_jnp[:, None, :] - centroids[None, :, :])**2, axis=2)
    
            # Assign each point to the closest centroid
            labels = jnp.argmin(distances_sq, axis=1)
    
            # --- Update step ---
            # Create a one-hot encoding of the labels
            one_hot_labels = jax.nn.one_hot(labels, n_clusters) # shape (n_samples, n_clusters)
    
            # Calculate the sum of points for each cluster
            # (k, n_samples) @ (n_samples, n_features) -> (n_clusters, n_features)
            sums = jnp.dot(one_hot_labels.T, X_jnp)
    
            # Calculate the number of points in each cluster
            counts = jnp.sum(one_hot_labels, axis=0) # shape (n_clusters,)
    
            # Calculate new centroids, handling empty clusters
            # If a cluster is empty (count=0), its centroid remains unchanged
            new_centroids = jnp.where(
                counts[:, None] > 0,          # Condition
                sums / counts[:, None],       # Value if true
                centroids                     # Value if false
            )
            return new_centroids

        # 2. Run the main K-means loop for n_iterations
        final_centroids = fori_loop(0, n_iterations, update_step, initial_centroids)

        # 3. Calculate final labels and inertia for the output
        final_distances_sq = jnp.sum((X_jnp[:, None, :] - final_centroids[None, :, :])**2, axis=2)
        final_labels = jnp.argmin(final_distances_sq, axis=1)

        # Inertia is the sum of squared distances for the assigned clusters
        inertia = jnp.sum(jnp.min(final_distances_sq, axis=1))
       # Store the results in the instance
        self.centroids_ = final_centroids
        self.labels_ = final_labels
        self.inertia_ = inertia
        self.results={
            "centroids": final_centroids,
            "labels": final_labels,
            "inertia": inertia
        }

    def predict(self, X: ArrayLike) -> Array:
        """
        Predicts the closest cluster for each sample in X based on trained centroids.

        Args:
            centroids (ArrayLike): The cluster centroids, shape (k, n_features).
            X (ArrayLike): The data to predict, shape (n_samples, n_features).

        Returns:
            Array: The index of the cluster each sample belongs to, shape (n_samples,).
        """
        # A check to ensure fit has been called.
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        X_jnp = jnp.asarray(X)
        centroids_jnp = jnp.asarray(self.centroids_)

        distances_sq = jnp.sum((X_jnp[:, None, :] - self.centroids_[None, :, :])**2, axis=2)
        labels = jnp.argmin(distances_sq, axis=1)
        self.predictions = labels
        return labels

    def plot(self, X: ArrayLike, show_contours: bool = True, show_centroids: bool = True):
        """
        Generates an advanced plot of the clustering results, including data points,
        centroids, and decision boundary contours.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, 2)
            The data that was used for fitting. The plot is only generated for 2D data.
        show_contours : bool, default=True
            If True, calculates and displays the K-Means decision boundaries.
        show_centroids : bool, default=True
            If True, displays the final cluster centroids.
        """
        # --- 1. Pre-computation and Checks ---

        # Check if the model is fitted
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Model must be fitted before plotting. Call .fit() first.")

        # This visualization is only meaningful for 2D data
        if X.shape[1] != 2:
            print(f"Warning: Plotting is only supported for 2D data. Your data has {X.shape[1]} features.")
            return
            
        # Optional dependency check for seaborn
        try:
            import seaborn as sns
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            print("Warning: Seaborn not found. Using default Matplotlib styles. "
                  "Install with: pip install seaborn")

        # --- 2. Plot Setup ---
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#f0f0f0') 
        ax.set_facecolor('#f0f0f0')

        # --- 3. Dynamic Color Palette for Data Points ---
        
        # Use standard numpy for unique, as it's on the CPU-bound labels array
        unique_labels = onp.unique(self.labels_)
        n_clusters = len(unique_labels)
        palette = sns.color_palette("viridis", n_colors=n_clusters) if 'sns' in locals() else plt.cm.viridis

        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[l] for l in onp.array(self.labels_)]

        # Plot the data points
        ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=25, alpha=0.9,
                   edgecolor='white', linewidth=0.5, label='Data Points')

        # --- 4. Plot Centroids ---

        if show_centroids:
            # Plot with a distinctive style
            ax.scatter(self.centroids_[:, 0], self.centroids_[:, 1],
                       marker='X', s=250, c='black',
                       edgecolor='white', linewidth=1.5, label='Centroids')

        # --- 5. Calculate and Plot Decision Contours ---

        if show_contours:
            # Define the grid for the contour plot
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = onp.meshgrid(onp.arange(x_min, x_max, 0.02),
                                  onp.arange(y_min, y_max, 0.02))

            # Create a JIT-compiled function to score every point on the grid
            @jit
            def score_samples(points, centroids):
                # Calculate the distance to the *nearest* centroid
                distances_sq = jnp.sum((points[:, None, :] - centroids[None, :, :])**2, axis=2)
                min_dist_sq = jnp.min(distances_sq, axis=1)
                return min_dist_sq

            # Predict the "score" for each point in the meshgrid
            # onp.c_ concatenates the flattened grids into a list of points
            grid_points = onp.c_[xx.ravel(), yy.ravel()]
            Z = score_samples(jnp.asarray(grid_points), self.centroids_)
            Z = Z.reshape(xx.shape)

            # Plot the contour lines
            ax.contour(xx, yy, Z, levels=10, colors='dimgray',
                       linewidths=0.8, linestyles='--')

        # --- 6. Final Styling ---

        ax.set_title("K-Means Clustering with Decision Boundaries", fontsize=16)
        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)
        # Enforce a square aspect ratio for a correct view of distances
        ax.set_aspect('equal', adjustable='box') 
        
        plt.show()