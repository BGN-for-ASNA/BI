import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import fori_loop
from functools import partial
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.utils.validation import check_is_fitted

# For type hinting
from jax.typing import ArrayLike
from jax import Array
from typing import Optional, Union
import time



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
    def __init__(self, n_clusters: int, n_iterations: int = 100, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Attributes that will be set after fitting (ending with an underscore)
        self.centroids_: Optional[Array] = None
        self.labels_: Optional[Array] = None
        self.inertia_: Optional[Array] = None

    def fit(self, X: ArrayLike, y: None = None):
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
        
        # Set up the JAX random key
        if self.random_state is None:
            # Use a non-deterministic seed if no state is provided
            seed = int(time.time())
        else:
            seed = self.random_state
        key = random.PRNGKey(seed)

        # Run the core JAX implementation
        centroids, labels, inertia = _kmeans_jax_impl(
            key, X_jnp, self.n_clusters, self.n_iterations
        )
        
        # Store the results as instance attributes
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self

    def predict(self, X: ArrayLike) -> Array:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : Array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # Check if fit has been called
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        X_jnp = jnp.asarray(X)
        return _predict_jax_impl(self.centroids_, X_jnp)

    def fit_predict(self, X: ArrayLike, y: None = None) -> Array:
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X), but more efficient.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_

    def plot_clusters(self, X: ArrayLike, show_centroids: bool = True):
        """
        Generates a 2D scatter plot of the data colored by cluster assignment.
        
        This is a helper method for visualization and requires matplotlib.
        It will only work for 2D data (n_features=2).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, 2)
            The data to plot. Usually the same data used for fitting.
        show_centroids : bool, default=True
            Whether to plot the final centroids on the graph.
        """
        if self.labels_ is None or self.centroids_ is None:
             raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data (n_features=2).")

        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis', s=50, alpha=0.7)
        
        if show_centroids:
            plt.scatter(
                self.centroids_[:, 0], self.centroids_[:, 1], 
                c='red', marker='X', s=200, edgecolor='black', label='Centroids'
            )
            plt.legend()
            
        plt.title('K-Means Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()