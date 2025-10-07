# For type hinting
from jax.typing import ArrayLike
from jax import Array
from typing import Optional, Union
import jax.numpy as jnp
import matplotlib.pyplot as plt
from BI.ML.KMeans import JAXKMeans
import numpy as onp
class ml():
    """
    A stateful handler to manage and interact with ML models using a fluent API.

    This handler allows selecting, fitting, and predicting with models
    in a chained, stateful manner.

    Example:
        handler = ModelHandler()
        handler.KMEANS(X, n_clusters=3).fit()
        predictions = handler.predict(X_test)
        handler.plot()
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.X = None
        self.y = None
        self.results = {}
        self.predictions = None
        self.model_params = {}

    def fit(self, X: ArrayLike) -> Array:
        """
        Fits the model to the provided data.

        Parameters
        ----------
        X : ArrayLike
            The input data to fit the model.

        Returns
        -------
        Array
            The fitted model's results.
        """
        if self.model is None:
            raise RuntimeError("No model has been initialized. Call a model method first.")
        
        self.X = jnp.asarray(X)
        self.model.fit(self.X)
        self.results = self.model.results
        return self.results

    def predict(self, X: ArrayLike) -> Array:
        """
        Prediction method for the fitted model.

        Parameters
        ----------
        X : ArrayLike
            The input data to predict.

        Returns
        -------
        Array
            The predicted cluster labels.
        """
        if self.model is None:
            raise RuntimeError("No model has been fitted. Call fit() first.")
        
        return self.model.predict(X)

    def plot(self, X: ArrayLike):
        """
        Plots the model output.
        """
        if self.model is None or self.X is None:
            raise RuntimeError("No model has been fitted. Call fit() first.")
        
        self.model.plot(X)

    def KMEANS(self, X: ArrayLike, n_clusters: int, n_iterations: int = 100, random_state: Optional[int] = None):
        """
        Initializes a KMeans model with the given parameters.

        Parameters
        ----------
        X : ArrayLike
            The input data to fit the model.
        n_clusters : int
            The number of clusters to form.
        n_iterations : int, default=100
            The number of iterations for the K-means algorithm.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        self
            Returns the instance itself for method chaining.
        """
        self.X = jnp.asarray(X)
        self.model_name = 'KMEANS'
        self.model = JAXKMeans(X, n_clusters=n_clusters, n_iterations=n_iterations, random_state=random_state)
        self.results = self.model.results
        return self
