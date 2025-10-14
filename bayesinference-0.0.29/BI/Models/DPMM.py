import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as Dist
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt

# Import the CLASS, not a shared instance
from BI.Utils.np_dists import UnifiedDist

class DPMM:
    """
    Dirichlet Process Mixture Model.
    Encapsulates the model definition, prediction, and plotting logic.
    """
    def __init__(self):
        """
        Initializes the model class with its own distribution handler.
        """
        # This instance gets its own personal notebook of distributions.
        self.dist = UnifiedDist()

    def dpmm(self, data, T=10):
        """
        The NumPyro model definition.
        Making it __call__ allows the instance to be used like a function.
        """
        D = data.shape[1]

        # 1) stick-breaking weights
        # CRITICAL: Use self.dist to call the distribution methods
        alpha = self.dist.gamma(1.0, 15.0, name='alpha')
        
        # Note: alpha must be defined before beta uses it.
        # I've fixed the order of arguments to match standard signature.
        beta = self.dist.beta(concentration1=1, concentration0=alpha, name='beta', shape=(T-1,))
        
        w = numpyro.deterministic("w", Dist.transforms.StickBreakingTransform()(beta))

        # 2) component parameters
        data_mean = jnp.mean(data, axis=0)
        with numpyro.plate("components", T):
            mu = self.dist.multivariate_normal(
                loc=data_mean, 
                covariance_matrix=100.0 * jnp.eye(D), 
                name='mu'
            ) # shape (T, D)
            
            sigma = self.dist.half_cauchy(
                1, shape=(D,), event=1, name='sigma'
            ) # shape (T, D)
            
            Lcorr = self.dist.lkj_cholesky(
                dimension=D, concentration=1.0, name='Lcorr'
            ) # shape (T, D, D)

            scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

        # 3) marginal mixture over obs
        self.dist.mixture_same_family(
            mixing_distribution=self.dist.categorical_probs(w, name='cat', create_obj=True),
            component_distribution=self.dist.multivariate_normal(
                loc=mu, scale_tril=scale_tril, name='mvn', create_obj=True
            ),
            name="obs",
            obs=data
        )

    def predict(self, data, sampler):
        """
        Predicts the DPMM density contours based on posterior samples and final labels.
        """
        # 1. Calculate posterior mean of all model parameters
        posterior_samples = sampler.get_samples()
        w_samps = posterior_samples['w']
        mu_samps = posterior_samples['mu']
        Lcorr_samps = posterior_samples['Lcorr']
        sigma_samps = posterior_samples['sigma']

        post_mean_w = jnp.mean(w_samps, axis=0)
        post_mean_mu = jnp.mean(mu_samps, axis=0)
        post_mean_sigma = jnp.mean(sigma_samps, axis=0)
        post_mean_Lcorr = jnp.mean(Lcorr_samps, axis=0)

        # Reconstruct the full covariance matrices
        post_mean_scale_tril = post_mean_sigma[..., None] * post_mean_Lcorr
        post_mean_cov = post_mean_scale_tril @ jnp.transpose(post_mean_scale_tril, (0, 2, 1))

        # Co-clustering logic to get final_labels
        # Helper function defined inside predict to keep it self-contained
        def get_cluster_probs(data_pt, w, mu, sigma, Lcorr):
            scale_tril = sigma[..., None] * Lcorr
            # Using NumPyro's Dist directly here for calculation is fine
            log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data_pt)
            log_probs = jnp.log(w) + log_liks
            norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
            return norm_probs

        # Vectorize over samples
        cluster_probs = jax.vmap(get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
            data, w_samps, mu_samps, sigma_samps, Lcorr_samps
        )
        
        similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)
        
        # Need to convert to NumPy for SciPy functions
        similarity_matrix_np = np.array(similarity_matrix)
        distance_matrix = 1 - similarity_matrix_np
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, a_min=0.0, a_max=None)
        
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, 'average')
        distance_threshold = 0.5
        final_labels = fcluster(Z, t=distance_threshold, criterion='distance')

        num_found_clusters = len(np.unique(final_labels))
        print(f"Model found {num_found_clusters} clusters.")

        return post_mean_w, post_mean_mu, post_mean_cov, final_labels

    def plot(self, data, sampler, figsize=(10, 8), point_size=10):
        """
        Plots the data points colored by cluster and the density contours.
        """
        # Call the internal predict method
        post_mean_w, post_mean_mu, post_mean_cov, final_labels = self.predict(data, sampler)

        # 2. Set up a grid of points to evaluate the GMM density
        # Convert to numpy for matplotlib
        data_np = np.array(data)
        x_min, x_max = data_np[:, 0].min() - 2, data_np[:, 0].max() + 2
        y_min, y_max = data_np[:, 1].min() - 2, data_np[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                             np.linspace(y_min, y_max, 150))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 3. Calculate the PDF of the GMM on the grid (using SciPy)
        num_components = post_mean_mu.shape[0]
        gmm_pdf = np.zeros(grid_points.shape[0])

        # Convert JAX arrays to NumPy for SciPy's multivariate_normal
        pm_w = np.array(post_mean_w)
        pm_mu = np.array(post_mean_mu)
        pm_cov = np.array(post_mean_cov)

        for k in range(num_components):
            weight = pm_w[k]
            mean = pm_mu[k]
            cov = pm_cov[k]
            component_pdf = multivariate_normal(mean=mean, cov=cov).pdf(grid_points)
            gmm_pdf += weight * component_pdf

        Z = gmm_pdf.reshape(xx.shape)

        # 4. Create the plot
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor('#f0f0f0')
            ax.set_facecolor('#f0f0f0')

            # Dynamically create a color palette based on the number of clusters found
            unique_labels = np.unique(final_labels)
            n_clusters = len(unique_labels)
            palette = sns.color_palette("viridis", n_colors=n_clusters)

            color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
            point_colors = [color_map[l] for l in final_labels]

            ax.scatter(data_np[:, 0], data_np[:, 1], c=point_colors, s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

            contour_color = 'navy'
            contour = ax.contour(xx, yy, Z, levels=10, colors=contour_color, linewidths=0.8)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

            ax.set_title("DPMM Probability Density Contours", fontsize=16)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.grid(True, linestyle=':', color='gray', alpha=0.6)

        plt.show()
