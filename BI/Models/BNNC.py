import jax.numpy as jnp
import jax
import numpyro
import jax.scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class bnnc:
    def __init__(self, parent):
        """
        Bayesian Neural Network Clustering (BNNC) Class.
        
        Args:
            parent: The BI parent instance.
        """
        self.__name__ = 'bnnc' 
        self.parent = parent

    def __call__(self, data, K=11, D_H1=10, empirical_bayes=False):
        """
        Makes the class instance callable. 
        Redirects the call to self.model().
        """
        return self.model(data, K=K, D_H1=D_H1, empirical_bayes=empirical_bayes)

    def model(self, data, K=11, D_H1=10, empirical_bayes=False):
        """
        BNNC marginal mixture model definition.
        """
        print("⚠️This function is still in development. Use it with caution.⚠️")
        m = self.parent
        N, D_X = data.shape
        
        if empirical_bayes:
            # Data-driven priors
            data_mean = jnp.mean(data, axis=0)
            data_std = jnp.std(data, axis=0) * 2.0
            sigma_base = data_std
            alpha_bnnc = 1.0 / K
        else:
            # Fixed uninformative priors
            data_mean = jnp.zeros(D_X)
            data_std = jnp.ones(D_X) * 5.0
            sigma_base = 1.0
            alpha_bnnc = 0.05
        
        # --- BNN Gating Network ---
        w1 = m.bnn.layer_linear(
            data, 
            dist=m.dist.normal(0, 1, name='w1_weight', shape=(D_X, D_H1)),
            activation='tanh'
        )
        
        w2 = m.bnn.layer_linear(
            w1,
            dist=m.dist.normal(0, 0.05, name='w2_weight', shape=(D_H1, K))
        )
        
        # Global parsimony prior: Dirichlet prior on global cluster weights
        # Alpha < 1 acts as a sparse prior to suppress inactive clusters
        pi = numpyro.sample('global_pi', numpyro.distributions.Dirichlet(jnp.ones(K) * alpha_bnnc))
        
        logits = w2 + jnp.log(pi + 1e-10)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        theta = jnp.exp(log_p)
        numpyro.deterministic('theta', theta)
        
        # --- Mixture Components ---
        mu = m.dist.normal(data_mean, data_std, name='mu', shape=(K, D_X))
        
        if empirical_bayes:
            sigma = m.dist.half_normal(sigma_base, name='sigma', shape=(K, D_X))
        else:
            sigma = m.dist.exponential(sigma_base, name='sigma', shape=(K, D_X))
        
        # --- Marginalized Gaussian Mixture ---
        data_exp = jnp.expand_dims(data, axis=1) # (N, 1, D_X)
        mu_exp = jnp.expand_dims(mu, axis=0)     # (1, K, D_X)
        sigma_exp = jnp.expand_dims(sigma, axis=0) # (1, K, D_X)
        
        log_pdf_clusters = jnp.sum(stats.norm.logpdf(data_exp, loc=mu_exp, scale=sigma_exp), axis=-1) # (N, K)
        weighted_log_pdf = log_p + log_pdf_clusters # (N, K)
        
        total_log_likelihood = jax.scipy.special.logsumexp(weighted_log_pdf, axis=-1)
        numpyro.factor('mixture_likelihood', jnp.sum(total_log_likelihood))

    def get_cluster_probs(self, data_pts, theta_prob, mu_p, sigma_p):
        X_e = jnp.expand_dims(data_pts, axis=1) # (N, 1, D_X)
        mu_e = jnp.expand_dims(mu_p, axis=0) # (1, K, D_X)
        sig_e = jnp.expand_dims(sigma_p, axis=0) # (1, K, D_X)
        
        log_liks = jnp.sum(stats.norm.logpdf(X_e, loc=mu_e, scale=sig_e), axis=-1)
        log_probs = jnp.log(theta_prob + 1e-8) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs

    def predict(self, data, sampler):
        """
        Performs Consensus Clustering.
        """
        print("⚠️This function is still in development. Use it with caution.⚠️")
        posterior_samples = sampler.get_samples(group_by_chain=False)

        theta_samps = posterior_samples['theta']          
        mu_samps = posterior_samples['mu']        
        sigma_samps = posterior_samples['sigma']  

        cluster_probs = jax.vmap(self.get_cluster_probs, in_axes=(None, 0, 0, 0))(
            data, theta_samps, mu_samps, sigma_samps
        )

        similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)

        distance_matrix = 1 - similarity_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)
        distance_matrix = jnp.clip(distance_matrix, a_min=0.0, a_max=None)
        
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, 'average')
        
        final_labels = fcluster(Z, t=0.5, criterion='distance')
        print(f"Model found {len(np.unique(final_labels))} clusters.")

        return theta_samps, mu_samps, sigma_samps, final_labels

    def plot(self, data, sampler, figsize=(10, 8), point_size=30):
        """
        Plots the Clustering results.
        """
        print("⚠️This function is still in development. Use it with caution.⚠️")
        theta_samps, mu_samps, sigma_samps, final_labels = self.predict(data, sampler)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#f0f0f0') 
        ax.set_facecolor('#f0f0f0')

        unique_labels = np.unique(final_labels)
        n_clusters = len(unique_labels)
        palette = sns.color_palette("viridis", n_colors=n_clusters) 
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[l] for l in final_labels]

        ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

        ax.set_title("BNNC Clustering Assignments", fontsize=16)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)

        plt.show()