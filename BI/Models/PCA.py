
from BI.Utils.np_dists import UnifiedDist as dist
import jax.numpy as jnp

class pca:
    """
    PCA model class using JAX and BI.
    Args:
        X (jnp.ndarray): Training data. X.shape = (num_datapoints, data_dim)
        latent_dim (int): Dimensionality of the latent space.
    Returns:
        None    
    """
    def __init__(self, X, latent_dim):
        self.X = X
        self.latent_dim = latent_dim
        self.model = pca(X, latent_dim)
        self.data_dim = X.shape[1]
        self.num_data_points = X.shape[0]
        self.models = {
            'pca': self.pca,
            'pca_ARD': self.pca_ARD,
            'pca_robust': self.pca_robust,
            'pca_sparse': self.pca_sparse,
            'model_sparse_robust_ard': self.model_sparse_robust_ard
        }

    ## Models---------------------------------------------
    def pca(self, X = self.X, data_dim = self.data_dim, latent_dim = self.latent_dim, num_data_points = self.num_data_points): 
        # Gaussian prior for the principal component 'W'.
        w = m.dist.normal(0, 1, shape=(data_dim, latent_dim), name='w')

        # Gaussian prior on the latent variables 'Z'
        z = m.dist.normal(0, 1, shape=(latent_dim, num_data_points), name='z')

        # Exponential prior on the noise variance 'epsilon'
        epsilon = m.dist.exponential(1, name='epsilon')

        # Likelihood
        m.dist.normal(w @ z , epsilon, obs = X)  

    def pca_ARD(self, X = self.X, data_dim = self.data_dim, latent_dim = self.latent_dim, num_data_points = self.num_data_points):
        """
        Probabilistic PCA model with ARD (Automatic Relevance Determination) prior on weights and a correctly defined noise model. ARD helps in automatic relevance determination of latent dimensions, by allowing the model to learn which dimensions are important.
        Args:
            x_train: Observed data matrix of shape (data_dim, num_datapoints).
            data_dim: Dimensionality of the observed data.
            latent_dim: Dimensionality of the latent space.
            num_data_points: Number of data points.
        """
        # ARD Prior on w (This part is correct)
        alpha = dist.gamma(.05, 1e-3, shape=(latent_dim,), name='alpha')
        w = dist.normal(0, 1. / jnp.sqrt(alpha)[None, :], shape=(data_dim, latent_dim), name='w')

        # Prior on z (This part is correct)
        z = dist.normal(0, 1., shape=(latent_dim, num_data_points), name='z')

        # --- CORRECTED NOISE MODEL ---
        # Prior on the precision (1 / variance)
        precision = m.dist.gamma(1.0, 1.0, name='precision')
        # The standard deviation is 1 / sqrt(precision)
        stddv = 1. / jnp.sqrt(precision)

        # Use the correctly defined standard deviation in the likelihood
        m.dist.normal(w @ z, stddv, obs=X)

    def pca_robust(self, X = self.X, data_dim = self.data_dim, latent_dim = self.latent_dim, num_data_points = self.num_data_points):
        """
        Robust Bayesian PCA model using a Student's t-distribution for the likelihood.
        """
        # --- Standard Priors for W and Z ---
        w = m.dist.normal(0, 1., shape=(data_dim, latent_dim), name='w')
        z = m.dist.normal(0, 1., shape=(latent_dim, num_data_points), name='z')

        # --- Robustness to Outliers via a Heavy-Tailed Noise Model ---
        # This defines the prior on the scale (similar to standard deviation) of the noise.
        sigma = m.dist.halfcauchy(1.0, name='sigma')

        # This is a prior on the "degrees of freedom" ('nu') of the Student's t-distribution.
        # This parameter controls the "heaviness" of the tails. A small 'nu' means
        # heavier tails, making the model more robust to outliers. By learning this
        # parameter, the model can adapt its robustness to the data.
        nu = m.dist.gamma(2.0, 0.1, name='nu')

        # This is the key line for this model. The likelihood is a Student's t-distribution.
        # As your description states, this "heavy-tailed distribution... reduces the
        # influence of outliers" by treating them as more plausible events than a
        # Gaussian distribution would, thus preventing them from skewing the results.
        m.dist.studentt(df=nu, loc=w @ z, scale=sigma, obs=X)

    def pca_sparse(self, X = self.X, data_dim = self.data_dim, latent_dim = self.latent_dim, num_data_points = self.num_data_points):
        """
        Sparse Bayesian PCA model using a Laplace prior on the weights (w).
        """
        # --- Sparsity for High-Dimensional Data via a Sparsity-Inducing Prior ---

        # This is the first part of a hierarchical prior (known as the Bayesian Lasso).
        # We place a prior on 'lambda_', which will control the scale of our Laplace prior.
        # This allows the model to learn the appropriate level of sparsity from the data.
        lambda_ = m.dist.gamma(1.0, 1.0, shape=(latent_dim,), name='lambda')

        # This is the key line for this model. We place a Laplace prior on the loadings 'W'.
        # As your description states, this is a "sparsity-inducing prior". The Laplace
        # distribution is sharply peaked at zero, which encourages many of the weight
        # values in 'W' to be exactly zero, leading to "more interpretable results".
        w = m.dist.laplace(0., 1. / lambda_[None, :], shape=(data_dim, latent_dim), name='w')

        # --- Standard Model Components ---

        # We place a standard Gaussian prior on the latent variables 'Z', as described
        # in the note: "We place Gaussian priors on both Z and W...".
        z = m.dist.normal(0., 1., shape=(latent_dim, num_data_points), name='z')

        # This section defines the standard Gaussian noise model, which is separate
        # from the sparsity-inducing prior on the weights.
        precision = m.dist.gamma(1.0, 1.0, name='precision')
        stddv = 1. / jnp.sqrt(precision)

        # This is the standard likelihood, same as in the ARD model. The generative story
        # is unchanged, but the prior on 'W' now enforces the desired sparsity property.
        m.dist.normal(w @ z, stddv, obs=X)

    def model_sparse_robust_ard(self, X = self.X, data_dim = self.data_dim, latent_dim = self.latent_dim, num_data_points = self.num_data_points):
        """
        A combined Sparse, Robust Bayesian PCA model with Automatic Relevance Determination (ARD).

        - Sparsity: Achieved with a Laplace prior on the weights 'w'.
        - Robustness: Achieved with a Student's t-distribution for the likelihood.
        - ARD: Achieved by placing a hierarchical prior on the scale of the Laplace
          distribution, allowing entire latent dimensions to be pruned.
        """
        # --- ARD and Sparsity-Inducing Prior on w ---

        # This is the ARD component. We define a relevance parameter 'lambda_' for each
        # latent dimension. A large lambda will signal that the corresponding
        # component is not relevant and should be shrunk away.
        # The Gamma prior ensures lambda_ is positive.
        lambda_ = m.dist.gamma(1.0, 1.0, shape=(latent_dim,), name='lambda')

        # This is the Sparsity component. We use a Laplace prior for the weights 'w'.
        # The COMBINED effect happens here: the scale of the Laplace distribution is
        # controlled by the ARD parameter 'lambda_'. If a component is irrelevant
        # (large lambda_), the scale becomes small, and the Laplace prior aggressively
        # forces the weights in that column of 'w' to zero.
        # This gives us both sparsity and automatic dimensionality selection.
        w = m.dist.laplace(0., 1. / lambda_[None, :], shape=(data_dim, latent_dim), name='w')

        # --- Standard Prior for Z ---

        # The prior on the latent variables 'Z' remains a standard Gaussian.
        z = m.dist.normal(0., 1., shape=(latent_dim, num_data_points), name='z')

        # --- Robustness to Outliers via a Heavy-Tailed Noise Model ---

        # Prior on the scale of the Student's t-distribution.
        sigma = m.dist.half_cauchy(1.0, name='sigma')

        # Prior on the degrees of freedom 'nu', which controls the robustness.
        nu = m.dist.gamma(2.0, 0.1, name='nu')

        # The likelihood is the Student's t-distribution, which makes the entire
        # model robust to outliers in the observed data 'x_train'.
        m.dist.studentt(df=nu, loc=w @ z, scale=sigma, obs=X)

    ## Get attributes---------------------------------------------
    # --- Functions 1 : 
    def create_reference_from_posterior(self,posterior_w):
        """ Creates a reference component matrix from the posterior samples using SVD. 
        This reference will be used to align all posterior samples.
        Args:
            posterior_w: Array of shape (num_samples, data_dim, latent_dim)
        Returns:
            reference_components: Array of shape (data_dim, latent_dim)
        """
        w_mean = posterior_w.mean(axis=0)
        U, s, Vh = jnp.linalg.svd(w_mean, full_matrices=False)
        reference_components = U @ Vh
        return reference_components

    # --- Functions 2
    def align_posterior_samples(self,posterior_w, reference_components):
        """ Aligns each posterior sample to the reference component matrix using Procrustes analysis.
        Args:
            posterior_w: Array of shape (num_samples, data_dim, latent_dim)
            reference_components: Array of shape (data_dim, latent_dim)
        Returns:
            aligned_w: Array of shape (num_samples, data_dim, latent_dim)
        """
        @jax.vmap
        def align_single_sample(w_sample):
            M = w_sample.T @ reference_components
            u, s, vh = jnp.linalg.svd(M)
            R = u @ vh
            return w_sample @ R
        return align_single_sample(posterior_w)

    # --- Functions 2 :  Function to enforce a deterministic sign convention
    def set_deterministic_sign(self,components):
        """
        Enforces a deterministic sign convention on the component matrix.
        For each component (column), it finds the element with the largest
        absolute value and flips the sign of the entire component if that
        element is negative.
        Args:
            components: The component matrix of shape (data_dim, latent_dim)
        Returns:
            The component matrix with a consistent sign convention.
        """
        # Process each component (column)
        for i in range(components.shape[1]):
            # Find the index of the element with the largest absolute value
            max_abs_idx = jnp.argmax(jnp.abs(components[:, i]))
            # Check the sign of this element
            if components[max_abs_idx, i] < 0:
                # If it's negative, flip the sign of the entire column
                components = components.at[:, i].set(components[:, i] * -1)
        return components

    # --- Function 3: 
    def analyze_aligned_posteriors(self,X_scaled, aligned_w_samples):
        """
        Calculates final PCA attributes using a self-contained, deterministic
        sign convention for the components.
        Args:
            X_scaled: Scaled data matrix of shape (num_datapoints, data_dim)
            aligned_w_samples: Aligned posterior samples of shape (num_samples, data_dim, latent_dim)
        Returns:
            Dictionary with keys for components, variance, etc.
        """
        # Step 1: Get the posterior mean of the aligned samples
        w_mean_aligned = aligned_w_samples.mean(axis=0)

        # Step 2: The orthogonal components are the left singular vectors (U)
        U, s, Vh = jnp.linalg.svd(w_mean_aligned, full_matrices=False)
        final_components = U

        # Step 3: Project the data onto these ORTHOGONAL components
        X_projected = X_scaled @ final_components

        # Step 4: Calculate variance and sort
        explained_variance_unordered = jnp.var(X_projected, axis=0)
        sort_indices = jnp.argsort(explained_variance_unordered)[::-1]

        explained_variance = explained_variance_unordered[sort_indices]
        final_components_sorted = final_components[:, sort_indices]

        # --- NEW: Step 4.5: Apply our own deterministic sign convention ---
        final_components_signed = set_deterministic_sign(final_components_sorted)

        # Step 5: Calculate the final ratio
        total_variance = jnp.var(X_scaled, axis=0).sum()
        explained_variance_ratio = explained_variance / total_variance

        return {
            'components': final_components_signed, # Use the sign-corrected components
            'explained_variance': explained_variance,
            'explained_variance_ratio': explained_variance_ratio,
        }

    ## Plots---------------------------------------------
    def extract_results_for_plot(self,bayesian_results,X):
        # --- Extract Bayesian PCA results ---
        components = bayesian_results['components']
        explained_variance_ratio = bayesian_results['explained_variance_ratio']
        cumulative_variance = jnp.cumsum(explained_variance_ratio)
        num_components = components.shape[1]
        # --- Project data onto the principal components for the score plot ---
        X_projected = X @ components

        component_indices = jnp.arange(1, num_components + 1)

    return components, explained_variance_ratio, cumulative_variance, X_projected, component_indices

    def components(self,bayesian_results, feature_names):
        """Plots the final Bayesian component matrix as a heatmap."""
        bayesian_comps = bayesian_results['components']
        pc_names = [f'PC{i+1}' for i in range(bayesian_comps.shape[1])]
        df_bayesian = pd.DataFrame(bayesian_comps, index=feature_names, columns=pc_names)
        # --- Visual Comparison: Heatmaps ---
        sns.heatmap(df_bayesian, annot=True, cmap='viridis')

    def cumulative_variance(
        self,
        X,
        bayesian_results,
        figsize=(16, 8),
        alpha=0.7,
        fontsize=10,
        color_bar='blue',
        color_cumulative_line='red',
        label_size=16,
        ):

        """Plots the cumulative explained variance for Bayesian PCA.
        Generates a scree plot with cumulative explained variance. 
        Args:
            X: The scaled data matrix of shape (num_datapoints, data_dim).
            bayesian_results: Dictionary containing 'explained_variance_ratio' and 'components'.    
            figsize: Tuple specifying the figure size.
            alpha: Transparency level for the bars.
            fontsize: Font size for annotations.
            color_bar: Color for the bars.
            color_cumulative_line: Color for the cumulative variance line.
            label_size: Font size for axis labels.
        Returns:
            None (displays the plot)
        """
        # --- Extract Bayesian PCA results ---
        components, explained_variance_ratio, cumulative_variance, X_projected, component_indices =     extract_results_for_plot(bayesian_results, X)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars and cumulative line
        ax.bar(
            component_indices,
            explained_variance_ratio,
            alpha=alpha,
            color=color_bar,
            align='center',
            label='Individual explained variance'
        )

        # Plot cumulative explained variance with red dots
        line, = ax.plot(
            component_indices,
            cumulative_variance,
            'ro-',
            color=color_cumulative_line,
            label='Cumulative explained variance'
        )

        # Add annotations for each red dot
        for i, (x, y) in enumerate(zip(component_indices, cumulative_variance)):
            ax.annotate(
                f'{y:.2f}',  # Format the cumulative variance to 2 decimal places
                xy=(x, y),   # Coordinates of the red dot
                xytext=(x, y + 0.05),  # Position of the annotation text (slightly above the dot)
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=fontsize,  # Font size
                color=color_cumulative_line,  # Text color
            )

        # Set labels and ticks
        ax.set_xticks(component_indices)
        ax.set_xlabel('Principal Component', fontsize=label_size)
        ax.set_ylabel('Explained Variance Ratio', fontsize=label_size)


        # Place legend outside the plot
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        )

        # Remove vertical grid lines
        ax.grid(axis='x', linestyle='')
        ax.grid(axis='y', linestyle='--')

        # Remove all spines (lines around the plot)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Adjust layout to prevent legend from being cut off
        plt.tight_layout()
        plt.show()
    
    def confidence_ellipse(self,x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        """
        Create a confidence ellipse for a set of x, y points.
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = jnp.cov(x, y)
        pearson = cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])

        ell_radius_x = jnp.sqrt(1 + pearson)
        ell_radius_y = jnp.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        scale_x = jnp.sqrt(cov[0, 0]) * n_std
        mean_x = jnp.mean(x)

        scale_y = jnp.sqrt(cov[1, 1]) * n_std
        mean_y = jnp.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def variable_correlation(self,X, bayesian_results, feature_names):
        """
        Generates a variable correlation plot with cos2 values for Bayesian PCA results.
        Args:
            X_scaled: The scaled data matrix of shape (num_datapoints, data_dim).
            bayesian_results: Dictionary containing 'components' and 'explained_variance_ratio'.
            feature_names: List of names for the original features.
        Returns:
            None (displays the plot)
        """
        # --- Extract Bayesian PCA results ---
        components, explained_variance_ratio, cumulative_variance, X_projected, component_indices =     extract_results_for_plot    (bayesian_results, X_scaled)

        # Variable contributions (loadings)
        loadings = components

        # Cos2 for variables (quality of representation)
        var_cos2 = loadings**2
        var_cos2_pc1_pc2 = jnp.sum(var_cos2[:, :2], axis=1)
        # Cos2 for individuals
        dist_sq_origin = jnp.sum(X_scaled**2, axis=1)
        ind_cos2 = X_projected**2
        ind_cos2 = ind_cos2 / dist_sq_origin[:, None]
        ind_cos2_pc1_pc2 = jnp.sum(ind_cos2[:, :2], axis=1)

        fig = plt.figure(figsize=(20, 20))


        # 1. Variable Correlation Plot (with cos2)
        ax1 = fig.add_subplot(2, 2, 1, aspect='equal')
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_ylim(-1.1, 1.1)

        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        ax1.add_artist(circle)

        # Draw circle
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        ax1.add_artist(circle)

        # Draw arrows colored by cos2
        for i in range(len(feature_names)):
            scatter = ax1.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                                head_width=0.03, head_length=0.05,
                                color=plt.cm.viridis(var_cos2_pc1_pc2[i]),
                                alpha=0.8)

            ax1.text(loadings[i, 0] * 1.5, loadings[i, 1] * 1.15, feature_names[i],ha='center', va='center',    fontsize=12)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(var_cos2_pc1_pc2), vmax=max  (var_cos2_pc1_pc2)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.set_label('cos2 (Quality of Representation)', rotation=270, labelpad=15)

        ax1.set_xlabel(f'Dim 1 ({explained_variance_ratio[0]:.1%})')
        ax1.set_ylabel(f'Dim 2 ({explained_variance_ratio[1]:.1%})')
        ax1.set_title('Variables Correlation Plot')
        ax1.axhline(0, linestyle='--', color='grey', linewidth=0.8)
        ax1.axvline(0, linestyle='--', color='grey', linewidth=0.8)

        ax1.grid(axis='x', linestyle='')
        ax1.grid(axis='y', linestyle='')

        # Remove all spines (lines around the plot)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        plt.show()

    def variable_correlation(self,X_scaled, bayesian_results, y=None, feature_names=None, target_names=None,
                             color_var=None, size_var=None, shape_var=None):
        """
        Generates a variable correlation plot with cos2 values for Bayesian PCA results.
        """
        # --- Extract Bayesian PCA results ---
        components, explained_variance_ratio, cumulative_variance, X_projected, component_indices =     extract_results_for_plot(bayesian_results, X_scaled)
        loadings = components  # Assign loadings from components
        # Calculate cos2 for variables
        var_cos2 = loadings**2
        var_cos2_pc1_pc2 = jnp.sum(var_cos2[:, :2], axis=1)

        # --- Individuals Plot with Grouping and Ellipses ---
        fig = plt.figure(figsize=(20, 20))
        ax2 = fig.add_subplot(2, 2, 2)

        # Plot the projected data points first
        if y is not None and target_names is not None:
            palette = sns.color_palette("hsv", len(target_names))
            for i, target_name in enumerate(target_names):
                indices = jnp.where(y == i)[0]
                ax2.scatter(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    label=target_name,
                    color=palette[i],
                    alpha=0.8
                )
                confidence_ellipse(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    ax2,
                    edgecolor=palette[i],
                    linewidth=2
                )

        # Plot with continuous color, size, and shape
        if shape_var is not None:
            unique_shapes = jnp.unique(shape_var)
            unique_shapes = [int(shape) if jnp.issubdtype(shape.dtype, jnp.integer) else float(shape)
                             for shape in unique_shapes]
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
            marker_dict = {shape: markers[i % len(markers)] for i, shape in enumerate(unique_shapes)}

        if color_var is not None and size_var is not None and shape_var is not None:
            if isinstance(size_var, (jnp.ndarray, list)):
                size_var = 100 * (jnp.array(size_var) - min(size_var)) / (max(size_var) - min(size_var)) + 10
            for shape in unique_shapes:
                indices = jnp.where(shape_var == shape)[0]
                ax2.scatter(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    c=color_var[indices],
                    s=size_var[indices],
                    cmap='viridis',
                    marker=marker_dict[shape],
                    alpha=0.8,
                    label=f'Shape: {shape}'
                )
            scatter = plt.cm.ScalarMappable(cmap='viridis')
            scatter.set_array(color_var)
            plt.colorbar(scatter, ax=ax2, label='Color Variable')
            # Place legend outside the plot
            ax2.legend(
                title='Shapes',
                bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot
                loc='upper left',           # Location of the legend
                borderaxespad=0.            # Padding between the axes and legend
            )

        elif color_var is not None and shape_var is not None:
            for shape in unique_shapes:
                indices = jnp.where(shape_var == shape)[0]
                ax2.scatter(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    c=color_var[indices],
                    cmap='viridis',
                    marker=marker_dict[shape],
                    alpha=0.8,
                    label=f'Shape: {shape}'
                )
            scatter = plt.cm.ScalarMappable(cmap='viridis')
            scatter.set_array(color_var)
            plt.colorbar(scatter, ax=ax2, label='Color Variable')
            # Place legend outside the plot
            ax2.legend(
                title='Shapes',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )

        elif size_var is not None and shape_var is not None:
            if isinstance(size_var, (jnp.ndarray, list)):
                size_var = 100 * (jnp.array(size_var) - min(size_var)) / (max(size_var) - min(size_var)) + 10
            for shape in unique_shapes:
                indices = jnp.where(shape_var == shape)[0]
                ax2.scatter(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    s=size_var[indices],
                    marker=marker_dict[shape],
                    alpha=0.8,
                    label=f'Shape: {shape}'
                )
            # Place legend outside the plot
            ax2.legend(
                title='Shapes',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )

        elif color_var is not None:
            scatter = ax2.scatter(
                X_projected[:, 0],
                X_projected[:, 1],
                c=color_var,
                cmap='viridis',
                alpha=0.8
            )
            plt.colorbar(scatter, ax=ax2, label='Color Variable')

        elif size_var is not None:
            if isinstance(size_var, (jnp.ndarray, list)):
                size_var = 100 * (jnp.array(size_var) - min(size_var)) / (max(size_var) - min(size_var)) + 10
            ax2.scatter(
                X_projected[:, 0],
                X_projected[:, 1],
                s=size_var,
                alpha=0.8
            )

        elif shape_var is not None:
            for shape in unique_shapes:
                indices = jnp.where(shape_var == shape)[0]
                ax2.scatter(
                    X_projected[indices, 0],
                    X_projected[indices, 1],
                    marker=marker_dict[shape],
                    alpha=0.8,
                    label=f'Shape: {shape}'
                )
            # Place legend outside the plot
            ax2.legend(
                title='Shapes',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )

        elif y is not None and target_names is not None:
            # Place legend outside the plot
            ax2.legend(
                title='Groups',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )

        else:
            ax2.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7)

        ax2.set_xlabel(f'Dim 1 ({explained_variance_ratio[0]:.1%})')
        ax2.set_ylabel(f'Dim 2 ({explained_variance_ratio[1]:.1%})')
        ax2.set_title('Individuals Plot with Confidence Ellipses')
        ax2.axhline(0, linestyle='--', color='grey', linewidth=0.8)
        ax2.axvline(0, linestyle='--', color='grey', linewidth=0.8)
        ax2.grid(axis='x', linestyle='')
        ax2.grid(axis='y', linestyle='')

        # Add arrows for features, scaled based on x and y limits
        if feature_names is not None:
            x_lim = ax2.get_xlim()
            y_lim = ax2.get_ylim()
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            scale_factor = 0.9
            for i, feature in enumerate(feature_names):
                scaled_x = loadings[i, 0] * scale_factor * x_range
                scaled_y = loadings[i, 1] * scale_factor * y_range
                ax2.arrow(
                    0, 0,
                    scaled_x, scaled_y,
                    head_width=0.03 * x_range,
                    head_length=0.05 * y_range,
                    color=plt.cm.viridis(var_cos2_pc1_pc2[i]),
                    alpha=0.8,
                    linewidth=1.5
                )
                ax2.text(
                    scaled_x * 1.15,
                    scaled_y * 1.15,
                    feature,
                    color=plt.cm.viridis(var_cos2_pc1_pc2[i]),
                    ha='center',
                    va='center',
                    fontsize=10
                )

        # Remove all spines (lines around the plot)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        # Adjust layout to prevent legend from being cut off
        plt.tight_layout()
        plt.show()

        return ax2

