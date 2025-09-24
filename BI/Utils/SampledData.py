from BI.Utils.ImportManager import LazyImporter
import numpy as np
import jax.numpy as jnp
from jax import tree_util
import itertools
import scipy.stats as stats

# Create a global instance of the importer
importer = LazyImporter()

# Schedule the heavy libraries for background import as soon as this module is loaded
importer.schedule_import("plotly.graph_objects", "go")
importer.schedule_import("plotly.express", "px")
importer.schedule_import("plotly.figure_factory", "ff")
importer.schedule_import("plotly.colors", "n_colors")
importer.schedule_import("seaborn", "sns")
importer.schedule_import("matplotlib.pyplot", "plt")

class SampledData:
    """
    A wrapper for a JAX numpy array that adds interactive plotting methods using Plotly
    while preserving all other JAX array functionalities through attribute delegation.
    """

    def __init__(self, data):
        """
        Initializes the SampledData object.

        Args:
            data (jnp.ndarray): The sampled JAX array.
        """
        # Ensure data is a JAX array for consistency
        self._data = jnp.asarray(data)

    def _wrap_result(self, result):
        """Wraps the result in a SampledData object if it's a JAX array with dimension > 0."""
        if isinstance(result, jnp.ndarray) and result.ndim > 0:
            return SampledData(result)
        return result

    def _extract_data(self, other):
        """Extracts the JAX array if the other object is a SampledData instance."""
        if isinstance(other, SampledData):
            return other._data
        return other

    def _tree_flatten(self):
        """
        Tells JAX how to flatten the object.
        Returns a tuple of array leaves and a tuple of non-array metadata.
        """
        children = (self._data,)  # The JAX arrays to be traced
        aux_data = None           # Any static data you need to reconstruct the class
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Tells JAX how to reconstruct the object from leaves and metadata.
        """
        # aux_data is None, children is a tuple containing the new JAX array
        return cls(children[0])

    def __repr__(self):
        return f"SampledData({self._data})"

    def hist(
        self, 
        title="Histogram of Sampled Data", 
        nbinsx=30, xaxis_title="Value", yaxis_title="Frequency", 
        template="plotly_white", 
        interactive = True,
        figsize=(6, 4),**kwargs
    ):
        if interactive:
            """Interactive histogram visualization."""
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")

            fig = go.Figure()

            if self._data.ndim == 1:
                 fig.add_trace(go.Histogram(x=np.array(self._data), nbinsx=nbinsx, **kwargs))

            elif self._data.ndim == 2:
                for i in range(self._data.shape[1]):
                    fig.add_trace(go.Histogram(x=np.array(self._data[:, i]), nbinsx=nbinsx, name=f"Var {i}  ", **kwargs))

            elif self._data.ndim == 3:
                n_samples, n_groups, n_times = self._data.shape
                bins = nbinsx
                bin_edges = jnp.linspace(-4, 4, bins + 1)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                colorscale = px.colors.qualitative.Plotly
                colors = [colorscale[g % len(colorscale)] for g in range(n_groups)]

                for g in range(n_groups):
                    for t in range(n_times):
                        values = np.array(self._data[:, t, g])
                        counts, _ = np.histogram(values, bins=bin_edges, density=True)

                        x = bin_centers
                        y = np.full_like(bin_centers, t)
                        z = counts + g

                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode="lines",
                            line=dict(width=5, color=colors[g]),
                            name=f"Group {g}, Time {t}"
                        ))
            fig.update_layout(title=title, template=template,
                             xaxis_title=xaxis_title, yaxis_title=yaxis_title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            
            plt.figure(figsize=figsize)
            if self._data.ndim == 1:
               sns.histplot(self._data, bins=nbinsx, kde=False)
            elif self._data.ndim == 2:
               for i in range(self._data.shape[1]):
                   sns.histplot(self._data[:, i], bins=nbinsx, kde=False, label=f"Var {i}", alpha=0.5)
               plt.legend()
            plt.title(title)
            plt.show()


    def corr_heatmap(self, title="Correlation Matrix", template="plotly_white", digits=5, interactive = True, figsize=(6, 5), **kwargs):
        """
        Visualizes the correlation matrix of the 2D data as a heatmap with annotations.
        Values are explicitly formatted as strings to ensure consistent rounding in the plot.
        The y-axis is inverted to match standard matrix representation.
        """
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if self._data.ndim != 2:
                raise ValueError(f"Correlation heatmap is only supported for 2D data. Your data has   {self.  _data.ndim} dimensions.")
            
            # --- FIX 1: Always calculate the correlation from the samples ---
            # The function expects self._data to be samples [n_samples, n_variables]
            if self._data.shape[0] < 2:
                raise ValueError("Cannot calculate correlation with fewer than 2 samples.")
            corr_matrix = self._data
        
            # --- FIX 2: Format annotations into strings for consistent display ---
            formatted_text = np.full(corr_matrix.shape, "", dtype=object)
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    # Use an f-string to force formatting with 'digits' decimal places
                    formatted_text[i, j] = f"{corr_matrix[i, j]:.{digits}f}"
        
            x = [f'Var {i}' for i in range(corr_matrix.shape[1])]
            y = [f'Var {i}' for i in range(corr_matrix.shape[0])]
            
            # Use the original correlation matrix for colors and the formatted text for annotations
            fig = ff.create_annotated_heatmap(
               z=corr_matrix, 
               x=x, 
               y=y, 
               annotation_text=formatted_text, # Use the string-formatted matrix here
               colorscale='Viridis', 
               **kwargs
            )
            
            # Invert the y-axis to have Var 0 at the bottom
            fig.update_yaxes(autorange='reversed')
            
            fig.update_layout(title_text=title, template=template)
            fig.show()

        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Heatmap requires 2D data.")
            plt.figure(figsize=figsize)
            sns.heatmap(self._data, cmap="viridis")
            plt.title(title)
            plt.show()


    def boxplot(self, title="Boxplot of Matrix Samples", template="plotly_white", interactive = True,figsize=(6, 4),  **kwargs):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Box(y= self._data[:, i], name=f"Var {i}", **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.boxplot(data= self._data)
            plt.title(title)
            plt.show()

    def violinplot(self, title="Violin Plot of Matrix Samples", template="plotly_white", interactive = True, figsize=(6, 4), **kwargs):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Violin(y= self._data[:, i], name=f"Var {i}", box_visible=True,     meanline_visible=True, **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.violinplot(data=self._data, inner="box")
            plt.title(title)
            plt.show()

    def pairplot(self, max_vars=5, title="Pairwise Scatter Plots", interactive = True):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Pairplot requires 2D data.")
            n_vars = min( self._data.shape[1], max_vars)
            fig = go.Figure()
            for i, j in itertools.combinations(range(n_vars), 2):
                fig.add_trace(go.Scatter(
                    x= self._data[:, i],
                    y= self._data[:, j],
                    mode="markers",
                    name=f"Var {i} vs Var {j}",
                    opacity=0.5
                ))
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Pairplot requires 2D data.")
            n_vars = min(self._data.shape[1], max_vars)
            import pandas as pd
            df = pd.DataFrame(self._data[:, :n_vars], columns=[f"Var {i}" for i in range(n_vars)])
            sns.pairplot(df, diag_kind="kde")
            plt.suptitle(title, y=1.02)
            plt.show()

    def timeseries(self, credible_interval=0.9, title="Sampled Time Series", interactive = True,figsize=(8, 4)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")

            if  self._data.ndim != 2:
                raise ValueError("Timeseries requires shape [n_samples, n_time].")
            mean =  self._data.mean(axis=0)
            lower = np.percentile( self._data, (1 - credible_interval) / 2 * 100, axis=0)
            upper = np.percentile( self._data, (1 + credible_interval) / 2 * 100, axis=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=mean, mode="lines", name="Mean"))
            fig.add_trace(go.Scatter(y=upper, mode="lines", name="Upper", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(y=lower, mode="lines", name="Lower", line=dict(dash="dash"),
                                     fill="tonexty", fillcolor="rgba(0,100,200,0.2)"))
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Timeseries requires shape [n_samples, n_time].")
            mean = self._data.mean(axis=0)
            lower = np.percentile(self._data, (1 - credible_interval) / 2 * 100, axis=0)
            upper = np.percentile(self._data, (1 + credible_interval) / 2 * 100, axis=0)

            plt.figure(figsize=figsize)
            plt.plot(mean, label="Mean")
            plt.fill_between(np.arange(len(mean)), lower, upper, alpha=0.3, label=f"{int    (credible_interval*100)}% CI")
            plt.title(title)
            plt.legend()
            plt.show()

    def scatter3d(self, title="3D Scatter of Samples", interactive = True,figsize=(6, 5)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if self._data.ndim != 2 or self._data.shape[1] < 3:
                raise ValueError("Need shape [n_samples, >=3] for 3D scatter.")
            fig = go.Figure(data=[go.Scatter3d(
            x=self._data[:, 0], y=self._data[:, 1], z=self._data[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.5)
            )])
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2 or self._data.shape[1] < 3:
                raise ValueError("Need shape [n_samples, >=3] for 3D scatter.")
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self._data[:, 0], self._data[:, 1], self._data[:, 2], s=5, alpha=0.5)
            ax.set_title(title)
            plt.show()

    def traceplot(self, title="Trace Plot of Samples", interactive = True,figsize=(10, 6)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Trace plot requires [n_samples, n_chains/variables].")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Scatter(y= self._data[:, i], mode="lines", name=f"Var {i}", opacity=0.7))
            fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Value")
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Trace plot requires [n_samples, n_chains/variables].")
            plt.figure(figsize=figsize)
            for i in range(self._data.shape[1]):
                plt.plot(self._data[:, i], label=f"Var {i}", alpha=0.7)
            plt.title(title)
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.legend()
            plt.show()
    
    def autocorr(self, lags=50, title="Autocorrelation Plot", interactive = True,figsize=(8, 4)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim == 1:
                series =  self._data
                acf = [np.corrcoef(series[:-k], series[k:])[0, 1] if k > 0 else 1 for k in range(lags)]
                fig = go.Figure([go.Bar(x=list(range(lags)), y=acf)])
                fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Autocorrelation")
                fig.show()
            elif  self._data.ndim == 2:
                fig = go.Figure()
                for i in range( self._data.shape[1]):
                    series =  self._data[:, i]
                    acf = [np.corrcoef(series[:-k], series[k:])[0, 1] if k > 0 else 1 for k in range(lags)]
                    fig.add_trace(go.Bar(x=list(range(lags)), y=acf, name=f"Var {i}", opacity=0.5))
                fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Autocorrelation")
                fig.show()
            else:
                raise ValueError("Autocorrelation requires 1D or 2D data.")
        else:
            plt=importer.get_module("plt")
            plt.figure(figsize=figsize)
            if self._data.ndim == 1:
                series = self._data
                acf = [np.corrcoef(series[:-k], series[k:])[0, 1] if k > 0 else 1 for k in range(lags)]
                plt.bar(range(lags), acf)
            elif self._data.ndim == 2:
                for i in range( self._data.shape[1]):
                    series = self._data[:, i]
                    acf = [jnp.corrcoef(series[:-k], series[k:])[0, 1] if k > 0 else 1 for k in range(lags)]
                    plt.bar(range(lags), acf, alpha=0.5, label=f"Var {i}")
                plt.legend()
            else:
                raise ValueError("Autocorrelation requires 1D or 2D data.")
            plt.title(title)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.show()
    # -----------------
    # Delegation
    
    def density(self, title="Density Plot", template="plotly_white", **kwargs):
        """
        Visualizes the distribution of the data using a density plot.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.figure_factory as ff
        from plotly.colors import n_colors

        if self._data.ndim > 2:
            raise ValueError(f"Density plot is only supported for 1D or 2D data. Your data has {self._data.ndim} dimensions.")

        if self._data.ndim == 1:
            hist_data = [np.array(self._data)]
            group_labels = ['Sample']
        else: # 2D
            hist_data = [np.array(self._data[:, i]) for i in range(self._data.shape[1])]
            group_labels = [f'Var {i}' for i in range(self._data.shape[1])]

        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2, **kwargs)
        fig.update_layout(title_text=title, template=template)
        fig.show()
        
    def ridgeline(self, title="Ridgeline Plot", template="plotly_white",interactive = True,category_labels=None, offset=2):
        if interactive:
            go=importer.get_module("go")
            n_colors=importer.get_module("n_colors")
            if self._data.ndim  not in [2, 3]:
                raise ValueError(f"Ridgeline plot requires 2D or 3D data. Your data has {self._data.ndim}       dimensions.")

            colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', self._data.shape[1], colortype='rgb')


            fig = go.Figure()
            for i, (data_line, color) in enumerate(zip(self._data, colors)):
                fig.add_trace(
                    go.Violin(x=data_line, line_color='black', name=i, fillcolor=color)
                    )

            # use negative ... cuz I'm gonna flip things later
            fig = fig.update_traces(orientation='h', side='negative', width=3, points=False, opacity=1)
            # reverse the (z)-order of the traces

            # flip the y axis (negative violin is now positive and traces on the top are now on the bottom)
            fig.update_layout(legend_traceorder='reversed', yaxis_autorange='reversed').show()

            fig.update_layout(
                title=title,
                template=template,
                showlegend=False,
                yaxis=dict(showticklabels=False, title="Categories"),
                xaxis_title="Value"
            )
            
        else:
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Ridgeline requires 2D [samples, categories].")
            n_samples, n_categories = self._data.shape
            if category_labels is None:
                category_labels = [f"Category {i}" for i in range(n_categories)]

            plt.figure(figsize=(8, n_categories * 0.7))
            for i in range(n_categories):
                values = self._data[:, i]
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min() - 1, values.max() + 1, 300)
                y_vals = kde(x_range)
                plt.fill_between(x_range, y_vals + i * offset, i * offset, alpha=0.6)
                plt.plot(x_range, y_vals + i * offset, lw=1)
            plt.yticks([i * offset for i in range(n_categories)], category_labels)
            plt.title(title)
            plt.show()


    def surface_3d(self, title="3D Surface Plot", template="plotly_white", **kwargs):
        """
        Visualizes 3D data as a surface plot.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.figure_factory as ff
        from plotly.colors import n_colors
        if self._data.ndim != 2:
            raise ValueError(f"3D Surface plot is only supported for 2D data. Your data has {self._data.ndim} dimensions.")

        fig = go.Figure(data=[go.Surface(z=self._data, colorscale='Viridis', **kwargs)])
        fig.update_layout(
            title=title, template=template,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            )
        )
        fig.show()

    def ppc_plot(self, observed_data, n_samples_to_plot=50, title="Posterior Predictive Check", template="plotly_white"):
        """
        Creates a posterior predictive check plot.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.figure_factory as ff
        from plotly.colors import n_colors
        if self._data.ndim != 2:
            raise ValueError(f"PPC plot expects 2D sampled data [n_draws, n_observations]. Your data has {self._data.ndim} dimensions.")

        observed_data = np.array(observed_data)
        sampled_data = np.array(self._data)

        fig = go.Figure()

        # Plot densities of posterior predictive samples (thin, semi-transparent lines)
        subset_indices = np.random.choice(sampled_data.shape[0], size=min(n_samples_to_plot, sampled_data.shape[0]), replace=False)
        for i in subset_indices:
            density = stats.gaussian_kde(sampled_data[i, :])
            x_vals = np.linspace(min(observed_data.min(), sampled_data.min()), max(observed_data.max(), sampled_data.max()), 200)
            fig.add_trace(go.Scatter(
                x=x_vals, y=density(x_vals),
                mode='lines',
                line=dict(width=1, color='rgba(70, 130, 180, 0.5)'), # SteelBlue with alpha
                showlegend=False
            ))

        # Plot density of observed data (thick, solid line)
        density_observed = stats.gaussian_kde(observed_data)
        x_vals = np.linspace(min(observed_data.min(), sampled_data.min()), max(observed_data.max(), sampled_data.max()), 200)
        fig.add_trace(go.Scatter(
            x=x_vals, y=density_observed(x_vals),
            mode='lines',
            line=dict(width=3, color='black'),
            name='Observed Data Density'
        ))

        # Update layout for a clean, modern look
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Density",
            template=template,
            legend=dict(x=0.01, y=0.98) # Position legend inside the plot
        )
        fig.show()

# ================== Array basics ==================
    def hdi(self, cred_mass=0.95):
        """
        Compute highest density interval (HDI) from samples.

        Args:
            samples: 1D jax.numpy array of samples
            cred_mass: float, credible mass (default 0.95)

        Returns:
            (hdi_min, hdi_max)
        """
        samples = jnp.sort(self._data)
        n = samples.shape[0]
        interval_idx_inc = int(jnp.floor(cred_mass * n))

        lows  = samples[: n - interval_idx_inc]
        highs = samples[interval_idx_inc:]
        widths = highs - lows

        min_idx = jnp.argmin(widths)
        return lows[min_idx], highs[min_idx]

# ================== ARITHMETIC OPERATORS ==================

    def __add__(self, other):
        return self._wrap_result(self._data + self._extract_data(other))
    def __radd__(self, other):
        return self._wrap_result(self._extract_data(other) + self._data)
    def __sub__(self, other):
        return self._wrap_result(self._data - self._extract_data(other))
    def __rsub__(self, other):
        return self._wrap_result(self._extract_data(other) - self._data)
    def __mul__(self, other):
        return self._wrap_result(self._data * self._extract_data(other))
    def __rmul__(self, other):
        return self._wrap_result(self._extract_data(other) * self._data)
    def __truediv__(self, other):
        return self._wrap_result(self._data / self._extract_data(other))
    def __rtruediv__(self, other):
        return self._wrap_result(self._extract_data(other) / self._data)
    def __floordiv__(self, other):
        return self._wrap_result(self._data // self._extract_data(other))
    def __rfloordiv__(self, other):
        return self._wrap_result(self._extract_data(other) // self._data)
    def __mod__(self, other):
        return self._wrap_result(self._data % self._extract_data(other))
    def __rmod__(self, other):
        return self._wrap_result(self._extract_data(other) % self._data)
    def __pow__(self, other):
        return self._wrap_result(self._data ** self._extract_data(other))
    def __rpow__(self, other):
        return self._wrap_result(self._extract_data(other) ** self._data)
    def __matmul__(self, other):
        return self._wrap_result(self._data @ self._extract_data(other))
    def __rmatmul__(self, other):
        return self._wrap_result(self._extract_data(other) @ self._data)

    # Unary operators
    def __neg__(self):
        return self._wrap_result(-self._data)
    def __pos__(self):
        return self._wrap_result(+self._data)
    def __abs__(self):
        return self._wrap_result(jnp.abs(self._data))

    # Bitwise operators
    def __and__(self, other):
        return self._wrap_result(self._data & self._extract_data(other))
    def __rand__(self, other):
        return self._wrap_result(self._extract_data(other) & self._data)
    def __or__(self, other):
        return self._wrap_result(self._data | self._extract_data(other))
    def __ror__(self, other):
        return self._wrap_result(self._extract_data(other) | self._data)
    def __xor__(self, other):
        return self._wrap_result(self._data ^ self._extract_data(other))
    def __rxor__(self, other):
        return self._wrap_result(self._extract_data(other) ^ self._data)
    def __invert__(self):
        return self._wrap_result(~self._data)
    def __lshift__(self, other):
        return self._wrap_result(self._data << self._extract_data(other))
    def __rlshift__(self, other):
        return self._wrap_result(self._extract_data(other) << self._data)
    def __rshift__(self, other):
        return self._wrap_result(self._data >> self._extract_data(other))
    def __rrshift__(self, other):
        return self._wrap_result(self._extract_data(other) >> self._data)

    # Comparison operators
    def __eq__(self, other):
        return self._data == self._extract_data(other)
    def __ne__(self, other):
        return self._data != self._extract_data(other)
    def __lt__(self, other):
        return self._data < self._extract_data(other)
    def __le__(self, other):
        return self._data <= self._extract_data(other)
    def __gt__(self, other):
        return self._data > self._extract_data(other)
    def __ge__(self, other):
        return self._data >= self._extract_data(other)

    # Built-in functions
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        for item in self._data:
            yield self._wrap_result(item) if isinstance(item, jnp.ndarray) else item
    def __contains__(self, item):
        return self._extract_data(item) in self._data
    def __bool__(self):
        return bool(self._data)
    def __int__(self):
        return int(self._data)
    def __float__(self):
        return float(self._data)
    def __complex__(self):
        return complex(self._data)

    # Array protocol methods
    def __array__(self, dtype=None):
        return np.array(self._data, dtype=dtype) if dtype else np.array(self._data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted_inputs = [self._extract_data(inp) for inp in inputs]
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)
        return self._wrap_result(result)

    # Indexing and slicing
    def __getitem__(self, idx):
        result = self._data[idx]
        return self._wrap_result(result) if isinstance(result, jnp.ndarray) and result.ndim > 0 else result

    def __setitem__(self, idx, value):
        self._data = self._data.at[idx].set(self._extract_data(value))

    def __getattr__(self, name):
        """
        Dynamically delegates attribute access to the underlying JAX array.
        This is the most direct and robust way to mimic the array's behavior.
        """
        attr = getattr(self._data, name)

        if callable(attr):
            # If the attribute is a method (like .sum(), .reshape()), return
            # a new function that calls the original method and wraps the result.
            def wrapper(*args, **kwargs):
                unwrapped_args = [self._extract_data(arg) for arg in args]
                unwrapped_kwargs = {k: self._extract_data(v) for k, v in kwargs.items()}

                result = attr(*unwrapped_args, **unwrapped_kwargs)
                return self._wrap_result(result)
            return wrapper
        else:
            # If the attribute is a property (like .T, .shape), just return its value,
            # wrapping it if it's an array.
            return self._wrap_result(attr)

    def __dir__(self):
            # Start with the attributes of this class
            own_attrs = set(super().__dir__())
            # Add the attributes from the wrapped data object
            data_attrs = set(dir(self._data))
            return sorted(list(own_attrs | data_attrs))

    # --- ADD THIS METHOD ---
    def __jax_array__(self):
        """
        Allows the class to be treated as a JAX array directly.
        This is the key to robust compatibility with transformations like vmap.
        """
        return self._data
        
    # --- KEEP YOUR PYTREE METHODS ---
    def _tree_flatten(self):
        children = (self._data,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(children[0])

tree_util.register_pytree_node(
    SampledData,
    lambda x: x._tree_flatten(),    # The flatten function is the instance method
    SampledData._tree_unflatten     # The unflatten function is the class method
)