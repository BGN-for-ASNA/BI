import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import jax.numpy as jnp
import itertools
import scipy.stats as stats
import re
from plotly.colors import n_colors
import seaborn as sns
import matplotlib.pyplot as plt
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
        """Wraps the result in a SampledData object if it's a JAX array."""
        if isinstance(result, jnp.ndarray):
            return SampledData(result)
        return result

    def _extract_data(self, other):
        """Extracts the JAX array if the other object is a SampledData instance."""
        if isinstance(other, SampledData):
            return other._data
        return other

    def __repr__(self):
        return f"SampledData({self._data})"

    def hist(
        self, 
        title="Histogram of Sampled Data", 
        nbinsx=30, xaxis_title="Value", yaxis_title="Frequency", 
        template="plotly_white", 
        interactive = True,
        figsize=(6, 4),
        **kwargs
    ):
        if interactive:
            """Interactive histogram visualization."""

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
            if self._data.ndim != 2:
                raise ValueError(f"Correlation heatmap is only supported for 2D data. Your data has   {self.  _data.ndim} dimensions.")
            
            # --- FIX 1: Always calculate the correlation from the samples ---
            # The function expects self._data to be samples [n_samples, n_variables]
            if self._data.shape[0] < 2:
                raise ValueError("Cannot calculate correlation with fewer than 2 samples.")
            corr_matrix = np.corrcoef(self._data, rowvar=False)
        
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
            if self._data.ndim != 2:
                raise ValueError("Heatmap requires 2D data.")
            plt.figure(figsize=figsize)
            sns.heatmap(self._data, cmap="viridis")
            plt.title(title)
            plt.show()


    def boxplot(self, title="Boxplot of Matrix Samples", template="plotly_white", interactive = True,figsize=(6, 4),  **kwargs):
        if interactive:
            
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Box(y= self._data[:, i], name=f"Var {i}", **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.boxplot(data= self._data)
            plt.title(title)
            plt.show()

    def violinplot(self, title="Violin Plot of Matrix Samples", template="plotly_white", interactive = True, figsize=(6, 4), **kwargs):
        if interactive:
            
            if  self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Violin(y= self._data[:, i], name=f"Var {i}", box_visible=True,     meanline_visible=True, **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            
            if self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.violinplot(data=self._data, inner="box")
            plt.title(title)
            plt.show()

    def pairplot(self, max_vars=5, title="Pairwise Scatter Plots", interactive = True):
        if interactive:
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
            
            if  self._data.ndim != 2:
                raise ValueError("Trace plot requires [n_samples, n_chains/variables].")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Scatter(y= self._data[:, i], mode="lines", name=f"Var {i}", opacity=0.7))
            fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Value")
            fig.show()
        else:
            
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
        print(f"Displaying: {title}")
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
        print(f"Displaying: {title}")
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
        print(f"Displaying: {title}")
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

    # ================== ARRAY PROPERTIES ==================

    @property
    def shape(self):
        return self._data.shape
    @property
    def dtype(self):
        return self._data.dtype
    @property
    def ndim(self):
        return self._data.ndim
    @property
    def size(self):
        return self._data.size
    @property
    def T(self):
        return SampledData(self._data.T)
    @property
    def real(self):
        return SampledData(self._data.real)
    @property
    def imag(self):
        return SampledData(self._data.imag)
    @property
    def flat(self):
        for item in self._data.flat:
            yield item

    # ================== ARRAY MANIPULATION METHODS ==================

    # Shape manipulation
    def reshape(self, *args, **kwargs):
        return SampledData(self._data.reshape(*args, **kwargs))
    def resize(self, *args, **kwargs):
        return SampledData(jnp.resize(self._data, *args, **kwargs))
    def flatten(self, *args, **kwargs):
        return SampledData(self._data.flatten(*args, **kwargs))
    def ravel(self, *args, **kwargs):
        return SampledData(self._data.ravel(*args, **kwargs))
    def squeeze(self, *args, **kwargs):
        return SampledData(jnp.squeeze(self._data, *args, **kwargs))
    def expand_dims(self, axis):
        return SampledData(jnp.expand_dims(self._data, axis))
    def transpose(self, *args, **kwargs):
        return SampledData(jnp.transpose(self._data, *args, **kwargs))
    def swapaxes(self, axis1, axis2):
        return SampledData(jnp.swapaxes(self._data, axis1, axis2))
    def moveaxis(self, source, destination):
        return SampledData(jnp.moveaxis(self._data, source, destination))
    def rollaxis(self, axis, start=0):
        return SampledData(jnp.rollaxis(self._data, axis, start))

    # Array joining
    def concatenate(self, others, axis=0):
        arrays = [self._data] + [self._extract_data(other) for other in others]
        return SampledData(jnp.concatenate(arrays, axis=axis))
    def append(self, values, axis=None):
        return SampledData(jnp.append(self._data, self._extract_data(values), axis))
    def insert(self, obj, values, axis=None):
        return SampledData(jnp.insert(self._data, obj, self._extract_data(values), axis))

    # Array splitting
    def split(self, indices_or_sections, axis=0):
        results = jnp.split(self._data, indices_or_sections, axis)
        return [SampledData(r) for r in results]
    def hsplit(self, indices_or_sections):
        results = jnp.hsplit(self._data, indices_or_sections)
        return [SampledData(r) for r in results]
    def vsplit(self, indices_or_sections):
        results = jnp.vsplit(self._data, indices_or_sections)
        return [SampledData(r) for r in results]

    # Tiling
    def tile(self, reps):
        return SampledData(jnp.tile(self._data, reps))
    def repeat(self, repeats, axis=None):
        return SampledData(jnp.repeat(self._data, repeats, axis))

    # Padding
    def pad(self, pad_width, mode='constant', **kwargs):
        return SampledData(jnp.pad(self._data, pad_width, mode, **kwargs))

    # Flipping and rotation
    def flip(self, axis=None):
        return SampledData(jnp.flip(self._data, axis))
    def fliplr(self):
        return SampledData(jnp.fliplr(self._data))
    def flipud(self):
        return SampledData(jnp.flipud(self._data))
    def rot90(self, k=1, axes=(0, 1)):
        return SampledData(jnp.rot90(self._data, k, axes))
    def roll(self, shift, axis=None):
        return SampledData(jnp.roll(self._data, shift, axis))

    # ================== MATHEMATICAL FUNCTIONS ==================

    # Trigonometric functions
    def sin(self):
        return SampledData(jnp.sin(self._data))
    def cos(self):
        return SampledData(jnp.cos(self._data))
    def tan(self):
        return SampledData(jnp.tan(self._data))
    def arcsin(self):
        return SampledData(jnp.arcsin(self._data))
    def arccos(self):
        return SampledData(jnp.arccos(self._data))
    def arctan(self):
        return SampledData(jnp.arctan(self._data))
    def arctan2(self, other):
        return SampledData(jnp.arctan2(self._data, self._extract_data(other)))
    def sinh(self):
        return SampledData(jnp.sinh(self._data))
    def cosh(self):
        return SampledData(jnp.cosh(self._data))
    def tanh(self):
        return SampledData(jnp.tanh(self._data))
    def arcsinh(self):
        return SampledData(jnp.arcsinh(self._data))
    def arccosh(self):
        return SampledData(jnp.arccosh(self._data))
    def arctanh(self):
        return SampledData(jnp.arctanh(self._data))
    def degrees(self):
        return SampledData(jnp.degrees(self._data))
    def radians(self):
        return SampledData(jnp.radians(self._data))

    # Exponential and logarithmic functions
    def exp(self):
        return SampledData(jnp.exp(self._data))
    def exp2(self):
        return SampledData(jnp.exp2(self._data))
    def expm1(self):
        return SampledData(jnp.expm1(self._data))
    def log(self):
        return SampledData(jnp.log(self._data))
    def log2(self):
        return SampledData(jnp.log2(self._data))
    def log10(self):
        return SampledData(jnp.log10(self._data))
    def log1p(self):
        return SampledData(jnp.log1p(self._data))
    def power(self, exponent):
        return SampledData(jnp.power(self._data, self._extract_data(exponent)))
    def sqrt(self):
        return SampledData(jnp.sqrt(self._data))
    def square(self):
        return SampledData(jnp.square(self._data))
    def cbrt(self):
        return SampledData(jnp.cbrt(self._data))

    # Rounding functions
    def round(self, decimals=0):
        return SampledData(jnp.round(self._data, decimals))
    def rint(self):
        return SampledData(jnp.rint(self._data))
    def ceil(self):
        return SampledData(jnp.ceil(self._data))
    def floor(self):
        return SampledData(jnp.floor(self._data))
    def trunc(self):
        return SampledData(jnp.trunc(self._data))
    def fix(self):
        return SampledData(jnp.fix(self._data))

    # Complex number functions
    def angle(self):
        return SampledData(jnp.angle(self._data))
    def conj(self):
        return SampledData(jnp.conj(self._data))
    def conjugate(self):
        return SampledData(jnp.conjugate(self._data))

    # Absolute and sign functions
    def fabs(self):
        return SampledData(jnp.fabs(self._data))
    def sign(self):
        return SampledData(jnp.sign(self._data))
    def copysign(self, other):
        return SampledData(jnp.copysign(self._data, self._extract_data(other)))

    # ================== STATISTICAL FUNCTIONS ==================

    def sum(self, axis=None, dtype=None, keepdims=False):
        result = jnp.sum(self._data, axis=axis, dtype=dtype, keepdims=keepdims)
        return self._wrap_result(result)
    def prod(self, axis=None, dtype=None, keepdims=False):
        result = jnp.prod(self._data, axis=axis, dtype=dtype, keepdims=keepdims)
        return self._wrap_result(result)
    def mean(self, axis=None, dtype=None, keepdims=False):
        result = jnp.mean(self._data, axis=axis, dtype=dtype, keepdims=keepdims)
        return self._wrap_result(result)
    def average(self, axis=None, weights=None):
        weights = self._extract_data(weights) if weights is not None else None
        result = jnp.average(self._data, axis=axis, weights=weights)
        return self._wrap_result(result)
    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        result = jnp.std(self._data, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return self._wrap_result(result)
    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        result = jnp.var(self._data, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return self._wrap_result(result)
    def median(self, axis=None, keepdims=False):
        result = jnp.median(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def percentile(self, q, axis=None, keepdims=False):
        result = jnp.percentile(self._data, q, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def quantile(self, q, axis=None, keepdims=False):
        result = jnp.quantile(self._data, q, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)

    # Extrema finding
    def min(self, axis=None, keepdims=False):
        result = jnp.min(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def max(self, axis=None, keepdims=False):
        result = jnp.max(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def argmin(self, axis=None, keepdims=False):
        result = jnp.argmin(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def argmax(self, axis=None, keepdims=False):
        result = jnp.argmax(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)
    def ptp(self, axis=None, keepdims=False):
        result = jnp.ptp(self._data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result)

    # Cumulative functions
    def cumsum(self, axis=None, dtype=None):
        return SampledData(jnp.cumsum(self._data, axis=axis, dtype=dtype))
    def cumprod(self, axis=None, dtype=None):
        return SampledData(jnp.cumprod(self._data, axis=axis, dtype=dtype))

    # Differences
    def diff(self, n=1, axis=-1):
        return SampledData(jnp.diff(self._data, n=n, axis=axis))
    def gradient(self, *varargs, **kwargs):
        result = jnp.gradient(self._data, *varargs, **kwargs)
        return [SampledData(r) for r in result] if isinstance(result, list) else SampledData(result)

    # Correlation and covariance
    def corrcoef(self, y=None, rowvar=True):
        y_data = self._extract_data(y) if y is not None else None
        return SampledData(jnp.corrcoef(self._data, y_data, rowvar=rowvar))
    def cov(self, y=None, rowvar=True, ddof=None):
        y_data = self._extract_data(y) if y is not None else None
        return SampledData(jnp.cov(self._data, y_data, rowvar=rowvar, ddof=ddof))

    # Histograms
    def histogram(self, bins=10, range=None, weights=None, density=None):
        weights = self._extract_data(weights) if weights is not None else None
        hist, bin_edges = jnp.histogram(self._data, bins=bins, range=range,
                                       weights=weights, density=density)
        return SampledData(hist), SampledData(bin_edges)

    # ================== SORTING AND SEARCHING ==================

    def sort(self, axis=-1, kind=None, order=None):
        return SampledData(jnp.sort(self._data, axis=axis, kind=kind, order=order))
    def argsort(self, axis=-1, kind=None, order=None):
        return SampledData(jnp.argsort(self._data, axis=axis, kind=kind, order=order))
    def lexsort(self, axis=-1):
        return SampledData(jnp.lexsort(self._data, axis=axis))
    def partition(self, kth, axis=-1):
        return SampledData(jnp.partition(self._data, kth, axis=axis))
    def argpartition(self, kth, axis=-1):
        return SampledData(jnp.argpartition(self._data, kth, axis=axis))
    def searchsorted(self, v, side='left', sorter=None):
        v_data = self._extract_data(v)
        sorter_data = self._extract_data(sorter) if sorter is not None else None
        return SampledData(jnp.searchsorted(self._data, v_data, side=side, sorter=sorter_data))

    # Finding elements
    def where(self, condition, x=None, y=None):
        x_data = self._extract_data(x) if x is not None else None
        y_data = self._extract_data(y) if y is not None else None
        if x_data is None and y_data is None:
            result = jnp.where(condition)
            return tuple(SampledData(r) for r in result)
        else:
            return SampledData(jnp.where(condition, x_data, y_data))
    def nonzero(self):
        result = jnp.nonzero(self._data)
        return tuple(SampledData(r) for r in result)

    # ================== LOGICAL FUNCTIONS ==================

    def all(self, axis=None, keepdims=False):
        return jnp.all(self._data, axis=axis, keepdims=keepdims)
    def any(self, axis=None, keepdims=False):
        return jnp.any(self._data, axis=axis, keepdims=keepdims)
    def logical_and(self, other):
        return SampledData(jnp.logical_and(self._data, self._extract_data(other)))
    def logical_or(self, other):
        return SampledData(jnp.logical_or(self._data, self._extract_data(other)))
    def logical_not(self):
        return SampledData(jnp.logical_not(self._data))
    def logical_xor(self, other):
        return SampledData(jnp.logical_xor(self._data, self._extract_data(other)))

    # Truth value testing
    def isfinite(self):
        return SampledData(jnp.isfinite(self._data))
    def isinf(self):
        return SampledData(jnp.isinf(self._data))
    def isnan(self):
        return SampledData(jnp.isnan(self._data))
    def isposinf(self):
        return SampledData(jnp.isposinf(self._data))
    def isneginf(self):
        return SampledData(jnp.isneginf(self._data))
    def iscomplex(self):
        return SampledData(jnp.iscomplexobj(self._data))
    def isreal(self):
        return SampledData(jnp.isrealobj(self._data))

    # Array comparison
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return jnp.allclose(self._data, self._extract_data(other), rtol=rtol, atol=atol, equal_nan=equal_nan)

    # ================== LINEAR ALGEBRA ==================

    def dot(self, other):
        return self._wrap_result(jnp.dot(self._data, self._extract_data(other)))
    def vdot(self, other):
        return self._wrap_result(jnp.vdot(self._data, self._extract_data(other)))
    def inner(self, other):
        return self._wrap_result(jnp.inner(self._data, self._extract_data(other)))
    def outer(self, other):
        return self._wrap_result(jnp.outer(self._data, self._extract_data(other)))
    def tensordot(self, other, axes=2):
        return self._wrap_result(jnp.tensordot(self._data, self._extract_data(other), axes=axes))
    def einsum(self, *operands, **kwargs):
        return self._wrap_result(jnp.einsum(self._data, *[self._extract_data(op) for op in operands], **kwargs))
    def linalg_det(self):
        return self._wrap_result(jnp.linalg.det(self._data))
    def linalg_slogdet(self):
        sign, logdet = jnp.linalg.slogdet(self._data)
        return self._wrap_result(sign), self._wrap_result(logdet)
    def linalg_eig(self):
        w, v = jnp.linalg.eig(self._data)
        return self._wrap_result(w), self._wrap_result(v)
    def linalg_eigh(self):
        w, v = jnp.linalg.eigh(self._data)
        return self._wrap_result(w), self._wrap_result(v)
    def linalg_eigvals(self):
        return self._wrap_result(jnp.linalg.eigvals(self._data))
    def linalg_eigvalsh(self):
        return self._wrap_result(jnp.linalg.eigvalsh(self._data))
    def linalg_inv(self):
        return self._wrap_result(jnp.linalg.inv(self._data))
    def linalg_lstsq(self, b, rcond=None):
        x, res, rank, s = jnp.linalg.lstsq(self._data, self._extract_data(b), rcond=rcond)
        return self._wrap_result(x), self._wrap_result(res), self._wrap_result(rank), self._wrap_result(s)
    def linalg_matrix_power(self, n):
        return self._wrap_result(jnp.linalg.matrix_power(self._data, n))
    def linalg_matrix_rank(self):
        return self._wrap_result(jnp.linalg.matrix_rank(self._data))
    def linalg_norm(self, ord=None, axis=None, keepdims=False):
        return self._wrap_result(jnp.linalg.norm(self._data, ord=ord, axis=axis, keepdims=keepdims))
    def linalg_pinv(self):
        return self._wrap_result(jnp.linalg.pinv(self._data))
    def linalg_qr(self, mode='reduced'):
        q, r = jnp.linalg.qr(self._data, mode=mode)
        return self._wrap_result(q), self._wrap_result(r)
    def linalg_solve(self, b):
        return self._wrap_result(jnp.linalg.solve(self._data, self._extract_data(b)))
    def linalg_svd(self, full_matrices=True, compute_uv=True):
        if compute_uv:
            u, s, vh = jnp.linalg.svd(self._data, full_matrices=full_matrices, compute_uv=compute_uv)
            return self._wrap_result(u), self._wrap_result(s), self._wrap_result(vh)
        else:
            s = jnp.linalg.svd(self._data, full_matrices=full_matrices, compute_uv=compute_uv)
            return self._wrap_result(s)

    # ================== FAST FOURIER TRANSFORM (FFT) ==================

    def fft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.fft(self._data, n=n, axis=axis, norm=norm))
    def ifft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.ifft(self._data, n=n, axis=axis, norm=norm))
    def fft2(self, s=None, axes=(-2, -1), norm=None):
        return self._wrap_result(jnp.fft.fft2(self._data, s=s, axes=axes, norm=norm))
    def ifft2(self, s=None, axes=(-2, -1), norm=None):
        return self._wrap_result(jnp.fft.ifft2(self._data, s=s, axes=axes, norm=norm))
    def fftn(self, s=None, axes=None, norm=None):
        return self._wrap_result(jnp.fft.fftn(self._data, s=s, axes=axes, norm=norm))
    def ifftn(self, s=None, axes=None, norm=None):
        return self._wrap_result(jnp.fft.ifftn(self._data, s=s, axes=axes, norm=norm))
    def rfft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.rfft(self._data, n=n, axis=axis, norm=norm))
    def irfft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.irfft(self._data, n=n, axis=axis, norm=norm))
    def rfft2(self, s=None, axes=(-2, -1), norm=None):
        return self._wrap_result(jnp.fft.rfft2(self._data, s=s, axes=axes, norm=norm))
    def irfft2(self, s=None, axes=(-2, -1), norm=None):
        return self._wrap_result(jnp.fft.irfft2(self._data, s=s, axes=axes, norm=norm))
    def rfftn(self, s=None, axes=None, norm=None):
        return self._wrap_result(jnp.fft.rfftn(self._data, s=s, axes=axes, norm=norm))
    def irfftn(self, s=None, axes=None, norm=None):
        return self._wrap_result(jnp.fft.irfftn(self._data, s=s, axes=axes, norm=norm))
    def hfft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.hfft(self._data, n=n, axis=axis, norm=norm))
    def ihfft(self, n=None, axis=-1, norm=None):
        return self._wrap_result(jnp.fft.ihfft(self._data, n=n, axis=axis, norm=norm))
    def fftfreq(self, d=1.0):
        return self._wrap_result(jnp.fft.fftfreq(self.size, d=d))
    def rfftfreq(self, d=1.0):
        return self._wrap_result(jnp.fft.rfftfreq(self.size, d=d))
    def fftshift(self, axes=None):
        return self._wrap_result(jnp.fft.fftshift(self._data, axes=axes))
    def ifftshift(self, axes=None):
        return self._wrap_result(jnp.fft.ifftshift(self._data, axes=axes))