import arviz as az

import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import pandas as pd
import numpy as np
import jax.numpy as jnp
import itertools
import scipy.stats as stats
import re
from BI.Utils.ImportManager import LazyImporter
class diagWIP():
    """
    The diag class serves as a comprehensive toolkit for diagnosing and visualizing the results of Bayesian models, 
    particularly those fitted using MCMC samplers like NumPyro. It is built to provide interactive plotting 
    functionalities using Plotly and operates directly on a dictionary of posterior samples. Statistical
    diagnostics like R-hat and ESS are computed directly using JAX.
    """

    def __init__(self, sampler):
        """
        Initialize the diagnostic class.

        Args:
            sampler: A fitted NumPyro MCMC sampler object.
        """
        self.sampler = sampler
        # Get samples with chain information preserved
        self.posterior_samples = sampler.get_samples(group_by_chain=True)
        self.priors_name = list(self.posterior_samples.keys())
        # Determine the number of chains from the shape of the first parameter
        if self.priors_name:
            self.num_chains = self.posterior_samples[self.priors_name[0]].shape[0]
            self.colors = pcolors.qualitative.Plotly
        else:
            self.num_chains = 0
            self.colors = []

        go=importer.schedule_import("plotly.graph_objects", "go")
        px=importer.schedule_import("plotly.express", "px")
        ff=importer.schedule_import("plotly.figure_factory", "ff")
        n_colors=importer.schedule_import("plotly.colors", "n_colors")
        make_subplots=importer.schedule_import("plotly.subplots", "make_subplots")
        sns=importer.schedule_import("seaborn", "sns")
        plt=importer.schedule_import("matplotlib.pyplot", "plt")
    #
    #  Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(self, backend="numpyro", sample_stats_name=['target_log_prob','log_accept_ratio','has_divergence','energy']):
        """Convert the sampler output to an arviz trace object.
        
        This method prepares the trace for use with arviz diagnostic tools.
        
        Returns:
            self.trace: The arviz trace object containing the diagnostic data
        """
        if backend == "numpyro":
            self.trace = az.from_numpyro(self.sampler)
            self.priors_name = list(self.trace['posterior'].data_vars.keys())
            return self.trace
        
        elif backend == "tfp":
            var_names= list(self.sampler.model_info.keys())
            sample_stats = {k:jnp.transpose(v) for k, v in zip(sample_stats_name, self.sampler.sample_stats)}
            trace = {}
            #First dim is the number of chains
            #Second dim is the number of sampling
            #The rest is the shape of the object
            for name, samp in zip(var_names, self.sampler.posterior):
                trace[name] = samp
    
            self.trace = az.from_dict(posterior=trace, sample_stats=sample_stats)
            self.priors_name = var_names
            return self.trace
    
    # --- Statistical Diagnostics ---

    def loo(self, pointwise=None, var_name=None, reff=None, scale=None):
        """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

        Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
        importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
        standard error and the effective number of parameters. Read more theory here
        https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1507.02646

        Parameters
        ----------
        pointwise: bool, optional
            If True the pointwise predictive accuracy will be returned. Defaults to
            ``stats.ic_pointwise`` rcParam.
        var_name : str, optional
            The name of the variable in log_likelihood groups storing the pointwise log
            likelihood data to use for loo computation.
        reff: float, optional
            Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
            of actual samples. Computed from trace by default.
        scale: str
            Output scale for loo. Available options are:

            - ``log`` : (default) log-score
            - ``negative_log`` : -1 * log-score
            - ``deviance`` : -2 * log-score

            A higher log-score (or a lower deviance or negative log_score) indicates a model with
            better predictive accuracy.

        Returns
        -------
        ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_loo: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_loo: effective number of parameters
        n_samples: number of samples
        n_data_points: number of data points
        warning: bool
            True if the estimated shape parameter of Pareto distribution is greater than
            ``good_k``.
        loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
                only if pointwise=True
        pareto_k: array of Pareto shape values, only if pointwise True
        scale: scale of the elpd
        good_k: For a sample size S, the thresold is compute as min(1 - 1/log10(S), 0.7)

            The returned object has a custom print method that overrides pd.Series method.
        """
        return az.loo(self.sampler, pointwise=pointwise, var_name=var_name, reff=reff, scale=scale)
    
    def WAIC(self,  pointwise=None, var_name=None, scale=None, dask_kwargs=None):
        """
        Compute the widely applicable information criterion.

        Estimates the expected log pointwise predictive density (elpd) using WAIC. Also calculates the
        WAIC's standard error and the effective number of parameters.
        Read more theory here https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1004.2316

        Parameters
        ----------
        pointwise: bool
            If True the pointwise predictive accuracy will be returned. Defaults to
            ``stats.ic_pointwise`` rcParam.
        var_name : str, optional
            The name of the variable in log_likelihood groups storing the pointwise log
            likelihood data to use for waic computation.
        scale: str
            Output scale for WAIC. Available options are:

            - `log` : (default) log-score
            - `negative_log` : -1 * log-score
            - `deviance` : -2 * log-score

            A higher log-score (or a lower deviance or negative log_score) indicates a model with
            better predictive accuracy.
        dask_kwargs : dict, optional
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

        Returns
        -------
        ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_waic: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_waic: effective number parameters
        n_samples: number of samples
        n_data_points: number of data points
        warning: bool
            True if posterior variance of the log predictive densities exceeds 0.4
        waic_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
                only if pointwise=True
        scale: scale of the elpd

            The returned object has a custom print method that overrides pd.Series method.
        """
        return az.waic(self.sampler, pointwise=pointwise, var_name=var_name, scale=scale, dask_kwargs=dask_kwargs)

    @staticmethod
    def compare(compare_dict, ic=None, method='stacking', b_samples=1000, alpha=1, seed=None, scale=None, var_name=None):
        r"""Compare models based on  their expected log pointwise predictive density (ELPD).

        The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
        cross-validation (LOO) or using the widely applicable information criterion (WAIC).
        We recommend loo. Read more theory here - in a paper by some of the
        leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353
    
        Parameters
        ----------
        compare_dict: dict of {str: InferenceData or ELPDData}
            A dictionary of model names and :class:`arviz.InferenceData` or ``ELPDData``.
        ic: str, optional
            Method to estimate the ELPD, available options are "loo" or "waic". Defaults to
            ``rcParams["stats.information_criterion"]``.
        method: str, optional
            Method used to estimate the weights for each model. Available options are:
    
            - 'stacking' : stacking of predictive distributions.
            - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
              weighting. The weights are stabilized using the Bayesian bootstrap.
            - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
              weighting, without Bootstrap stabilization (not recommended).
    
            For more information read https://arxiv.org/abs/1704.02030
        b_samples: int, optional default = 1000
            Number of samples taken by the Bayesian bootstrap estimation.
            Only useful when method = 'BB-pseudo-BMA'.
            Defaults to ``rcParams["stats.ic_compare_method"]``.
        alpha: float, optional
            The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap. Only
            useful when method = 'BB-pseudo-BMA'. When alpha=1 (default), the distribution is uniform
            on the simplex. A smaller alpha will keeps the final weights more away from 0 and 1.
        seed: int or np.random.RandomState instance, optional
            If int or RandomState, use it for seeding Bayesian bootstrap. Only
            useful when method = 'BB-pseudo-BMA'. Default None the global
            :mod:`numpy.random` state is used.
        scale: str, optional
            Output scale for IC. Available options are:
    
            - `log` : (default) log-score (after Vehtari et al. (2017))
            - `negative_log` : -1 * (log-score)
            - `deviance` : -2 * (log-score)
    
            A higher log-score (or a lower deviance) indicates a model with better predictive
            accuracy.
        var_name: str, optional
            If there is more than a single observed variable in the ``InferenceData``, which
            should be used as the basis for comparison.
    
        Returns
        -------
        A DataFrame, ordered from best to worst model (measured by the ELPD).
        The index reflects the key with which the models are passed to this function. The columns are:
        rank: The rank-order of the models. 0 is the best.
        elpd: ELPD estimated either using (PSIS-LOO-CV `elpd_loo` or WAIC `elpd_waic`).
            Higher ELPD indicates higher out-of-sample predictive fit ("better" model).
            If `scale` is `deviance` or `negative_log` smaller values indicates
            higher out-of-sample predictive fit ("better" model).
        pIC: Estimated effective number of parameters.
        elpd_diff: The difference in ELPD between two models.
            If more than two models are compared, the difference is computed relative to the
            top-ranked model, that always has a elpd_diff of 0.
        weight: Relative weight for each model.
            This can be loosely interpreted as the probability of each model (among the compared model)
            given the data. By default the uncertainty in the weights estimation is considered using
            Bayesian bootstrap.
        SE: Standard error of the ELPD estimate.
            If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
        dSE: Standard error of the difference in ELPD between each model and the top-ranked model.
            It's always 0 for the top-ranked model.
        warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
            This could be indication of WAIC/LOO starting to fail see
            http://arxiv.org/abs/1507.04544 for details.
        scale: Scale used for the ELPD.

        References
        ----------
        .. [1] Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using
            leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017)
            see https://doi.org/10.1007/s11222-016-9696-4
        """
        return az.compare(compare_dict = compare_dict, ic=ic, method='stacking', b_samples=b_samples, alpha=alpha, seed=seed, scale=None, var_name=var_name)

    @staticmethod
    def plot_compare(
        comp_df,
        insample_dev=False,
        plot_standard_error=True,
        plot_ic_diff=False,
        order_by_rank=True,
        legend=False,
        title=True,
        figsize=None,
        textsize=None,
        labeller=None,
        plot_kwargs=None,
        ax=None,
        backend=None,
        backend_kwargs=None,
        show=None,
    ):
        r"""Summary plot for model comparison.

        Models are compared based on their expected log pointwise predictive density (ELPD).
        This plot is in the style of the one used in [2]_. Chapter 6 in the first edition
        or 7 in the second.

        Notes
        -----
        The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
        cross-validation (LOO) or using the widely applicable information criterion (WAIC).
        We recommend LOO in line with the work presented by [1]_.

        Parameters
        ----------
        comp_df : pandas.DataFrame
            Result of the :func:`arviz.compare` method.
        insample_dev : bool, default False
            Plot in-sample ELPD, that is the value of the information criteria without the
            penalization given by the effective number of parameters (p_loo or p_waic).
        plot_standard_error : bool, default True
            Plot the standard error of the ELPD.
        plot_ic_diff : bool, default False
            Plot standard error of the difference in ELPD between each model
            and the top-ranked model.
        order_by_rank : bool, default True
            If True ensure the best model is used as reference.
        legend : bool, default False
            Add legend to figure.
        figsize : (float, float), optional
            If `None`, size is (6, num of models) inches.
        title : bool, default True
            Show a tittle with a description of how to interpret the plot.
        textsize : float, optional
            Text size scaling factor for labels, titles and lines. If `None` it will be autoscaled based
            on `figsize`.
        labeller : Labeller, optional
            Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
            Read the :ref:`label_guide` for more details and usage examples.
        plot_kwargs : dict, optional
            Optional arguments for plot elements. Currently accepts 'color_ic',
            'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
            'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
        ax : matplotlib_axes or bokeh_figure, optional
            Matplotlib axes or bokeh figure.
        backend : {"matplotlib", "bokeh"}, default "matplotlib"
            Select plotting backend.
        backend_kwargs : bool, optional
            These are kwargs specific to the backend being used, passed to
            :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
            For additional documentation check the plotting method of the backend.
        show : bool, optional
            Call backend show function.

        Returns
        -------
        axes : matplotlib_axes or bokeh_figure

        See Also
        --------
        plot_elpd : Plot pointwise elpd differences between two or more models.
        compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
        loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
        waic : Compute the widely applicable information criterion.

        References
        ----------
        .. [1] Vehtari et al. (2016). Practical Bayesian model evaluation using leave-one-out
           cross-validation and WAIC https://arxiv.org/abs/1507.04544

        .. [2] McElreath R. (2022). Statistical Rethinking A Bayesian Course with Examples in
           R and Stan, Second edition, CRC Press.



        """
        return az.plot_compare(comp_df, insample_dev=insample_dev, plot_standard_error=plot_standard_error, plot_ic_diff=plot_ic_diff, order_by_rank=order_by_rank, legend=legend, title=title, figsize=figsize, textsize=textsize, labeller=labeller, plot_kwargs=plot_kwargs, ax=ax, backend=backend, backend_kwargs=backend_kwargs, show=show)

    def rhat(self, *args, **kwargs):
        """Calculate R-hat statistics for convergence.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.rhat
            
        Returns:
            rhat: R-hat values
        """        
        self.rhat = az.rhat(self.trace, *args, **kwargs)
        return self.rhat 

    def ess(self, *args, **kwargs):
        """Calculate effective sample size (ESS).
        
        Args:
            *args, **kwargs: Additional arguments for arviz.ess
            
        Returns:
            ess: Effective sample sizes
        """        
        self.ess = az.ess(self.trace, *args, **kwargs)
        return self.ess 

    # --- Plotting Functions arviz dependent---
    def plot_ess(self,):
        """Plot evolution of effective sample size across iterations.
        
        Returns:
            fig: ESS evolution plot
        """        
        self.ess_plot = az.plot_ess(self.trace, var_names=self.priors_name, kind="evolution")
    def rank(self, *args, **kwargs):
        """Create rank plots for MCMC chains.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.plot_rank
            
        Returns:
            fig: Rank plots
        """        
        rank, axes = plt.subplots(1, len( self.priors_name))
        az.plot_rank(self.trace , var_names= self.priors_name, ax=axes, *args, **kwargs)
        self.rank = rank
    
    # --- Plotting Functions  plotly dependent---
    
    def summary(self, round_to=2, hdi_prob=0.89):
        # This already uses numpy, so it's fine
        summary_stats = {}
        for var_name, samples in self.posterior_samples.items():
            all_chain_samples = samples.flatten()
            mean = np.mean(all_chain_samples)
            median = np.median(all_chain_samples)
            std = np.std(all_chain_samples)
            hdi = az.hdi(np.array(all_chain_samples), hdi_prob=hdi_prob) # az.hdi is just a numpy func
            summary_stats[var_name] = {'mean': mean, 'median': median, 'std': std,
                f'hdi_{hdi_prob*100}%_lower': hdi[0], f'hdi_{hdi_prob*100}%_upper': hdi[1]}
        self.tab_summary = pd.DataFrame(summary_stats).T.round(round_to)
        return self.tab_summary
    
    def pair(self, var_names=None, colorscale="Viridis", max_points=1000, 
             point_color='rgba(40, 150, 200, 0.4)'):
        if var_names is None: var_names = self.priors_name
        n_vars = len(var_names)
        df = pd.DataFrame({k: self.posterior_samples[k].flatten() for k in var_names})
        plot_df = df.sample(n=max_points, random_state=42) if len(df) > max_points else df
        fig = make_subplots(rows=n_vars, cols=n_vars, horizontal_spacing=0.03, vertical_spacing=0.03)

        for i in range(n_vars):
            for j in range(n_vars):
                var1, var2 = var_names[i], var_names[j]
                if i == j:
                    fig.add_trace(go.Histogram(x=df[var1], name=f'Hist {var1}', 
                                               marker_color='#440154'), row=i+1, col=j+1)
                elif i > j:
                    fig.add_trace(go.Histogram2dContour(x=df[var2], y=df[var1], colorscale=colorscale,
                        showscale=False, name='Density', contours=dict(coloring='lines'), line=dict(width=1)
                    ), row=i+1, col=j+1)
                    fig.add_trace(go.Scatter(x=plot_df[var2], y=plot_df[var1], mode='markers', name='Samples',
                        marker=dict(size=3, color=point_color)), row=i+1, col=j+1)
                    median_x, median_y = df[var2].median(), df[var1].median()
                    fig.add_trace(go.Scatter(x=[median_x], y=[median_y], mode='markers', name='Median',
                        marker=dict(symbol='square', color='black', size=8)), row=i+1, col=j+1)
        
        fig.update_layout(title_text="Pair Plot: Histograms, Density, and Samples",
            height=250 * n_vars, width=250 * n_vars, showlegend=False, plot_bgcolor='white')
        
        for i in range(n_vars):
             fig.update_yaxes(title_text=var_names[i], row=i+1, col=1, showline=True, linewidth=1, linecolor='black', mirror=True)
        for j in range(n_vars):
             fig.update_xaxes(title_text=var_names[j], row=n_vars, col=j+1, showline=True, linewidth=1, linecolor='black', mirror=True)


        return fig
    
    # ... other plotting functions (plot_trace, posterior, autocor, forest, density, model_checks) remain the same ...
    def plot_trace(self, var_names=None):
        if var_names is None:
            var_names = self.priors_name
        fig = make_subplots(rows=len(var_names), cols=2, 
                            subplot_titles=[f'{var} Trace' for var in var_names] + 
                                         [f'{var} Rank (All Chains)' for var in var_names])
        for i, var in enumerate(var_names):
            samples_per_chain = self.posterior_samples[var]
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(go.Scatter(y=samples_per_chain[chain_idx], mode='lines', 
                                         name=f'{var} chain {chain_idx}', 
                                         legendgroup=f'chain{chain_idx}',
                                         line=dict(color=color), showlegend=(i==0)), 
                              row=i+1, col=1)
            all_samples = samples_per_chain.flatten()
            fig.add_trace(go.Histogram(x=all_samples, name=f'{var} rank', nbinsx=30,
                                       marker_color='grey'), row=i+1, col=2)
        fig.update_layout(height=300*len(var_names), title_text="Trace and Rank Plots")

        return fig

    def posterior(self, var_names=None, figsize=(800, 400)):
        if var_names is None:
            var_names = self.priors_name
        fig = make_subplots(rows=1, cols=len(var_names), subplot_titles=var_names)
        for i, var in enumerate(var_names):
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(go.Histogram(x=self.posterior_samples[var][chain_idx], 
                                           name=f'Chain {chain_idx}',
                                           legendgroup=f'chain{chain_idx}',
                                           marker_color=color,
                                           showlegend=(i==0),
                                           opacity=0.6,
                                           nbinsx=50), row=1, col=i+1)
        fig.update_layout(title_text="Posterior Distributions (Overlaid Chains)", 
                          width=figsize[0], height=figsize[1], barmode='overlay')

        return fig

    def autocor(self, var_names=None):
        if var_names is None:
            var_names = self.priors_name
        fig = make_subplots(rows=len(var_names), cols=1, subplot_titles=[f"Autocorrelation of {var}" for var in var_names])
        for i, var in enumerate(var_names):
            for chain_idx in range(self.num_chains):
                samples = self.posterior_samples[var][chain_idx]
                autocorr = [1.0] + [np.corrcoef(samples[:-t], samples[t:])[0, 1] for t in range(1, 40)]
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(go.Bar(y=autocorr, name=f'Chain {chain_idx}', 
                                     legendgroup=f'chain{chain_idx}',
                                     marker_color=color,
                                     showlegend=(i==0)), 
                              row=i+1, col=1)
        fig.update_layout(height=250*len(var_names), title_text="Autocorrelation Plots by Chain", barmode='group')

        return fig

    def forest(self, var_names=None, hdi_prob=0.95):
        if var_names is None:
            var_names = self.priors_name
        y_labels, means, hdi_lower, hdi_upper = [], [], [], []
        for var in var_names:
            all_samples = self.posterior_samples[var].flatten()
            y_labels.append(var)
            means.append(np.mean(all_samples))
            hdi = az.hdi(np.array(all_samples), hdi_prob=hdi_prob)
            hdi_lower.append(hdi[0])
            hdi_upper.append(hdi[1])
        fig = go.Figure()
        error_upper = np.array(hdi_upper) - np.array(means)
        error_lower = np.array(means) - np.array(hdi_lower)
        fig.add_trace(go.Scatter(
            x=means, y=y_labels,
            error_x=dict(type='data', symmetric=False, array=error_upper, arrayminus=error_lower),
            mode='markers', marker=dict(color='blue'), name='Mean and HDI'
        ))
        fig.update_layout(title_text=f'Forest Plot (Mean and {hdi_prob*100}% HDI across all chains)')

        return fig

    def density(self, var_names=None, shade=0.4):
        if var_names is None:
            var_names = self.priors_name
        fig = make_subplots(rows=len(var_names), cols=1, subplot_titles=[f"Density of {var}" for var in var_names])
        for i, var in enumerate(var_names):
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                rgb_color = pcolors.hex_to_rgb(color)
                fill_color = f'rgba({rgb_color[0]},{rgb_color[1]},{rgb_color[2]},{shade})'
                chain_samples = self.posterior_samples[var][chain_idx]
                with sns.plotting_context(rc={"figure.figsize": (1, 1)}):
                    kde_plot = sns.kdeplot(chain_samples)
                    kde = kde_plot.get_lines()[0].get_data()
                    plt.close()
                fig.add_trace(go.Scatter(x=kde[0], y=kde[1], fill='tozeroy', mode='lines',
                                         name=f'Chain {chain_idx}', legendgroup=f'chain{chain_idx}',
                                         showlegend=(i==0), fillcolor=fill_color, line_color=color),
                              row=i+1, col=1)
        fig.update_layout(height=300*len(var_names), title_text="Density Plots (Overlaid Chains)")

        return fig

    def model_checks(self):
        """Perform comprehensive model diagnostics with interactive plots."""
        print("Displaying Posterior Plots (Overlaid Chains):")
        self.posterior().show()
        print("\nDisplaying Autocorrelation Plots (by Chain):")
        self.autocor().show()
        print("\nDisplaying Trace Plots (by Chain):")
        self.plot_trace().show()
        print("\nDisplaying Forest Plot (All Chains Combined):")
        self.forest().show()
        print("\nDisplaying Pair Plot (All Chains Combined):")
        self.pair().show()