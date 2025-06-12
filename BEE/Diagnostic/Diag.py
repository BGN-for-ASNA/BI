  
import arviz as az
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt


class diag:

    def __init__(self, sampler):
        """Initialize the diagnostic class. Currently empty but can be extended for initialization needs."""
        self.sampler = sampler

    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(self):
        """Convert the sampler output to an arviz trace object.
        
        This method prepares the trace for use with arviz diagnostic tools.
        
        Returns:
            self.trace: The arviz trace object containing the diagnostic data
        """
        self.trace = az.from_numpyro(self.sampler)
        self.priors_name = list(self.trace['posterior'].data_vars.keys())
        return self.trace

    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        """Calculate summary statistics for the posterior distribution.
        
        Args:
            round_to (int): Number of decimal places to round results
            kind (str): Type of statistics to compute (default: "stats")
            hdi_prob (float): Probability for highest posterior density interval
            *args, **kwargs: Additional arguments for arviz.summary
            
        Returns:
            pd.DataFrame: Summary statistics of the posterior distribution
        """        
        if self.trace is None:
            self.to_az()
        self.tab_summary = az.summary(self.trace , round_to=round_to, kind=kind, hdi_prob=hdi_prob, *args, **kwargs)
        return self.tab_summary 

    def plot_trace(self, var_names= None, kind="rank_bars", *args, **kwargs): 
        """Create a trace plot for visualizing MCMC diagnostics.
        
        Args:
            var_names (list): List of variable names to include
            kind (str): Type of plot (default: "rank_bars")
            *args, **kwargs: Additional arguments for arviz.plot_trace
            
        Returns:
            plot: The trace plot object
        """        
        if self.trace is None:
            self.to_az()
        self.plot_trace = az.plot_trace(self.trace, var_names=self.priors_name, kind=kind, *args, **kwargs)
        return self.plot_trace 

    def prior_dist(self, N = 100):
        """Visualize prior distributions compared to log probability.
        
        Args:
            N (int): Number of samples to draw from priors
            
        Returns:
            fig: Matplotlib figure containing the prior distribution plots
        """        
        samples = self.sample.sample(N)
        prob = self.log_prob(samples)
        post_df = self.model_output_to_df(samples)

        fig, axs = plt.subplots(ncols=post_df.shape[1])
        for a in range(post_df.shape[1]-1): 
                sns.histplot(post_df.iloc[:,a], 
                     kde=True, stat="density",
                     edgecolor=(1, 1, 1, .4), 
                     ax=axs[a]).set_title(post_df.columns[a]) 

        sns.histplot(list(prob.numpy()), kde=True, stat="density",
                     edgecolor=(1, 1, 1, .4)).set_title("${\\rm logit}$")
        self.plot_priors = fig
        return fig

    def posterior(self, figsize=(8, 4)):
        """Create posterior distribution plots.
        
        Args:
            figsize (tuple): Size of the figure (width, height)
            
        Returns:
            fig: Matplotlib figure containing posterior plots
        """        
        posterior, axes = plt.subplots(1, len(self.priors_name), figsize=(figsize))
        axes = az.plot_posterior(self.trace , var_names=self.priors_name, ax=axes)
        axes.flatten()[0].get_figure() 
        self.plot_posterior = posterior

    def autocor(self, *args, **kwargs):
        """Plot autocorrelation of the MCMC chains.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.plot_autocorr
            
        Returns:
            fig: Autocorrelation plot
        """        
        self.autocor = az.plot_autocorr(self.trace , var_names=self.priors_name, *args, **kwargs)

    def traces(self, *args, **kwargs):
        """Create trace plots for MCMC chains.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.plot_trace
            
        Returns:
            fig: Trace plots
        """        
        self.traces =  az.plot_trace(self.trace, compact=False, *args, **kwargs)

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
    
    def forest(self, list = None, kind = "ridgeplot", ess = True, var_names = None, *args, **kwargs):
        """Create a forest plot of estimated values.
        
        Args:
            list: Data to plot (default: self.trace)
            kind (str): Type of plot (default: "ridgeplot")
            ess (bool): Include effective sample size
            var_names (list): Variables to include
            *args, **kwargs: Additional arguments for arviz.plot_forest
            
        Returns:
            fig: Forest plot
        """        
        if var_names is None:
            var_names = self.priors_name
        if list is None:
            list = self.trace
        self.forest = az.plot_forest(list, var_names = var_names,  kind = kind, ess = ess, *args, **kwargs)
        return self.forest
    
    def waic(self, *args, **kwargs):
        """Calculate WAIC (Watanabe-Akaike information criterion).
        
        Args:
            *args, **kwargs: Additional arguments for arviz.waic
            
        Returns:
            waic: WAIC result
        """        
        self.waic = az.waic(self.trace, *args, **kwargs)
        return self.waic
    
    def compare(self, dict, *args, **kwargs):
        """Compare models using WAIC or LOOIC.
        
        Args:
            dict: Dictionary of models to compare
            *args, **kwargs: Additional arguments for arviz.compare
            
        Returns:
            comp: Comparison result
        """        
        self.comparison = az.compare(dict, *args, **kwargs)
        return self.comparison 

    def rhat(self, *args, **kwargs):
        """Calculate R-hat statistics for convergence.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.rhat
            
        Returns:
            rhat: R-hat values
        """        
        self.rhat = az.rhat(self.trace, *args, **kwargs)
        return self.rhat 

    def  ess(self, *args, **kwargs):
        """Calculate effective sample size (ESS).
        
        Args:
            *args, **kwargs: Additional arguments for arviz.ess
            
        Returns:
            ess: Effective sample sizes
        """        
        self.ess = az.ess(self.trace, *args, **kwargs)
        return self.ess 

    def pair(self, var_names = None, 
                  kind=["scatter", "kde"],
                  kde_kwargs={"fill_last": False},
                  marginals=True,
                  point_estimate="median",
                  figsize=(11.5, 5),
                  *args, **kwargs):
        """Create pairwise plots of the posterior distribution.
        
        Args:
            var_names (list): Variables to include
            kind (list): Type of plots ("scatter" and/or "kde")
            kde_kwargs (dict): Additional arguments for KDE plots
            marginals (bool): Include marginal distributions
            point_estimate (str): Point estimate to plot
            figsize (tuple): Size of the figure
            *args, **kwargs: Additional arguments for arviz.plot_pair
            
        Returns:
            fig: Pair plot
        """                  
        if var_names is None:
            var_names = self.priors_name
        self.pair_plot = az.plot_pair(self.trace, var_names = var_names,                   
                                      kind=kind,
                                      kde_kwargs=kde_kwargs,
                                      marginals=marginals,
                                      point_estimate=point_estimate,
                                      figsize=figsize,
                                      *args, **kwargs)
        return self.pair_plot   
    
    def density(self, var_names=None, shade=0.2, *args, **kwargs):
        """Plot density plots of the posterior distribution.
        
        Args:
            var_names (list): Variables to include
            shade (float): Transparency of the filled area
            *args, **kwargs: Additional arguments for arviz.plot_density
            
        Returns:
            fig: Density plots
        """        
        if var_names is None:
            var_names = self.priors_name

        self.density = az.plot_density(
                            self.trace,
                            var_names=var_names,
                            shade=shade,
                            *args, **kwargs
                        )
        return self.density
    
    def plot_ess(self,):
        """Plot evolution of effective sample size across iterations.
        
        Returns:
            fig: ESS evolution plot
        """        
        self.ess_plot = az.plot_ess(self.trace, var_names=self.priors_name, kind="evolution")
        return self.ess_plot
    
    def model_checks(self):
        """Perform comprehensive model diagnostics.
        
        Creates various diagnostic plots including:
        - Posterior distributions
        - Autocorrelation plots
        - Trace plots
        - Rank plots
        - Forest plots
        
        Stores plots in instance variables:
        self.plot_posterior, self.autocor, self.traces, self.rank, self.forest
        """        
        params = self.priors_name
        posterior = self.hmc_posterior
        trace = self.trace 

        posterior, axes = plt.subplots(1, len(params), figsize=(8, 4))
        axes = az.plot_posterior(trace, var_names=params, ax=axes)
        axes.flatten()[0].get_figure() 

        autocor = az.plot_autocorr(trace, var_names=params)

        traces = az.plot_trace(trace, compact=False)

        rank, axes = plt.subplots(1, len(params))
        az.plot_rank(trace, var_names=params, ax=axes)

        forest = az.plot_forest(trace, var_names = params)

        #summary = az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

        self.plot_posterior = posterior
        self.autocor = autocor
        self.traces = traces
        self.rank = rank
        self.forest = forest

