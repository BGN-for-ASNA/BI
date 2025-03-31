  
import arviz as az
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt


class diag:

    def __init__(self):
        pass

    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(self):
        """Convert the sampler output to an arviz trace object.
        
        This method prepares the trace for use with arviz diagnostic tools.
        
        Returns:
            self.trace: The arviz trace object.
        """
        self.trace = az.from_numpyro(self.sampler)
        self.priors_name = list(self.trace['posterior'].data_vars.keys())
        return self.trace

    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        
        if self.trace is None:
            self.to_az()
        self.tab_summary = az.summary(self.trace , round_to=round_to, kind=kind, hdi_prob=hdi_prob, *args, **kwargs)
        return self.tab_summary 

    def diag_plot_trace(self, var_names= None, kind="rank_bars", *args, **kwargs): 
        if self.trace is None:
            self.to_az()
        self.plot_trace = az.plot_trace(self.trace, var_names=self.priors_name, kind=kind, *args, **kwargs)
        return self.plot_trace 

    def diag_prior_dist(self, N = 100):
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

    def diag_posterior(self):
        posterior, axes = plt.subplots(1, len(self.priors_name), figsize=(8, 4))
        axes = az.plot_posterior(self.trace , var_names=self.priors_name, ax=axes)
        axes.flatten()[0].get_figure() 
        self.plot_posterior = posterior

    def diag_autocor(self, *args, **kwargs):
        self.autocor = az.plot_autocorr(self.trace , var_names=self.priors_name, *args, **kwargs)

    def diag_traces(self, *args, **kwargs):
        self.traces =  az.plot_trace(self.trace, compact=False, *args, **kwargs)

    def diag_rank(self, *args, **kwargs):
        rank, axes = plt.subplots(1, len( self.priors_name))
        az.plot_rank(self.trace , var_names= self.priors_name, ax=axes, *args, **kwargs)
        self.rank = rank
    
    def diag_forest(self, list = None, kind = "ridgeplot", ess = True, var_names = None, *args, **kwargs):
        if var_names is None:
            var_names = self.priors_name
        if list is None:
            list = self.trace
        self.forest = az.plot_forest(list, var_names = var_names,  kind = kind, ess = ess, *args, **kwargs)
        return self.forest
    
    def diag_waic(self, *args, **kwargs):
        self.waic = az.waic(self.trace, *args, **kwargs)
        return self.waic
    
    def diag_compare(self, dict, *args, **kwargs):
        self.comparison = az.compare(dict, *args, **kwargs)
        return self.comparison 

    def diag_rhat(self, *args, **kwargs):
        self.rhat = az.rhat(self.trace, *args, **kwargs)
        return self.rhat 

    def  diag_ess(self, *args, **kwargs):
        self.ess = az.ess(self.trace, *args, **kwargs)
        return self.ess 

    def diag_pair(self, var_names = None, 
                  kind=["scatter", "kde"],
                  kde_kwargs={"fill_last": False},
                  marginals=True,
                  point_estimate="median",
                  figsize=(11.5, 5),
                  *args, **kwargs):
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
    
    def diag_density(self, var_names=None, shade=0.2, *args, **kwargs):
        if var_names is None:
            var_names = self.priors_name

        self.density = az.plot_density(
                            self.trace,
                            var_names=var_names,
                            shade=shade,
                            *args, **kwargs
                        )
        return self.density
    
    def diag_plot_ess(self,):
        self.ess_plot = az.plot_ess(self.trace, var_names=self.priors_name, kind="evolution")
        return self.ess_plot
    
    def model_checks(self):
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

