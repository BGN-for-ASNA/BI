import setup
import warnings
import arviz as az
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax as jax
import numpy as np
import jax.random as random

from setup.device import setup
from data.manip import manip
from utils.array import Mgaussian as gaussian
from utils.array import factors 
from network.net import net
from mcmc.Link import link
from mcmc.Model import model
from mcmc.Samplers import samplers

from utils.unified_dists import tfpLight as dist


import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import *
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import inspect
import re

class bi(manip, dist, gaussian, factors, net, link, samplers ):
    def __init__(self, platform='cpu', cores=None, deallocate = False, dtype = jnp.float32):
        setup(platform, cores, deallocate) 
        jax.config.update("jax_enable_x64", True)
        self.trace = None
        self.dtype = dtype
        super().__init__()

    # Dist functions (sampling and model)--------------------------
    class dist(dist):
        pass
    
    # Links functions--------------------------
    class link(link):
        pass

    # Network functions--------------------------
    class net(net):
        pass

    # Model building functions--------------------------
    class model(model):
        pass

    # MCMC samplers --------------------------
    class samplers(samplers):
        pass

    # Get posteriors ----------------------------------------------------------------------------
    def trace_to_az(self, sample_stats_name=['target_log_prob','log_accept_ratio','has_divergence','energy']):

        var_names= list(self.model_info.keys())
        sample_stats = {k:jnp.transpose(v) for k, v in zip(sample_stats_name, self.sample_stats)}
        trace = {}
        for name, samp in zip(var_names, self.posterior):
            if len(samp.shape) == 2:
                transposed_shape = [1, 0]
            elif len(samp.shape) == 3:
                transposed_shape = [1, 0, 2]
            else:
                transposed_shape = [1, 0, 2, 3]
            trace[name] = jnp.transpose(samp, transposed_shape)
        self.trace = az.from_dict(posterior=trace, sample_stats=sample_stats)
        self.priors_name = var_names
        return self.trace

    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        if self.trace is None:
            self.trace_to_az()
        self.tab_summary = az.summary(self.trace , round_to=round_to, kind=kind, hdi_prob=hdi_prob, *args, **kwargs)
        return self.tab_summary 
   
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
        self.comparaison = az.compare(dict, *args, **kwargs)
        return self.comparaison 

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

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import random as r
