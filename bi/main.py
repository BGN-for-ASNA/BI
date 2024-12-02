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
from Network import Net
from Mutils import Mgaussian as gaussian
from Mutils import factors 
from unified_dists import tfpLight as dist
from samplers import NUTS

import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import *
import inspect
import re

class bi(dist, gaussian, factors):
    def __init__(self, platform='cpu', cores=None, dealocate = False):
        setup.setup(platform, cores, dealocate) 
        self.trace = None
        

    def setup(self, platform='cpu', cores=None, dealocate = False):
        setup.setup(platform, cores, dealocate) 

    # Dist functions (sampling and model)--------------------------
    class dist(dist):
        pass

    # Network functions--------------------------
    class net(Net):
        pass

    # Import data----------------------------
    def data(self, path, **kwargs):
        self.data_original_path = path
        self.data_args = kwargs
        self.df = pd.read_csv(path, **kwargs)
        self.data_modification = {}
        return self.df
   
    def OHE(self, cols = 'all'):
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            OHE = pd.get_dummies(self.df, columns=colCat, dtype=int)
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            OHE = pd.get_dummies(self.df, columns=cols, dtype=int)

        OHE.columns = OHE.columns.str.replace('.', '_')
        OHE.columns = OHE.columns.str.replace(' ', '_')


        self.df = pd.concat([self.df , OHE], axis=1)
        self.data_modification['OHE'] = cols
        return OHE

    def index(self, cols = 'all'):
        self.index_map = {}
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            for a in range(len(colCat)):                
                self.df["index_"+ colCat[a]] =  self.df.loc[:,colCat[a]].astype("category").cat.codes
                self.df["index_"+ colCat[a]] = self.df["index_"+ colCat[a]].astype(np.int64)
                self.index_map[colCat[a]] = dict(enumerate(self.df[colCat[a]].astype("category").cat.categories ) )
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            for a in range(len(cols)):
                self.df["index_"+ cols[a]] =  self.df.loc[:,cols[a]].astype("category").cat.codes
                self.df["index_"+ cols[a]] = self.df["index_"+ cols[a]].astype(np.int64)

                self.index_map[cols[a]] = dict(enumerate(self.df[cols[a]].astype("category").cat.categories ) )

        self.df.columns = self.df.columns.str.replace('.', '_')
        self.df.columns = self.df.columns.str.replace(' ', '_')

        self.data_modification['index'] = cols # store info of indexed columns
        
        return self.df

    def scale(self, cols = 'all'):
        if cols == 'all':
            for col in self.df.columns:                
                self.df.iloc[:, cols] = (self.df.iloc[:,col] - self.df.iloc[:,col].mean())/self.df.iloc[:,col].sd()

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = (self.df.loc[:, cols[a]] - self.df.loc[:, cols[a]].mean()) / self.df.loc[:, cols[a]].std()


        self.data_modification['scale'] = cols # store info of scaled columns
        
        return self.df
    
    def data_to_model(self, cols):
        jax_dict = {}
        for col in cols:
            jax_dict[col] = jnp.array(self.df.loc[:,col].values)
        self.data_modification['data_on_model'] = cols # store info of data used in the model
        self.data_on_model = jax_dict

    # link functions ----------------------------------------------------------------
    @staticmethod
    @jit
    def logit(x):
        return jnp.log(x / (1 - x))

    # Sampler ----------------------------------------------------------------------------
    def get_model_args(self, model):
        # Get the signature of the function
        signature = inspect.signature(model)

        # Extract argument names
        return [param.name for param in signature.parameters.values()]


    def get_model_var(self):
        arguments = self.get_model_args(self.model)
        var_model = [item for item in self.data_on_model.keys() if item in arguments]
        var_model = {key: self.data_on_model[key] for key in arguments if key in self.data_on_model}
        self.var_model = var_model
        self.model_to_send = self.model(**var_model)

    def get_model_distributions(self, model):
        source_code = inspect.getsource(model)
        lines = source_code.split('\n')
        variables = {}
        for line in lines:
            if not line or line.startswith('def') or 'independent' in line.lower() or not 'yield' in line:
                continue
            # Split the line into key and value
            key, value = line.split('=', 1)
            # Remove leading and trailing whitespace
            key = key.strip()
            # Find all words before the brackets
            words = re.findall(r'\b\w+\b(?=\()', value)
            # Create a dictionary with 'distribution' as the key and words as the value
            distribution = {
                'distribution': words[0]
            }
            # Add the key-value pair to the dictionary
            variables[key] = distribution
        self.model_info = variables

    def initialise(self, infos, init_params):
        init_params2 = []
        bijectors = []
        i = 0
        for key in infos.keys():  
            tmp = infos[key]['distribution'].lower()
            if 'lkj' in tmp:
                infos[key]['shape'] = int(init_params[i].shape[0])
                init_params2.append(jnp.array(jnp.eye(infos[key]['shape'])))            
                bijectors.append(tfb.CorrelationCholesky())
            elif 'exponential' in tmp:
                 init_params2.append(jnp.array(jnp.ones_like(init_params[i])))
                 infos[key]['shape'] = init_params[i].shape
                 bijectors.append(tfb.Exp())
            else:
                init_params2.append(jnp.array(jnp.ones_like(init_params[i])))
                infos[key]['shape'] = init_params[i].shape
                bijectors.append(tfb.Identity())
            i+=1
        return init_params2, bijectors

    def run(self, model, obs, 
            n_chains = 1, init = None, target_log_prob_fn = None,
            num_results = 500,
            num_burnin_steps=500,
            num_steps_between_results=0,
            parallel_iterations = 10,
            seed=0,
            name=None):

        if model is None:
            raise CustomError("Argument model can't be None")

        init_key, key = jax.random.split(jax.random.PRNGKey(int(seed)))
        init_key = jnp.array(init_key)
        self.model = model
        self.get_model_distributions(model)        
        self.obs = self.data_on_model[obs]
        self.get_model_var()

        self.tensor = JointDistributionCoroutine(model(**self.var_model))
        init_params = self.tensor.sample(seed = init_key)    
        _, bijectors = initialise(infos, init_params)


        sampler = NUTS(model = self.model_to_send, obs = self.obs, infos = self.model_info, n_chains = n_chains, init = init, target_log_prob_fn = target_log_prob_fn,
        num_results = num_results, num_burnin_steps=num_burnin_steps, num_steps_between_results=num_steps_between_results,
        parallel_iterations = parallel_iterations, seed=seed,name=name)

        return sampler


    # Get posteriors ----------------------------------------------------------------------------
    @staticmethod
    def trace_to_az(posterior, 
                       sample_stats,
                       var_names=None, 
                       sample_stats_name=['target_log_prob','log_accept_ratio','has_divergence','energy']):
        sample_stats = {k:jnp.transpose(v) for k, v in zip(sample_stats_name, sample_stats)}
        trace = {}
        for name, samp in zip(var_names, posterior):
            if len(samp.shape) == 2:
                transposed_shape = [1, 0]
            elif len(samp.shape) == 3:
                transposed_shape = [1, 0, 2]
            else:
                transposed_shape = [1, 0, 2, 3]
            trace[name] = jnp.transpose(samp, transposed_shape)
        trace = az.from_dict(posterior=trace, sample_stats=sample_stats)
        return trace

    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        if self.trace is None:
            self.to_az()
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
