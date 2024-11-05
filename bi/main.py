import setup
import inspect
import warnings
import arviz as az
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpyro
import time as tm
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax as jax
import numpy as np
import jax.random as random
from Mutils import Mgaussian as gaussian
from Mutils import factors 
from network.net import net
from setup.device import setup
from utils.unified_dists import UnifiedDist as dist
from numpyro.infer import Predictive
from numpyro.handlers import condition, seed
import inspect


class bi(dist, gaussian, factors, net):
    def __init__(self, platform='cpu', cores=None, dealocate = False):
        setup(platform, cores, dealocate) 
        import numpyro
        jax.config.update("jax_enable_x64", True)
        self.numpypro = numpyro
        self.trace = None
        self.priors_name = None
        self.data_on_model = None
        self.data_modification = {}
        self.pandas_to_jax_dtype_map = {
            'int64': jnp.int64,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'float64': jnp.float64,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }
        

    def setup(self, platform='cpu', cores=None, dealocate = False):
        setup.setup(platform, cores, dealocate) 

    # Dist functions (sampling and model)--------------------------
    class dist(dist):
        pass

    # Network functions--------------------------
    class net(net):
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
                self.df.loc[:, col] = (self.df.loc[:,col] - self.df.loc[:,col].mean())/self.df.loc[:,col].sd()

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = (self.df.loc[:, cols[a]] - self.df.loc[:, cols[a]].mean()) / self.df.loc[:, cols[a]].std()


        self.data_modification['scale'] = cols # store info of scaled columns
        
        return self.df
    
    def to_float(self, cols = 'all', type = 'float32'):
        if cols == 'all':
            for col in self.df.columns:                
                self.df.loc[:, col] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.loc[:,cols[a]].astype(type)


        self.data_modification['float'] = cols # store info of scaled columns
        
        return self.df

    def to_int(self, cols = 'all', type = 'int32'):
        if cols == 'all':
            for col in self.df.columns:                
                self.df.iloc[:, cols] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.iloc[:,cols[a]].astype(type)


        self.data_modification['int'] = cols # store info of scaled columns

    def pd_to_jax(self, model, bit = '32'):
        params = inspect.signature(model).parameters
        args_without_defaults = []
        args_with_defaults = {}
        for param_name, param in params.items():
            if param.default == inspect.Parameter.empty:
                args_without_defaults.append(param_name)
            else:
                args_with_defaults[param_name] = (param.default, type(param.default).__name__)

        test = all(elem in self.df.columns for elem in args_without_defaults)
        result = dict()
        if test:
            for arg in args_without_defaults:
                varType = str(self.df[arg].dtype)
                result[arg] = jnp.array(self.df[arg], dtype = self.pandas_to_jax_dtype_map.get(varType))
        else:
            return "Error, no"

        for k in args_with_defaults.keys():
            print(args_with_defaults[k][1])
            result[k] = jnp.array(args_with_defaults[k][0], dtype =self.pandas_to_jax_dtype_map.get(str(args_with_defaults[k][1]) + bit))

        return result     

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
    def run(self, 
            model = None, 
            potential_fn=None,
            kinetic_fn=None,
            step_size=1.0,
            inverse_mass_matrix=None,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=0.8,
            trajectory_length=None,
            max_tree_depth=10,
            init_strategy= numpyro.infer.init_to_uniform,
            find_heuristic_step_size=False,
            forward_mode_differentiation=False,
            regularize_mass_matrix=True,
            
            num_warmup = 500,
            num_samples = 500,
            num_chains=1,
            thinning=1,
            postprocess_fn=None,
            chain_method="parallel",
            progress_bar=True,
            jit_model_args=False):
            
        if model is None:
            raise CustomError("Argument model can't be None")
         
        self.model = model
        if self.data_on_model is None:
            self.data_on_model = self.pd_to_jax(self.model)

        self.sampler = MCMC(NUTS(self.model,
                                potential_fn=potential_fn,
                                kinetic_fn=kinetic_fn,
                                step_size=step_size,
                                inverse_mass_matrix=inverse_mass_matrix,
                                adapt_step_size=adapt_step_size,
                                adapt_mass_matrix=adapt_mass_matrix,
                                dense_mass=dense_mass,
                                target_accept_prob=target_accept_prob,
                                trajectory_length=trajectory_length,
                                max_tree_depth=max_tree_depth,
                                init_strategy=init_strategy,
                                find_heuristic_step_size=find_heuristic_step_size,
                                forward_mode_differentiation=forward_mode_differentiation,
                                regularize_mass_matrix=regularize_mass_matrix), 
                                num_warmup = num_warmup,
                                num_samples = num_samples,
                                num_chains=num_chains,
                                thinning=thinning,
                                postprocess_fn=postprocess_fn,
                                chain_method=chain_method,
                                progress_bar=progress_bar,
                                jit_model_args=jit_model_args)

        self.sampler.run(jax.random.PRNGKey(0), **self.data_on_model)

    # Sample model--------------------------
    @staticmethod
    def sample_from_model(model, observed=None, model_args=(), model_kwargs={}, num_samples=1000, params=None,  rng_key=None):
        """
        Generic sampling function for NumPyro models that allows conditioning on
        parameters and observed values.

        Parameters:
        -----------
        model : callable
            NumPyro model function
        num_samples : int
            Number of samples to draw
        params : dict, optional
            Dictionary of parameter values to condition on, e.g., {'a': 1.0, 'b': 2.0}
        observed : dict, optional
            Dictionary of observed values to condition on. Can also contain model arguments.
        rng_key : jax.random.PRNGKey, optional
            Random number generator key
        model_args : tuple
            Additional positional arguments to pass to the model
        model_kwargs : dict
            Additional keyword arguments to pass to the model

        Returns:
        --------
        dict : Dictionary of samples for all random variables in the model

        Example:
        --------
        from main import* 
        def model(weight, height = None):    
            a = dist.normal( 178, 20, name = 'a',shape= [1])
            b = dist.lognormal(  0, 1, name = 'b',shape= [1])   
            s = dist.uniform( 0, 50, name = 's',shape = [1])
            lk("y", Normal(a + b * weight , s), obs=height)

        m.sample_from_model(model, model_kwargs = {'weight': m.data_on_model['weight']},  params = {'a': 10}, num_samples = 1)
        """

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Get model's required arguments
        sig = inspect.signature(model)
        required_params = {
            name: param 
            for name, param in sig.parameters.items() 
            if param.default == inspect.Parameter.empty
        }

        # Extract model arguments from observed data if present
        if observed is not None:
            model_kwargs.update({
                name: observed[name]
                for name in required_params
                if name in observed
            })

        # Separate conditioning values from model arguments in observed
        if observed is not None:
            conditioning_vars = {
                k: v for k, v in observed.items() 
                if k not in required_params
            }
        else:
            conditioning_vars = {}

        # Combine parameter and observation conditions
        conditions = {}
        if params is not None:
            conditions.update(params)
        conditions.update(conditioning_vars)

        # Check if all required arguments are provided
        missing_args = [
            name for name in required_params 
            if name not in model_kwargs
        ]
        if missing_args:
            raise ValueError(
                f"Missing required model arguments: {missing_args}. "
                "Please provide these in the observed dictionary or model_kwargs."
            )

        # If we have conditions, wrap the model
        if conditions:
            model = condition(model, conditions)

        # Create the predictive object and sample
        predictive = Predictive(model, num_samples=num_samples)
        samples = predictive(rng_key, **model_kwargs)

        return samples

    def sample(self, model_kwargs={}, params=None, samples = 1000,   rng_key=None):
        """Sample model with within data

        See sample_from_model for further informations
        """
        return bi.sample_from_model(self.model, model_kwargs = self.data_on_model, observed=None, model_args=(),  num_samples=samples, params=params,  rng_key=rng_key)

    # Get posteriors ----------------------------------------------------------------------------
    @staticmethod
    def get_posteriors(group_by_chain=False):
        self.sampler.get_samples()

    # Log probability ----------------------------------------------------------------------------
    def log_prob(self, model, seed = 0, **kwargs):
        """Compute the log probability of a model, the Transforms parameters to constrained space, the gradient of the negative log probability. 

        Args:
            model (_type_): _description_
            seed (int, optional): _description_. Defaults to 0.
            **kwargs: 

        Returns:
            _type_: _description_
        """
        # getting log porbability
        rng_key = jax.random.PRNGKey(int(seed))
        init_params, potential_fn, constrain_fn, model_trace = numpyro.infer.util.initialize_model(rng_key, model, 
        model_args=(kwargs))
        print('init_params:  ', init_params)
        print('constrain_fn: ', constrain_fn(init_params.z))
        print('potential_fn: ', -potential_fn(init_params.z)) #log prob
        print('grad:         ', jax.grad(potential_fn)(init_params.z))
        return init_params, potential_fn, constrain_fn, model_trace 
        
    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(self):
        self.trace = az.from_numpyro(self.sampler)
        self.priors_name = list(self.trace['posterior'].data_vars.keys())
        return self.trace

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


from numpyro import sample as lk
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
#from jax import random
import random as r
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import*