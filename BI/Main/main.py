
import inspect
import ast
import warnings
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import numpyro
import time as tm
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax as jax
import numpy as np
import jax.random as random
import numpy as np
import random as pyrand

from BI.Data.manip import manip
from BI.Utils.array import Mgaussian as gaussian
from BI.Utils.array import factors 
from BI.SetDevice.set import setup_device

from BI.Utils.link import link
from BI.Diagnostic.Diag import diag
from BI.Network.Net import net
from BI.NBDA.NBDA import NBDA
from BI.Models.surv import survival
from BI.Models.GMM import *
from BI.Models.DPMM import *
from BI.Models.models import models
from BI.ML.ml import ml
from BI.BNN.bnn import bnn 
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition


class bi(manip):
    def __init__(self, platform='cpu', cores=None, deallocate = False, print_devices_found =    True, backend='numpyro'):
        """
        Initialize the BI class with platform, cores, deallocate, print_devices_found, and backend parameters.
        
        Args:
            platform (str, optional): Platform to use. Defaults to 'cpu'.
            cores (int, optional): Number of cores. Defaults to None.
            deallocate (bool, optional): Whether to deallocate. Defaults to False.
            print_devices_found (bool, optional): Whether to print devices found. Defaults to True.
            backend (str, optional): Backend to use. Defaults to 'numpyro'.
        """
        manip.__init__(self)
        setup_device(platform, cores, deallocate, print_devices_found) 
        
        self.data_on_model = None
        self.priors_name = None
        self.tab_summary = None
        self.model_name = None
        self.nbdaModel = False
        self.obs_args = None
        self.model2 = None 
        self.trace = None
        self.history = {}
        self.backend = backend

        self.gaussian = gaussian
        self.survival = survival(self)
        self.factor = factors
        self.link = link

        self.dpmm = dpmm
        self.gmm = gmm    
        self.NBDA = NBDA

        self.models = models(self)


        self.net = net()
        self.ml= ml()   
        self.bnn= bnn()  

        if backend == 'numpyro':
            from BI.Utils.np_dists import UnifiedDist as np_dists
            self.dist=np_dists 
            jax.config.update("jax_enable_x64", True)

        elif backend == 'tfp':
            from BI.Utils.tfp_dists import UnifiedDist as tfp_dists  
            from BI.Samplers.mcmc_tfp import mcmc as mcmc_tfp          
            self.dist=tfp_dists 
            self.sampler = mcmc_tfp()
            jax.config.update("jax_enable_x64", False)

    def fit(self, 
            model = None, 
            obs=None,
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
            jit_model_args=False,
            seed = 0):
        """
        Fit the model using the specified backend and parameters.
        
        Args:
            model: Model to fit.
            obs: Observed data.
            potential_fn: Potential function.
            kinetic_fn: Kinetic function.
            step_size: Step size for the sampler.
            inverse_mass_matrix: Inverse mass matrix.
            adapt_step_size: Whether to adapt step size.
            adapt_mass_matrix: Whether to adapt mass matrix.
            dense_mass: Whether to use dense mass.
            target_accept_prob: Target acceptance probability.
            trajectory_length: Length of the trajectory.
            max_tree_depth: Maximum tree depth.
            init_strategy: Initialization strategy.
            find_heuristic_step_size: Whether to find heuristic step size.
            forward_mode_differentiation: Whether to use forward mode differentiation.
            regularize_mass_matrix: Whether to regularize mass matrix.
            num_warmup: Number of warmup samples.
            num_samples: Number of samples.
            num_chains: Number of chains.
            thinning: Thinning factor.
            postprocess_fn: Postprocess function.
            chain_method: Chain method.
            progress_bar: Whether to show progress bar.
            jit_model_args: Whether to JIT model arguments.
            seed: Random seed.
        """
        if model is None:
            if self.nbdaModel == False:
                print( "Argument model can't be None")
                
            else:
                self.model = self.nbda.model
                self.model_name = 'NBDA' 
        else:
            self.model = model
            self.model_name = model.__name__

        if self.nbdaModel == False:
            if self.data_on_model is None :
                if self.backend == 'numpyro':
                    self.data_on_model = self.pd_to_jax(self.model)
                else:
                    self.data_on_model = self.pd_to_jax(self.model, bit = "32")

        if self.model_name == 'gmm':
            if 'initial_means' not in self.data_on_model:
                self.ml.KMEANS(self.data_on_model['data'], n_clusters=self.data_on_model['K'])
                self.data_on_model['initial_means'] = self.ml.results['centroids']

        if self.backend == 'numpyro':
            from BI.Samplers.mcmc_numpyro import mcmc_numpyro
            self.sampler = mcmc_numpyro(
                model=self.model,
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
                regularize_mass_matrix=regularize_mass_matrix,
                num_warmup = num_warmup,
                num_samples = num_samples,
                num_chains=num_chains,
                thinning=thinning,
                postprocess_fn=postprocess_fn,
                chain_method=chain_method,
                progress_bar=progress_bar,
                jit_model_args=jit_model_args
                )
                        
            self.sampler.run(jax.random.PRNGKey(seed), **self.data_on_model)
            self.posteriors = self.sampler.get_samples()
            self.diag = diag(sampler = self.sampler)
            self.get_history()

        elif self.backend == 'tfp':
            from BI.Utils.tfp_dists import UnifiedDist as tfp_dists
            from BI.Samplers.mcmc_tfp import mcmc as mcmc_tfp
            from BI.Samplers.Model_handler import model_handler 

            #self.mcmc= mcmc_tfp() 
            self.sampler.data_on_model = self.data_on_model
            self.sampler.model = self.model


            self.sampler.run(model = self.model, obs = obs) 
            self.diag = diag(sampler = self.sampler)
            trace = {}
            var_names= list(self.sampler.model_info.keys())
            for name, samp in zip(var_names, self.sampler.posterior):
                trace[name] = samp
            self.posteriors = trace
            self.get_history()

    def randint(self, low, high, shape):
        """
        Generate random integers in the given range.
        
        Args:
            low: Lowest possible value.
            high: Highest possible value.
            shape: Shape of the output array.
        
        Returns:
            Array of random integers.
        """        
        return pyrand.randint(low, high, shape)

    # Get posteriors ----------------------------------------------------------------------------
    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        """
        Generate a summary of the posterior distribution.
        
        Args:
            round_to: Number of decimal places to round.
            kind: Type of summary statistics.
            hdi_prob: Probability for HDI interval.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Summary statistics.
        """        
        if self.trace is None:
            self.diag.to_az(backend=self.backend)
        self.tab_summary = az.summary(self.diag.trace , round_to=round_to, kind=kind, hdi_prob=hdi_prob, *args, **kwargs)
        return self.tab_summary 

    def get_posterior_means(self):
        """
        Get the posterior means of the variables.
        
        Returns:
            Dictionary of variable names and their posterior means.
        """        
        d = self.summary()
        posterior_means = d['mean'].values
        posterior_names = d.index.tolist()
        return {var: mean for var, mean in zip(posterior_names, posterior_means)}

    # Sample model ----------------------------------------------------------------------------
    def visit_call(self, node, obs_args):
        """
        Parse a function call node to find `lk` calls with `obs` arguments and add those argument names to the `obs_args` list.
        
        Args:
            node: AST node to parse.
            obs_args: List to store observed argument names.
        """
        # Check if the function called is `lk`
        if isinstance(node.func, ast.Name) and node.func.id == "lk":
            # Check for keyword arguments in the `lk` function
            for kw in node.keywords:
                if kw.arg == "obs":  # Look for `obs=`
                    # Add the variable name (if available) to obs_args
                    if isinstance(kw.value, ast.Name):
                        obs_args.append(kw.value.id)

    def find_obs_in_model(self, model_func):
        """
        Extract observed argument names from `obs` in `lk` calls in `model_func`.
        
        Args:
            model_func: Model function to analyze.
        
        Returns:
            List of observed argument values.
        """
        # Get the source code of the function
        source_code = inspect.getsource(self.model)
        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)
        # List to hold the 'obs' values
        obs_values = []

        # Traverse the AST to find all function calls with 'obs' keyword argument
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == 'obs':
                        # Extract the value passed to 'obs'
                        value = ast.unparse(keyword.value)
                        obs_values.append(value)

        self.obs_args = obs_values
        return obs_values

    # Create a new model function with the modified signature
    def build_model_with_Y_None(self, model):
        """
        Modify the original model function to make the observed arguments optional.
        
        Args:
            model: Model function to modify.
        
        Returns:
            Modified model function with optional observed arguments.
        """
        # Extract `obs` argument names
        obs = self.find_obs_in_model(model)
        # Modify the function's signature to make the observed argument optional
        sig = inspect.signature(model)
        parameters = []
        for name, param in sig.parameters.items():
            if name in obs:
                parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None))
            else:
                parameters.append(param)
        def model_with_None(*args, **kwargs):
            # Default values for obs arguments if not passed
            for obs_name in obs:
                if obs_name not in kwargs:
                    kwargs[obs_name] = None
            # Call the original model function with the modified arguments
            return model(*args, **kwargs)
        # Update the signature of the new model
        model_with_None.__signature__ = sig.replace(parameters=parameters)
        self.model2 = model_with_None
        return model_with_None

    def sample(self,  data = None, remove_obs = True, posterior = True,  samples = 1,  seed = 0):
        """
        Sample from the model.
        
        Args:
            data: Data to use for sampling. Defaults to None.
            remove_obs: Whether to remove observed data. Defaults to True.
            posterior: Whether to use posterior. Defaults to True.
            samples: Number of samples. Defaults to 1.
            seed: Random seed. Defaults to 0.
        
        Returns:
            Samples from the model.
        """
        rng_key = jax.random.PRNGKey(int(seed))
        self.build_model_with_Y_None(self.model)

        if data is None:
            data = self.data_on_model.copy() 
        
        if remove_obs:
            for intem in self.obs_args:            
                del data[intem]

        if posterior == False:
            posterior = None

        if posterior == True and samples > 1 :
            tmp = self.get_posterior_means()
            posterior = {key: jnp.repeat(value, samples) for key, value in tmp.items()}

        else: 
            posterior = self.get_posterior_means()

        predictive = Predictive(self.model2, posterior_samples=posterior, num_samples=samples)
        return predictive(rng_key, **data)
    
    # Log probability ----------------------------------------------------------------------------
    def log_prob(self, model, seed = 0, **kwargs):
        """
        Compute the log probability of a model.
        
        Args:
            model: Model to compute log probability for.
            seed: Random seed. Defaults to 0.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple containing init_params, potential_fn, constrain_fn, and model_trace.
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

    def get_history(self):
        """
        Get the history of the model fit.
        
        Returns:
            Dictionary containing model, model_name, data, sampler, and posteriors.
        """        
        self.history = {
            'model': self.model,
            'model_name': self.model_name,  
            'data': self.data_on_model,
            'sampler': self.sampler,
            'posteriors': self.posteriors,
        }

    def plot(self, X, y=None, figsize=(10, 6), **kwargs):
        """
        Plot the model results.
        
        Args:
            X: Data to plot.
            y: Target variable. Defaults to None.
            figsize: Figure size. Defaults to (10, 6).
            **kwargs: Additional keyword arguments.
        """        
        if self.model_name == 'gmm':
            plot_gmm(X, sampler= self.sampler,figsize=figsize,**kwargs)
        if self.model_name == 'dpmm':
            plot_dpmm(X, sampler= self.sampler,figsize=figsize,**kwargs)
