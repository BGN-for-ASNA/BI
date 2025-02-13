import setup
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

from data.manip import manip
from utils.array import Mgaussian as gaussian
from utils.array import factors 
from network.net import net
from setup.device import setup
from Surv.surv import survival
from utils.link import link
from diagnostic.Diag import diag

from utils.unified_dists import UnifiedDist as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition



class bi(manip, dist, gaussian, factors, net, survival, link, diag):
    def __init__(self, platform='cpu', cores=None, deallocate = False):
        setup(platform, cores, deallocate) 
        jax.config.update("jax_enable_x64", True)
        self.trace = None
        self.priors_name = None
        self.data_on_model = None
        self.tab_summary = None
        self.obs_args = None
        self.model2 = None # Model with NONE as default
        super().__init__()

    def setup(self, platform='cpu', cores=None, deallocate = False):
        setup.setup(platform, cores, deallocate) 

    def lk(self,*args, **kwargs):
        numpyro.sample(*args, **kwargs)
        
    def randint(self, low, high, shape):
        return pyrand.randint(low, high, shape)

    # Dist functions (sampling and model)--------------------------
    class dist(dist):
        pass

    # Network functions--------------------------
    class net(net):
        pass

    # Survival functions--------------------------
    class surv(survival):
        pass

    # Link functions--------------------------
    class link(link):
        pass

    # Link functions--------------------------
    class diag(diag):
        pass    

    # MCMC ----------------------------------------------------------------------------
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
            jit_model_args=False,
            seed = 0):
            
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

        self.sampler.run(jax.random.PRNGKey(seed), **self.data_on_model)
        self.posteriors = self.sampler.get_samples()

    # Get posteriors ----------------------------------------------------------------------------
    def summary(self, round_to=2, kind="stats", hdi_prob=0.89, *args, **kwargs): 
        if self.trace is None:
            self.to_az()
        self.tab_summary = az.summary(self.trace , round_to=round_to, kind=kind, hdi_prob=hdi_prob, *args, **kwargs)
        return self.tab_summary 

    def get_posterior_means(self):
        d = self.summary()
        posterior_means = d['mean'].values
        posterior_names = d.index.tolist()
        return {var: mean for var, mean in zip(posterior_names, posterior_means)}

    # Sample model ----------------------------------------------------------------------------
    def visit_call(self, node, obs_args):
        """
        Parse a function call node to find `lk` calls with `obs` arguments
        and add those argument names to the `obs_args` list.
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
        """
        # Get the source code of the function
        source_code = inspect.getsource(model_func)
        # Parse the source code into an AST
        tree = ast.parse(source_code)
        # Prepare a list to collect observed arguments
        obs_args = []
        # Traverse nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):  # Only process function calls
                self.visit_call(node, obs_args)
        self.obs_args = obs_args
        return obs_args

    # Create a new model function with the modified signature
    def build_model2(self, model):
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
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            remove_obs (bool, optional): _description_. Defaults to True.
            posterior (bool, optional): If true use posterior obtained from run function. If false it dones't use posterior. Defaults to True.
            samples (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        rng_key = jax.random.PRNGKey(int(seed))
        self.build_model2(self.model)

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
        

#from numpyro import sample as lk
#import random as r
##from numpyro.infer import MCMC, NUTS, Predictive
#from numpyro.distributions import*
#import numpy as np
