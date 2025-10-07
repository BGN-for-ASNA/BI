from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re


class model_handler():
    """    The model_handler class is designed to facilitate the management and manipulation of Bayesian models within the BI framework. It provides methods to extract model arguments, variables, and distributions from a given model function. The class also includes functionality to initialize model parameters and bijectors based on the model's structure and the nature of its variables. This class serves as a bridge between raw model definitions and their practical application in Bayesian inference, ensuring that models are correctly configured for fitting and analysis.
    """
    def __init__(self):
        self.data_on_model = None # Come from data manip data_to_model function
        self.args = None
        self.vars = None
        self.model = None
        self.model_to_send = None
        self.model_info= None
        self.init_params2 = None
        self.bijectors = None

    def get_model_args(self, model):
        # Get the signature of the function
        signature = inspect.signature(model)

        # Extract argument names
        self.args = [param.name for param in signature.parameters.values()]
        return self.args 

    def get_model_var(self, model):
        arguments = self.get_model_args(model)
        var_model = [item for item in self.data_on_model.keys() if item in arguments]
        var_model = {key: self.data_on_model[key] for key in arguments if key in self.data_on_model}
        self.vars= var_model # data for the model converted as jnp arrays
    

    def get_model_distributions(self, model):
        source_code = inspect.getsource(model)
        lines = source_code.split('\n')
        variables = {}
        for line in lines:
            if not line or line.startswith('def') or 'obs' in line.lower() or not 'yield' in line:
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

    # You need to pass the name of the observed variable to this function
    def initialise(self, infos, init_params, obs_name=None):
        init_params2 = []
        bijectors = []
        i = 0 # This now correctly tracks the index for init_params

        for key in infos.keys():
            # --- NEW: Check if this is the observed variable ---
            if key == obs_name:
                print(f"INFO: Skipping bijector for observed variable '{key}'.")
                continue  # Skip to the next key in the loop
            # --- END NEW ---

            dist_name = infos[key]['distribution'].lower()
            
            # Now it's safe to access init_params[i] because we've skipped
            # the observed variable, and `i` corresponds to the unobserved params.
            param_shape = init_params[i].shape

            # --- Correlation Matrix ---
            if 'lkj' in dist_name or 'correlation' in dist_name:
                print(f"INFO: Found LKJ/Correlation parameter '{key}'. Applying CorrelationCholesky bijector.")
                init_params2.append(jnp.eye(param_shape[0]))
                bijectors.append(tfb.CorrelationCholesky())

            # --- Positive-Only Parameters (scale, rates) ---
            elif 'exponential' in dist_name or 'half' in dist_name or 'gamma' in dist_name or 'chi2' in dist_name:
                print(f"INFO: Found Positive parameter '{key}'. Applying Exp bijector.")
                init_params2.append(jnp.ones_like(init_params[i]))
                bijectors.append(tfb.Exp())
            
            # (... rest of your elif conditions for beta, dirichlet, etc. ...)

            # --- Default: Unconstrained Parameters ---
            else:
                print(f"WARNING: No specific bijector found for '{key}' (dist: {dist_name}). Assuming Unconstrained.")
                init_params2.append(jnp.zeros_like(init_params[i])) # Start at 0 for unconstrained
                bijectors.append(tfb.Identity())

            # IMPORTANT: Only increment `i` for unobserved variables
            i += 1
            
        return init_params2, bijectors