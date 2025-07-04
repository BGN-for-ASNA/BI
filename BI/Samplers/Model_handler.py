from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re


class model_handler():
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

    def initialise(self, infos, init_params):
        init_params2 = []
        bijectors = []
        i = 0
        for key in infos.keys():
            dist_name = infos[key]['distribution'].lower()
            param_shape = init_params[i].shape

            # --- Correlation Matrix ---
            if 'lkj' in dist_name or 'correlation' in dist_name:
                print(f"INFO: Found LKJ/Correlation parameter '{key}'. Applying CorrelationCholesky bijector.")
                # The bijector works in the unconstrained space, so init is tricky.
                # A simple identity matrix is a safe starting point for the Cholesky factor.
                init_params2.append(jnp.eye(param_shape[0]))
                bijectors.append(tfb.CorrelationCholesky())

            # --- Positive-Only Parameters (scale, rates) ---
            elif 'exponential' in dist_name or 'half' in dist_name or 'gamma' in dist_name or 'chi2' in dist_name:
                print(f"INFO: Found Positive parameter '{key}'. Applying Exp bijector.")
                # Start at 1.0 in the constrained space (so log(1)=0 in unconstrained)
                init_params2.append(jnp.ones_like(init_params[i]))
                bijectors.append(tfb.Exp())

            # --- Unit Interval Parameters (probabilities) ---
            elif 'beta' in dist_name:
                print(f"INFO: Found Unit Interval parameter '{key}'. Applying Sigmoid bijector.")
                # Start at 0.5 in constrained space (so logit(0.5)=0 in unconstrained)
                init_params2.append(jnp.full_like(init_params[i], 0.5))
                bijectors.append(tfb.Sigmoid())

            # --- Simplex Parameters (probability vectors) ---
            elif 'dirichlet' in dist_name:
                print(f"INFO: Found Simplex parameter '{key}'. Applying SoftmaxCentered bijector.")
                # Start with a uniform probability vector.
                uniform_prob = 1.0 / param_shape[-1]
                init_params2.append(jnp.full_like(init_params[i], uniform_prob))
                bijectors.append(tfb.SoftmaxCentered())

            # --- Default: Unconstrained Parameters ---
            else:
                print(f"WARNING: No specific bijector found for '{key}' (dist: {dist_name}). Assuming Unconstrained.")
                init_params2.append(jnp.zeros_like(init_params[i])) # Start at 0 for unconstrained
                bijectors.append(tfb.Identity())

            i += 1
        return init_params2, bijectors
