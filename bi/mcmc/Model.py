from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re


class model():
    def __init__():
        self.data_on_model = None # Come from data manip data_to_model function
        self.args = None
        self.vars = None
        self.model = None
        self.model_to_send = None
        self.model_info
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
