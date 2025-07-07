#%%
"""
This file wraps TFP functions to enable different calls depending on the provided arguments.
"""

import re
import inspect
from tensorflow_probability.substrates import jax as tfp

# Get all callable distribution objects from tfp.distributions
all_names = dir(tfp.distributions)
class_dict = {
    name: getattr(tfp.distributions, name)
    for name in all_names
    if not name.startswith("_") and inspect.isclass(getattr(tfp.distributions, name))
}

# Create a Python file and write the import statement and class with methods to it
with open("tfp_dists.py", "w") as file:
    # Write the import statements
    file.write("import jax\n")
    file.write("import jax.numpy as jnp\n")
    file.write("from tensorflow_probability.substrates import jax as tfp\n")
    file.write("import tensorflow_probability.substrates.jax.distributions as tfd\n\n")
    file.write("tfb = tfp.bijectors\n")
    file.write("root = tfd.JointDistributionCoroutine.Root\n\n")

    # Write the class definition
    file.write("class UnifiedDist:\n")
    file.write("    \"\"\"A class that wraps TFP distributions for a unified sampling interface.\"\"\"\n\n")
    
    # Write the generated methods
    for key, value in class_dict.items():
        try:
            # Use inspect to get the signature of the distribution's __init__ method
            signature = inspect.signature(value)
            parameters = signature.parameters
            # Get a set of existing parameter names for quick lookups
            existing_param_names = set(parameters.keys())

            # --- KEY CHANGE STARTS HERE ---

            # Define the custom arguments we want to add to our wrapper
            custom_args_to_add = {
                'shape': '()',
                'sample': 'False',
                'seed': '0',
                'obs': 'None',
                'wrap': 'True'
            }
            
            # Build the list of new parameter strings, ONLY if they don't already exist
            new_params_list = []
            for arg_name, default_value in custom_args_to_add.items():
                if arg_name not in existing_param_names:
                    new_params_list.append(f"{arg_name}={default_value}")

            # Get the original parameters as a string
            original_param_str = ", ".join([str(p) for p in parameters.values() if p.name != 'self'])

            # Combine original and new parameters for the final signature
            # This logic handles cases where there are no original params or no new params
            all_params = [p for p in [original_param_str] + new_params_list if p]
            full_signature = ", ".join(all_params)

            # --- KEY CHANGE ENDS HERE ---
            
            # Create the method definition string
            method_name = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower() # Convert CamelCase to snake_case
            method_str = f"    @staticmethod\n"
            method_str += f"    def {method_name}({full_signature}):\n"
            
            # Create a rich docstring
            docstring = f"Wrapper for the tfd.{value.__name__} distribution.\n\n"
            docstring += "    This method can either return a distribution object for use in a\n"
            docstring += "    probabilistic model or directly return samples.\n\n"
            docstring += "    Original TFP Arguments:\n"
            for param in parameters.values():
                if param.name != 'self':
                    docstring += f"        {param.name}: Default = {param.default}\n"
            docstring += "\n    Wrapper Arguments:\n"
            docstring += "        shape (tuple): The shape of the sample. Defaults to ().\n"
            docstring += "        sample (bool): If True, draw samples from the distribution. \n"
            docstring += "                       If False, return a Root distribution object. Defaults to False.\n"
            docstring += "        seed (int): The PRNG seed for sampling. Defaults to 0.\n"
            
            # Format and indent the docstring
            indented_docstring = '\n        '.join(docstring.splitlines())
            method_str += f'        """\n        {indented_docstring}\n        """\n'
            
            # Create the argument string for instantiating the TFP distribution
            # This correctly forwards all the original arguments
            arg_names = [p.name for p in parameters.values() if p.name != 'self']
            arg_str = ", ".join([f"{arg}={arg}" for arg in arg_names])
            
            # Add the method body
            method_str += f"        dist = tfd.{value.__name__}({arg_str})\n"
            method_str += f"        if sample:\n"
            method_str += f"            prng_key = jax.random.PRNGKey(seed)\n"
            method_str += f"            return dist.sample(sample_shape=shape, seed=prng_key)\n"
            method_str += f"        if obs is not None:\n"
            method_str += f"             final_dist = tfd.Independent(dist, reinterpreted_batch_ndims=shape)\n"
            method_str += f"        else:\n"
            method_str += f"             final_dist = tfd.Sample(dist, sample_shape=shape) if shape else dist\n"
            method_str += f"        if wrap:\n"
            method_str += f"            return root(final_dist)\n"
            method_str += f"        else:\n"
            method_str += f"            return final_dist\n"
            
            # Write the method string to the file
            file.write(method_str + "\n\n")
        except (ValueError, TypeError) as e:
            # Some objects in tfp.distributions might not be classes or inspectable
            print(f"Skipping {key} due to inspection error: {e}")

print("Successfully generated tfp_dists.py")



# %%
