#%%
import re
import inspect
from tensorflow_probability.substrates import jax as tfp

all_names = dir(tfp.distributions)

# Create a dictionary with all names
class_dict = {name: getattr(tfp.distributions, name) for name in all_names}

def merge_function_signature(param_str):
    # Add your custom arguments
    additional_args = "shape=(), sample=False, seed=0,"

    # Regex patterns
    pos_args_pattern = r"(\w+)(?=\s*(?:,|$))"  # Matches positional arguments
    kw_args_pattern = r"(\w+\s*=\s*(?:[^,]+|'[^']*'|\"[^\"]*\"|None|True|False|args|kwargs|\d+\.?\d*|[1-9]\d*|1e-?\d+|0))"  # Matches keyword arguments, including complex ones

    # Find positional and keyword arguments
    positional_args = re.findall(pos_args_pattern, param_str)
    keyword_args = re.findall(kw_args_pattern, param_str)

    # Filter out unwanted arguments from positional args
    filtered_positional_args = [
        item for item in positional_args if item not in {'args', 'kwargs', 'None', 'False', 'True', '0', '06','2','20','8', '5','100'}
    ]

    # Construct the full signature
    if filtered_positional_args:
        full_signature = f"{', '.join(filtered_positional_args)}, {additional_args} {', '.join(keyword_args)}, *args, **kwargs"
    else:
        full_signature = f"{additional_args}, {', '.join(keyword_args)}, *args, **kwargs"

    # Remove possible trailing commas and spaces
    full_signature = full_signature.strip(", ")
    full_signature = full_signature.replace(', ,', ',')    
    full_signature = full_signature.replace(',,', ',')
    full_signature = full_signature.replace("<class 'jax.numpy.int32'>", 'jax.numpy.int32')
    full_signature = full_signature.replace("<class 'jax.numpy.float32'>", 'jax.numpy.float32')
    full_signature = full_signature.replace("quadrature_fn=<function quadrature_scheme_lognormal_quantiles at 0x0000014B29A33F60>", 'quadrature_fn=tfp.distributions.quadrature_scheme_lognormal_quantiles')

    return full_signature

# Create a Python file and write the import statement and class with methods to it
with open("unified_dists.py", "w") as file:
    # Write the import statement
    file.write("import jax \n")
    file.write("import jax.numpy as jnp\n")
    file.write("from tensorflow_probability.substrates import jax as tfp\n")
    file.write("from tensorflow_probability.substrates.jax.distributions import*\n")
    file.write("import tensorflow_probability.substrates.jax.distributions as tfd\n")
    file.write("tfb = tfp.bijectors\n")
    file.write("root = tfd.JointDistributionCoroutine.Root\n\n")

    
    # Write the class definition with __init__ method
    file.write("class tfpLight:\n\n")
    file.write("    def __init__(self):\n")
    file.write("        pass\n\n")
    
    # Write the generated methods with enhanced docstrings and dynamic signatures
    for key, value in class_dict.items():
        if callable(value):
            try:
                # Use inspect to get the signature of the function
                signature = inspect.signature(value)
                parameters = signature.parameters
                
                # Build the method signature string
                param_str = ", ".join([str(param) for param in parameters.values()])
                full_signature = f"{param_str}, shape=(), sample = False, seed = 0, name = 'x'"
                
                # Create the method definition string with dynamic arguments
                method_name = key.lower()
                method_str = f"    @staticmethod\n"
                #method_str = f"    @partial(jit, static_argnames=['sample'])\n"
                method_str += f"    def {method_name}({full_signature}):\n"
                
                # Create a docstring with the method name and parameters
                docstring = f"{value.__name__} distribution.\n\n"
                docstring += "    Arguments:\n"
                for param in parameters.values():
                    docstring += f"        {param.name}: {param.default}\n"
                docstring += "        shape: Shape of samples to be drawn.\n"
                
                # Format and indent the docstring
                indented_docstring = '\n    '.join(docstring.splitlines())
                method_str += f'        """\n        {indented_docstring}\n        """\n'
                
                # Create the argument string for the return statement
                arg_names = [param.name for param in parameters.values()]
                arg_str = ", ".join([f"{arg}={arg}" for arg in arg_names])
                
                
                # Add the method body with explicit argument passing                
                method_str += f"        if sample:\n"
                method_str += f"            seed = jax.random.PRNGKey(seed)\n"
                method_str += f"            return tfd.{value.__name__}({arg_str}).sample(sample_shape = shape, seed = seed)\n"
                method_str += f"        else: \n"
                method_str += f"            return root(tfd.Sample(tfd.{value.__name__}({arg_str}), shape))\n"
                
                # Write the method string to the file
                file.write(method_str + "\n")
            except Exception as e:
                print(f"Error creating method for {key}: {e}")
        else:
            print(f"Ignoring non-callable object for key {key}: {value}")




# %%
