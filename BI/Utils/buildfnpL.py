# %%
import inspect
import numpyro

# Gather all distribution functions from numpyro.distributions
all_names = dir(numpyro.distributions)

# Create a dictionary with all names
class_dict = {name: getattr(numpyro.distributions, name) for name in all_names}

# Create a Python file and write the import statement and class with methods to it
with open("unified_dists.py", "w") as file:
    # Write the import statement
    file.write("from functools import partial\n")
    file.write("import jax\n")
    file.write("from jax import random\n")
    file.write("from jax import jit\n")
    file.write("import numpyro as numpyro\n\n")
    
    
    # Write the class definition with __init__ method
    file.write("class UnifiedDist:\n\n")
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
                full_signature = f"{param_str}, shape=(), sample = False, seed = 0, name = 'x',obs=None"
                
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
                method_str += f"        if obs != None:\n"      
                method_str += f"                return numpyro.sample(name, numpyro.distributions.{value.__name__}({arg_str}),obs)\n"  
                method_str += f"        else: \n"  
                method_str += f"                if sample== True:\n"
                method_str += f"                        seed = random.PRNGKey(seed)\n"
                method_str += f"                        return numpyro.distributions.{value.__name__}({arg_str}).sample(seed, shape)\n"                
                method_str += f"                else: \n"
                method_str += f"                        return numpyro.sample(name, numpyro.distributions.{value.__name__}({arg_str}).expand(shape))\n"
                
                # Write the method string to the file
                file.write(method_str + "\n")
            except Exception as e:
                print(f"Error creating method for {key}: {e}")
        else:
            print(f"Ignoring non-callable object for key {key}: {value}")


# %%
