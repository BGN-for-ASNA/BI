# %%
"""
This file wraps NumPyro functions to enable different calls depending on the provided arguments.
"""

import inspect
import re
import numpyro
from numpyro.distributions import Distribution

# --- Filter for actual, instantiable Distribution classes ---
dist_classes = {}
for name in dir(numpyro.distributions):
    obj = getattr(numpyro.distributions, name)
    if (
        inspect.isclass(obj) and issubclass(obj, Distribution) and not name.startswith("_")
        and name not in ["Distribution", "ExpandedDistribution", "TransformedDistribution", "IndependentDistribution"]
    ):
        dist_classes[name] = obj
dist_classes['TransformedDistribution'] = numpyro.distributions.TransformedDistribution
dist_classes['Independent'] = numpyro.distributions.Independent

all_names = dir(numpyro.distributions)
dist_classes = {name: getattr(numpyro.distributions, name) for name in all_names}
# --- Generate the wrapper file ---
with open("dists.py", "w") as file:
    file.write("import jax\n")
    file.write("from jax import random\n")
    file.write("import numpyro\n\n")
    file.write("class UnifiedDist:\n\n")
    file.write("    def __init__(self):\n")
    file.write("        pass\n\n")

    file.write("    def mask(self,mask):\n")
    file.write("        return numpyro.handlers.mask(mask=mask)\n\n")

    file.write("    def plate(self,name, shape):\n")
    file.write("        return numpyro.plate(name, shape)\n\n")


    for name, dist_class in dist_classes.items():
        try:

            signature = inspect.signature(dist_class)
            parameters = signature.parameters
            param_str = ", ".join([str(param) for param in parameters.values()])

            # <-- MODIFIED: Renamed 'to_event_dims' to 'event' and added 'mask'
            wrapper_args = "name='x', obs=None, mask=None, sample=False, seed=0, shape=(), event=0,create_obj=False"
            full_signature = f"{param_str}, {wrapper_args}"

            method_name = name.lower()
            method_str = f"    @staticmethod\n"
            method_str += f"    def {method_name}({full_signature}):\n"

            # <-- MODIFIED: Updated docstrings for 'event' and 'mask'
            docstring_parts = [f"{name} distribution wrapper."]
            docstring_parts.append("\n    Original Arguments:\n    -----------------")
            for param in parameters.values():
                desc =  f"{param.name}: {param.default}\n"
                indented_desc = '\n        '.join(desc.split('\n'))
                docstring_parts.append(f"    {indented_desc}")
            docstring_parts.append("\n    Wrapper Arguments:\n    ------------------")
            docstring_parts.append("    shape (tuple): A multi-purpose argument for shaping.")
            docstring_parts.append("        - When sample=False (model building), this is used with `.expand(shape)` to set the distribution's batch shape.")
            docstring_parts.append("        - When sample=True (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.")
            docstring_parts.append("    event (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).")
            docstring_parts.append("    mask (jnp.ndarray, bool): Optional boolean array to mask observations. This is passed to the `infer={'obs_mask': ...}` argument of `numpyro.sample`.")
            docstring_parts.append("    create_obj (bool): If True, returns the raw NumPyro distribution object instead of creating a sample site.")
            docstring_parts.append("        This is essential for building complex distributions like `MixtureSameFamily`.")

            full_docstring = "\n".join(docstring_parts)
            indented_full_docstring = '\n        '.join(full_docstring.split('\n'))
            method_str += f'        """\n        {indented_full_docstring}\n        """\n'

            arg_names = [param.name for param in parameters.values()]
            arg_str = ", ".join([f"{arg}={arg}" for arg in arg_names])
            method_str += f"        d = numpyro.distributions.{name}({arg_str})\n"

            # The `sample` flag is for direct JAX array sampling, completely separate from model building.
            method_str += f"        if sample:\n"
            method_str += f"            seed_key = random.PRNGKey(seed)\n"
            method_str += f"            return d.sample(seed_key, sample_shape=shape)\n"

            # This `else` block now handles ALL model-building logic (creating sample sites OR objects).
            method_str += f"        else:\n"

            # --- This is the common logic for modifying the distribution object ---
            # It applies to both `create_obj=True` and `create_obj=False`.
            method_str += f"            if shape:\n"
            method_str += f"                d = d.expand(shape)\n"
            method_str += f"            if event > 0:\n"
            method_str += f"                d = d.to_event(event)\n"
            # --- End of common logic ---

            # --- This is the new switch ---
            # If the user wants the raw object, we return it now.
            method_str += f"            if create_obj:\n"
            method_str += f"                return d\n"
            
            # Otherwise, we proceed with the original behavior: creating a sample site.
            method_str += f"            else:\n"
            method_str += f"                infer_dict = {{'obs_mask': mask}} if mask is not None else None\n"
            method_str += f"                return numpyro.sample(name, d, obs=obs, infer=infer_dict)\n"

            file.write(method_str + "\n")
        except (ValueError, TypeError) as e:
            print(f"Could not generate wrapper for '{name}': {e}")

print("Successfully generated 'dists.py'")
# %%
