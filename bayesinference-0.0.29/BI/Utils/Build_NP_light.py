# %%
"""
This file wraps NumPyro distribution functions to enable different calls depending on the provided arguments (sampling, model building, etc.). It also generates the documentation for each distribution function.

The file created, `dists.py`, is edited by hand to handle some issues:
1. The `mask` argument is present twice in some functions, so it is removed.
2. `*args` and `**kwargs` are not the last arguments in some functions.
3. Some function names are changed due to camel case conversion to snake case (e.g., `l_k_j` -> `lkj`).

All required changes can be tracked through the VSCode Problems list. `dists.py` is then renamed to `np_dists.py`, which is the file used by BI to import the distributions.

"""

import inspect
import re
import ollama
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

# --- Filter for functions in the distributions module ---
dist_functions = {}
for name in dir(numpyro.distributions):
    obj = getattr(numpyro.distributions, name)
    if inspect.isfunction(obj) and not name.startswith("_"):
        dist_functions[name] = obj


dist_classes = {**dist_classes, **dist_functions}
dist_classes['NegativeBinomial'] = numpyro.distributions.conjugate.NegativeBinomial

#%%

def camel_to_snake(name):
    name = name.strip()  # Remove trailing/leading spaces
    s1 = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    return s1.lower()

def generate_doc_string(name, dist_class, name_snake):
    fn_source_code = inspect.getsource(dist_class)
    prompt = f"""
    As an expert Python programmer specializing in the NumPyro library, your task is to generate a Google-style docstring  for the provided Python function source code. **GENERATE ONLY THE DOCSTRING NOTHING ELSE**.

    Follow these instructions precisely:

    1.  **Analyze the Function**: From the `{fn_source_code}` provided below, identify the statistical distribution it  represents and its required arguments. Note that numpyro use jax.numpy.array.

    2.  **Construct the Docstring**: Generate a complete Google-style docstring with the following sections in this exact   order:

        *   **Summary**: A concise,  summary of what the function does (e.g., "Samples from a [Distribution Name]   distribution.") with general description of the distribution.

        *   **Distribution Formula**: Include the LaTeX formula for the probability density function (PDF) or probability   mass function (PMF) of the distribution, formatted within a `.. math::` block.

        *   **Args**: List and describe all arguments in list format with empty line between each arguments. 

            *   **Distribution Args**: First, list the distribution's own parameters (like `loc` or `scale`). 

            *   **Sampling / Modeling Args**: Then, include the following four standard arguments with their exact descriptions:

                *   `shape` (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.

                *   `event` (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).

                *   `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.

                *   `create_obj` (bool): If True, returns the raw NumPyro distribution object instead of creating a sample  site. This is essential for building complex distributions like `MixtureSameFamily`.

        *   **Returns**: 
           *   **When `sample=False`**: A NumPyro {name} distribution object (for model building).

           *   **When `sample=True`**: A JAX array of samples drawn from the {name} distribution (for direct sampling).

           *   **When `create_obj=True`**: The raw NumPyro distribution object (for advanced use cases).

        *   **Example Usage**: Provide a code block showing a simple, direct sampling call to the function. The structure must be:
                from BI import bi
                m = bi('cpu')
                m.dist.{name_snake}(required_args_with_defaults, sample=True)
 
                - Replace only `required_args_with_defaults` with the distribution's essential parameters set to reasonable  default values (e.g., `loc=0.0, scale=1.0`).

        *   **Wrapper of:**: Add a final note pointing to the official NumPyro documentation, using this exact URL structure:  `https://num.pyro.ai/en/stable/distributions.html#function_name_lowercase`
            - Replace `function_name_lowercase` with the distribution's name in PascalCase (e.g., Normal, Bernoulli).

    3.  **Final Output**: Return **only the complete docstring** and nothing else. Do not include any explanatory text or   the original function code in your response.

    **Function Source Code:**
    ```python
    {fn_source_code}
    ```
        """

    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        options={
        'temperature': 0  # lower = more deterministic
        }
    )
    return response['message']['content']
Doc = {}

# %%
# up to 2h to run

for name, dist_class in dist_classes.items():
    if name not in Doc.keys():
        if name not in ["SineBivariateVonMises"]:
            print(f"building doc for {name}")
            doc = generate_doc_string(name, dist_class, camel_to_snake(name) )
            print(doc)
            Doc[name] = doc


# %%
import json
with open('Doc_python.json', 'w') as f:
    json.dump(Doc, f)
# %%
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
        if name in Doc:
            try:
                signature = inspect.signature(dist_class)
                parameters = signature.parameters
                param_str = ", ".join([str(param) for param in parameters.values()])
    
                # <-- MODIFIED: Renamed 'to_event_dims' to 'event' and added 'mask'
                wrapper_args = "name='x', obs=None, mask=None, sample=False, seed=0,    shape=(), event=0,create_obj=False"
                full_signature = f"{param_str}, {wrapper_args}"
    
                method_name = camel_to_snake(name)
                method_str = f"    @staticmethod\n"
                method_str += f"    def {method_name}({full_signature}):\n"
    
                # <-- MODIFIED: Updated docstrings for 'event' and 'mask'
                #docstring_parts = [f"{name} distribution wrapper."]
                #docstring_parts.append("\n    Original Arguments:\n        -----------------")
                #for param in parameters.values():
                #    desc =  f"{param.name}: {param.default}\n"
                #    indented_desc = '\n        '.join(desc.split('\n'))
                #    docstring_parts.append(f"    {indented_desc}")
                #docstring_parts.append("\n    Wrapper Arguments:\n     ------------------")
                #docstring_parts.append("    shape (tuple): A multi-purpose argument    for shaping.")
                #docstring_parts.append("        - When sample=False (model     building), this is used with `.expand(shape)` #to set the   distribution's batch shape.")
                #docstring_parts.append("        - When sample=True (direct     sampling), this is used as `sample_shape` to #draw a raw JAX array  of the given shape.")
                #docstring_parts.append("    event (int): The number of batch   dimensions to reinterpret as event #dimensions (used in model     building).")
                #docstring_parts.append("    mask (jnp.ndarray, bool): Optional     boolean array to mask observations. This #is passed to the `infer=  {'obs_mask': ...}` argument of `numpyro.sample`.")
                #docstring_parts.append("    create_obj (bool): If True, returns the    raw NumPyro distribution object #instead of creating a sample site.")
                #docstring_parts.append("        This is essential for building     complex distributions like #`MixtureSameFamily`.")
    
                #full_docstring = "\n".join(docstring_parts)
                full_docstring = Doc[name]
                indented_full_docstring = '\n        '.join(full_docstring.split    ('\n'))
                method_str += f'        \n        {indented_full_docstring}\n           \n'
    
                arg_names = [param.name for param in parameters.values()]
                arg_str = ", ".join([f"{arg}={arg}" for arg in arg_names])
                method_str += f"        d = {str(dist_class.__module__)}.{str   (dist_class.__name__)}({arg_str})\n"
    
                # The `sample` flag is for direct JAX array sampling, completely    separate from model building.
                method_str += f"        if sample:\n"
                method_str += f"            seed_key = random.PRNGKey(seed)\n"
                method_str += f"            return d.sample(seed_key,   sample_shape=shape)\n"
    
                # This `else` block now handles ALL model-building logic (creating  sample sites OR objects).
                method_str += f"        else:\n"
    
                # --- This is the common logic for modifying the distribution object    ---
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
                
                # Otherwise, we proceed with the original behavior: creating a  sample site.
                method_str += f"            else:\n"
                method_str += f"                infer_dict = {{'obs_mask': mask}} if    mask is not None else None\n"
                method_str += f"                return numpyro.sample(name, d,  obs=obs, infer=infer_dict)\n"
    
                file.write(method_str + "\n")
            except (ValueError, TypeError) as e:
                print(f"Could not generate wrapper for '{name}': {e}")

print("Successfully generated 'dists.py'")
# %%
