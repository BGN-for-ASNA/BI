#%%
"""This script generates Python code examples from docstrings of distribution classes in the BI library.
It uses the `ollama` model to extract code examples and save them in a Python file for testing.
The generated code examples are designed to be compatible with JAX and the BI library's distribution methods.
The script also handles potential errors during the execution of the generated code examples.
"""
import json
import ollama
import inspect
import os
import sys
newPath = os.path.dirname(os.path.abspath(""))
if newPath not in sys.path:
    sys.path.append(newPath)
from BI import bi
m = bi('cpu')
dist_classes = {}
for name in dir(m.dist):
    obj = getattr(m.dist, name)
    if (
        obj and not name.startswith("_")
        and name not in ["Distribution", "ExpandedDistribution", "TransformedDistribution", "IndependentDistribution"]
    ):
        dist_classes[name] = obj
dist_classes
#%%
def generate_code_example(docstring):
    prompt = f"""
    Given the following python docstring extract the code example from it.
    remove the following lines:
        - from BI import bi
        - m = bi('cpu')
    Note that sample = True must all the time present (e.g. m.dist.normal(loc = 0, scale = 1, sample = True))
    Return only the code example, do not wrap it in a code block.
    {docstring}
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


#%%
test = {}
for name, doc in dist_classes.items():
    print(f"Generating code example for {name}")
    test[name] = generate_code_example(dist_classes[name].__doc__)

#%%
with open("test.py", "w") as file:
    file.write(
"""
import os
import sys
newPath = os.path.dirname(os.path.dirname(os.path.abspath("")))
if newPath not in sys.path:
    sys.path.append(newPath)
from BI import bi
import jax.numpy as jnp
m = bi('cpu')
erros = []
"""
)

    for name, doc in test.items():
        file.write(f"try:\n")
        file.write(f"    {test[name]}\n")
        file.write(f"except Exception as e:\n")
        file.write(f"    print(f'Error in {name}: {{e}}')\n")
        file.write(f"    erros.append(f'Error in {name}: {{e}}')\n")
        file.write(f"\n")
        
    file.write(f"print(erros):\n")