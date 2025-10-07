#%%
from BI import bi

m = bi()
dist_doc = {}
no=["__class__",
     "__delattr__",
     "__dict__",
     "__dir__",
     "__doc__",
     "__eq__",
     "__format__",
     "__ge__",
     "__getattribute__",
     "__getstate__",
     "__gt__",
     "__hash__",
     "__init__",
     "__init_subclass__",
     "__le__",
     "__lt__",
     "__module__",
     "__ne__",
     "__new__",
     "__reduce__",
     "__reduce_ex__",
     "__repr__",
     "__setattr__",
     "__sizeof__",
     "__str__",
     "__subclasshook__",
     "__weakref__",
     "sineskewed"]
for name in dir(m.dist):
    if name in no:
        continue
    dist_doc[name] = m.dist.__dict__[name].__doc__



#%%
import json
with open("pythonDoc.json", "w") as f:
    json.dump(dist_doc, f)


#%%%
# From the json file generate, withe the bellow prompt we I asked roxygen2 formatting from gemini 2 pro with formated output option:
"""
You are an expert code conversion assistant. Your task is to convert Python docstrings (in Google style) into R's roxygen2 documentation format.
Instructions:
For each key-value pair in the input dictionary:
The key is the function's name.
The value is the Python docstring.
Convert the docstring into a complete roxygen2 comment block, where each line begins with #'.
Use the following mapping:
Python (Summary Line) -> R (@title)
Python (Description Body) -> R (@details)
Python (Args:) -> R (@param name description) for each argument.
Python (Returns:) -> R (@return description)
The @examples tag is mandatory and must be constructed as follows:
bi.dist.[function_name]([parameter_names], sample = TRUE)
Replace [function_name] with the key from the dictionary.
Replace [parameter_names] with a comma-separated list of the actual parameter names from the @param section.
"""