#%%
from BI import bi
#%%
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

    dist_doc[name] = name.__doc__

#%%
import json
with open("pythonDoc.json", "w") as f:
    json.dump(dist_doc, f)


#%%%
# From the json file generate, withe the bellow prompt we I asked roxygen formatting from gemini 2 pro