#%%
import pandas as pd
def OHE(d):
    colCat = list(d.select_dtypes(['object']).columns)    
    OHE = pd.get_dummies(d, columns=colCat, dtype=int)
    return OHE
    
# %%
import tensorflow
import tensorflow_probability


# %%
