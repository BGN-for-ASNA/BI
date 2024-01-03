#%%
import pandas as pd
import numpy as np
def OHE(d, cols = 'all'):
    if cols == 'all':
        colCat = list(d.select_dtypes(['object']).columns)    
        OHE = pd.get_dummies(d, columns=colCat, dtype=int)
    else:
        OHE = pd.get_dummies(d, columns=cols, dtype=int)
    
    OHE.columns = OHE.columns.str.replace('.', '_')
    OHE.columns = OHE.columns.str.replace(' ', '_')
    return OHE

def index(d, cols = 'all'):
    index = d
    if cols == 'all':
        colCat = list(d.select_dtypes(['object']).columns)    
        for a in range(len(colCat)):
            d["index_"+ colCat[a]] =  d.loc[:,colCat[a]].astype("category").cat.codes
            d["index_"+ colCat[a]] = d["index_"+ colCat[a]].astype(np.int64)
    else:
        for a in range(len(cols)):
            d["index_"+ cols[a]] =  d.loc[:,cols[a]].astype("category").cat.codes
            d["index_"+ cols[a]] = d["index_"+ cols[a]].astype(np.int64)

    d.columns = d.columns.str.replace('.', '_')
    d.columns = d.columns.str.replace(' ', '_')
    return d