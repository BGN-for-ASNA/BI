#%%
import pandas as pd
def OHE(d, cols = 'all'):
    if cols == 'all':
        colCat = list(d.select_dtypes(['object']).columns)    
        OHE = pd.get_dummies(d, columns=colCat, dtype=int)
    else:
        OHE = pd.get_dummies(d, columns=cols, dtype=int)
    
    OHE.columns = OHE.columns.str.replace('.', '_')
    OHE.columns = OHE.columns.str.replace(' ', '_')
    return OHE
    