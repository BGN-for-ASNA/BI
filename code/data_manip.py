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

#%%
class data():
    # Import data----------------------------
    def import_csv(self, path, **kwargs):
        self.df_original_path = path
        self.df_args = kwargs
        self.df = pd.read_csv(path, **kwargs)
        return self.df
   
    # Data manipulation----------------------------
    def OHE(self, cols = 'all'):
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            OHE = pd.get_dummies(self.df, columns=colCat, dtype=int)
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            OHE = pd.get_dummies(self.df, columns=cols, dtype=int)

        OHE.columns = OHE.columns.str.replace('.', '_')
        OHE.columns = OHE.columns.str.replace(' ', '_')


        self.df = pd.concat([self.df , OHE], axis=1)
        self.data_modification['OHE'] = cols
        return OHE

    def index(self, cols = 'all'):
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            for a in range(len(colCat)):
                self.df["index_"+ colCat[a]] =  self.df.loc[:,colCat[a]].astype("category").cat.codes
                self.df["index_"+ colCat[a]] = self.df["index_"+ colCat[a]].astype(np.int64)
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            for a in range(len(cols)):
                self.df["index_"+ cols[a]] =  self.df.loc[:,cols[a]].astype("category").cat.codes
                self.df["index_"+ cols[a]] = self.df["index_"+ cols[a]].astype(np.int64)

        self.df.columns = self.df.columns.str.replace('.', '_')
        self.df.columns = self.df.columns.str.replace(' ', '_')

        self.data_modification['index'] = cols
        return self.df

