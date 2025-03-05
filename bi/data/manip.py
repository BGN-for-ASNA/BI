import pandas as pd
import jax.numpy as jnp
import numpy as np
import inspect
import jax

class manip():
    def __init__(self):
        self.data_modification = {}
        self.pandas_to_jax_dtype_map = {
            'int64': jnp.int64,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'float64': jnp.float64,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }
    # Import data----------------------------
    def data(self, path, **kwargs):
        self.data_original_path = path
        self.data_args = kwargs
        self.df = pd.read_csv(path, **kwargs)
        self.data_modification = {}
        return self.df
   
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
        self.index_map = {}
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            for a in range(len(colCat)):                
                self.df["index_"+ colCat[a]] =  self.df.loc[:,colCat[a]].astype("category").cat.codes
                self.df["index_"+ colCat[a]] = self.df["index_"+ colCat[a]].astype(np.int64)
                self.index_map[colCat[a]] = dict(enumerate(self.df[colCat[a]].astype("category").cat.categories ) )
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            for a in range(len(cols)):
                self.df["index_"+ cols[a]] =  self.df.loc[:,cols[a]].astype("category").cat.codes
                self.df["index_"+ cols[a]] = self.df["index_"+ cols[a]].astype(np.int64)

                self.index_map[cols[a]] = dict(enumerate(self.df[cols[a]].astype("category").cat.categories ) )

        self.df.columns = self.df.columns.str.replace('.', '_')
        self.df.columns = self.df.columns.str.replace(' ', '_')

        self.data_modification['index'] = cols # store info of indexed columns
        
        return self.df
    
    @jax.jit
    def scale_var(self, x):
        return (x - x.mean()) / x.std()

    def scale(self, x = None, cols = 'all'):
        if x is not None:
            return self.scale_var(x)
        else:
            if cols == 'all':
                for col in self.df.columns:                
                    self.df.loc[:, col] = (self.df.loc[:,col] - self.df.loc[:,col].mean())/self.df.loc[:,col].sd()

            else:
                for a in range(len(cols)):
                    self.df.loc[:, cols[a]] = (self.df.loc[:, cols[a]] - self.df.loc[:, cols[a]].mean()) / self.df.loc[:, cols[a]].std()


            self.data_modification['scale'] = cols # store info of scaled columns

            return self.df
    
    def to_float(self, cols = 'all', type = 'float32'):
        if cols == 'all':
            for col in self.df.columns:                
                self.df.loc[:, col] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.loc[:,cols[a]].astype(type)


        self.data_modification['float'] = cols # store info of scaled columns
        
        return self.df

    def to_int(self, cols = 'all', type = 'int32'):
        if cols == 'all':
            for col in self.df.columns:                
                self.df.iloc[:, cols] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.iloc[:,cols[a]].astype(type)


        self.data_modification['int'] = cols # store info of scaled columns

    def pd_to_jax(self, model, bit = '32'):
        params = inspect.signature(model).parameters
        args_without_defaults = []
        args_with_defaults = {}
        for param_name, param in params.items():
            if param.default == inspect.Parameter.empty:
                args_without_defaults.append(param_name)
            else:
                args_with_defaults[param_name] = (param.default, type(param.default).__name__)

        test = all(elem in self.df.columns for elem in args_without_defaults)
        result = dict()
        if test:
            for arg in args_without_defaults:
                varType = str(self.df[arg].dtype)
                result[arg] = jnp.array(self.df[arg], dtype = self.pandas_to_jax_dtype_map.get(varType))
        else:
            return "Error, no"

        for k in args_with_defaults.keys():
            print(args_with_defaults[k][1])
            result[k] = jnp.array(args_with_defaults[k][0], dtype =self.pandas_to_jax_dtype_map.get(str(args_with_defaults[k][1]) + bit))

        return result     

    def data_to_model(self, cols):
        jax_dict = {}
        for col in cols:
            jax_dict[col] = jnp.array(self.df.loc[:,col].values)
        self.data_modification['data_on_model'] = cols # store info of data used in the model
        self.data_on_model = jax_dict
