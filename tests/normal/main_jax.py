#%%
import random as r
from data import*
from build_jax import*
from fit_jax import*
from diagnostic import*
import pandas as pd
import numpy as np  
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import re


#%%
class model(data, define, write, fit, diagnostic):
    def __init__(self, 
                 formula = None, 
                 df = None,
                 float = 32,  
                 cpu = None,               
                 **kwargs): 

        if cpu is None: # Set up maximum number of cores
            Ncores = os.cpu_count()    
            platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
            jax.config.update("jax_platform_name", platform)  
            xla_flags = os.getenv("XLA_FLAGS", "")
            xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
            os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(Ncores)] + xla_flags)
            jax.config.update("jax_platform_name", platform)
            print('jax.local_device_count ',jax.local_device_count(backend=None))

        self.f = formula
        self.Tensoflow = False 

        if float == 16:
           self.float_prior = tf.float16
           self.int = tf.int16
        if float == 32:
           self.float_prior = tf.float32
           self.int = tf.int32
        if float == 64:
           self.float_prior = tf.float64
           self.int = tf.int64
        self.float = float  

        if df is None:
            self.df = pd.DataFrame({'A' : []})
        else:
            if isinstance(df, pd.DataFrame):
                self.df = df
            elif isinstance(df, str):
                self.df = self.import_csv(df, **kwargs)
            #self.df = self.convert_to_float(self.df)  
            self.df = self.change_float_precision(self.df, self.float)       

        self.model_path = 'output/mymodel.py'
        self.df_path = 'output/mydf.csv'
        self.data_modification = {}



        self.model_info = {}
        self.model_info["multiple_lks"] = False
        self.model_info["with_indices"] = False
        self.model_info["indices"] = {}   
        self.model_info['Multilevel'] = False
        self.model_info['Multilevel_diag'] = {}       
        self.model_info['Multilevel_indices'] = {}    
        self.model_info['Multilevel_indices_dim'] = {}
        self.model_info['Categorical'] = False
        self.model_info['catN'] = 0
        self.model_dict = {}
        self.prior_dict = {}
        self.priors_name = []
        
        # GPU configuration ----------------------------
        #self.gpu = gpu
        
        local_device_protos = device_lib.list_local_devices()
        self.devices = {}
        cpu = {}
        gpu = {}
        for a in range(len(local_device_protos)):
            if local_device_protos[a].device_type == 'CPU':
                cpu[str(a)] = local_device_protos[a].name.replace('/device:', '')
            else:
                gpu[str(a)] =  local_device_protos[a].name.replace('/device:', '')
        self.devices['CPU'] = cpu
        self.devices['GPU'] = gpu
                  
        if formula is not None:
            #self.formula(self.f)
            self.build_model()

    def change_float_precision(self, df, precision=64):
        # Get columns of integer and float types
        int_columns = df.select_dtypes(include='int').columns
        float_columns = df.select_dtypes(include='float').columns
    
        # Convert integer columns to float with specified precision
        for col in int_columns:
            df[col] = df[col].astype(f'float{precision}')
    
        # Convert float columns to float with specified precision
        for col in float_columns:
            df[col] = df[col].astype(f'float{precision}')
    
        return df
    
    def convert_to_float(self, df):
        for col in df.columns:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                df[col] = df[col].astype('float' + str(self.float))
        return df
    
    def build_model(self):
        # Gather formula input informations
        self.get_model_info()
        self.merge()

        if self.model_info['Multilevel']:            
            self.extract_mvn_dimetion()

        self.write_tensor()
        
    def sample(self, *args, **kwargs):
        init_key, sample_key = random.split(random.PRNGKey(int(r.randint(0, 10000000))))
        self.samples = self.tensor.sample(seed=jnp.array(init_key), *args, **kwargs)
        return self.samples

    
    def log_prob(self, *args, **kwargs):
        self.prob = self.tensor.log_prob(*args, **kwargs)
        return self.prob
    
    def fit(self, observed_data, num_chains=4):
        self.obs_names = list(observed_data.keys())[0]
        self.observed_data_jax = jnp.array(list(observed_data.values())[0])
        self.res = self.parallele_chains(num_chains)
        self.posterior, self.sample_stats = self.res 
        p = dict(zip(self.tensor._flat_resolve_names(), self.posterior))
        self.az_trace = self.tfp_trace_to_arviz(self.posterior, self.sample_stats, p)    


