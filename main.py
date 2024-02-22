
from code.data import*
from code.build import*
from code.fit import*
from code.diagnostic import*
import pandas as pd
import numpy as np  
import tensorflow as tf
from tensorflow.python.client import device_lib

class model(data, define, write, fit, diagnostic):
    def __init__(self, 
                 formula = None, 
                 df = None,
                 float = 32,  
                 gpu = False,               
                 **kwargs):      
        self.f = formula
        self.Tensoflow = False     
        if df is None:
            self.df = pd.DataFrame({'A' : []})
        else:
            if isinstance(df, pd.DataFrame):
                self.df = df
            elif isinstance(df, str):
                self.df = self.import_csv(df, **kwargs)
            self.df = self.convert_to_float(self.df)         

        self.model_path = 'output/mymodel.py'
        self.df_path = 'output/mydf.csv'
        self.data_modification = {}

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

        self.model_info = {}
        self.model_info["multiple_lks"] = False
        self.model_info["with_indices"] = False
        self.model_info["indices"] = {}   
        self.model_info['Multilevel'] = False
        self.model_info['Multilevel_diag'] = {}             
        self.model_dict = {}
        self.prior_dict = {}
        self.priors_name = []
        # GPU configuration ----------------------------
        self.gpu = gpu
        
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

    def convert_to_float(self, df):
        for col in df.columns:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                df[col] = df[col].astype(float)
        return df
    
    def build_model(self):
        # Gather formula input informations
        self.get_model_info()
        self.merge()
        self.write_tensor()
        
    def sample(self, *args, **kwargs):
        self.samples = self.tensor.sample(*args, **kwargs)
        return self.samples
    
    def log_prob(self, *args, **kwargs):
        self.prob = self.tensor.log_prob(*args, **kwargs)
        return self.prob
    
    def fit(self, observed_data,
            init = None,
            bijectors = None,
            parallel_iterations=1,
            num_results=2000,
            num_burnin_steps=500,
            step_size=0.065,
            num_leapfrog_steps=5,
            num_adaptation_steps=400,
            num_chains=4):
        if self.gpu: 
            if len(self.devices['GPU']) > 0:
                with tf.device(next(iter(self.devices['GPU'].values()))):
                    self.posterior, self.trace, self.sample_stats = self.run_model(observed_data,
                                    params = self.priors_dict.keys(),
                                    init = init,
                                    bijectors = bijectors,
                                    parallel_iterations=parallel_iterations,
                                    num_results=num_results,
                                    num_burnin_steps=num_burnin_steps,
                                    step_size=step_size,
                                    num_leapfrog_steps=num_leapfrog_steps,
                                    num_adaptation_steps=num_adaptation_steps,
                                    num_chains=num_chains)
        else:
            with tf.device(next(iter(self.devices['CPU'].values()))):
                self.posterior, self.trace, self.sample_stats = self.run_model(observed_data,
                                params = self.priors_name,
                                init = init,
                                bijectors = bijectors,
                                parallel_iterations=parallel_iterations,
                                num_results=num_results,
                                num_burnin_steps=num_burnin_steps,
                                step_size=step_size,
                                num_leapfrog_steps=num_leapfrog_steps,
                                num_adaptation_steps=num_adaptation_steps,
                                num_chains=num_chains)

