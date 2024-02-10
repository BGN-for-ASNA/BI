
from code.data import*
from code.build import*
from code.fit import*
from code.diagnostic import*
import pandas as pd
import numpy as np  
import tensorflow as tf
from tensorflow.python.client import device_lib
## Distribution functions -----------------------------------------------------
def get_distribution_classes():
    # Get all names defined in the distributions module
    all_names = dir(tfd)
    
    # Filter to include only classes
    class_names = [name for name in all_names if isinstance(getattr(tfd, name), type)]
    
    # Create a dictionary of class names and corresponding classes
    class_dict = {name: getattr(tfd, name) for name in class_names}
    
    return class_dict
tf_classes = get_distribution_classes()

def exportTFD(tf_classes):
    for key in tf_classes.keys():
        globals()[key] = tf_classes[key]
exportTFD(tf_classes)

class model(data, define, write, fit, diagnostic):
    def __init__(self, 
                 formula = None, 
                 df = None,
                 float = 32,  
                 gpu = False,               
                 **kwargs):      
        self.f = formula
        if df is None:
         self.df = pd.DataFrame({'A' : []})
        else:
            if isinstance(df, pd.DataFrame):
                self.df = df
            elif isinstance(df, str):
                self.df = self.import_csv(df, **kwargs)
            # Select columns with 'float64' dtype  
            float64_cols = list(self.df.select_dtypes(include='float'+ str(float)))
            # The same code again calling the columns
            self.df[float64_cols] = self.df[float64_cols].astype('float'+ str(float))

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
            self.formula(self.f)
            self.build_model()

    def build_model(self):
        # Gather formula input informations
        self.get_var()
        self.get_priors_names()
        self.get_model_type()
        self.get_undeclared_params()
        self.get_mains_info() 
        
        # Formula input to tensorflow probability model
        self.tensor_prior()
        self.write_main_text()
        for key in self.main_text.keys():
            self.tensor[key] = self.create_function_from_string(func_str = self.main_text[key], name = key)

        self.tensor_dict = self.tensor
        self.tensor = tfd.JointDistributionNamed(self.tensor)
        
    def sample(self, *args, **kwargs):
        self.samples = self.tensor.sample(*args, **kwargs)
        return self.samples
    
    def log_prob(self, *args, **kwargs):
        self.prob = self.tensor.log_prob(*args, **kwargs)
        return self.prob
    
    def fit(self, observed_data,
            parallel_iterations=1,
            num_results=2000,
            num_burnin_steps=500,
            step_size=0.065,
            num_leapfrog_steps=5,
            num_adaptation_steps=400,
            num_chains=4):
        if self.gpu: 

            if len(devices['GPU']) > 0:
                with tf.device(next(iter(self.devices['GPU'].values()))):
                    self.posterior, self.trace, self.sample_stats = self.run_model(observed_data,
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
                                parallel_iterations=parallel_iterations,
                                num_results=num_results,
                                num_burnin_steps=num_burnin_steps,
                                step_size=step_size,
                                num_leapfrog_steps=num_leapfrog_steps,
                                num_adaptation_steps=num_adaptation_steps,
                                num_chains=num_chains)

