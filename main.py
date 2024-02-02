
from code.data import*
from code.build import*
from code.fit import*
from code.diagnostic import*
import pandas as pd
import numpy as np  
import tensorflow as tf
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
                 gpu = True,               
                 **kwargs):      
        self.f = formula
        if df is None:
         self.df = pd.DataFrame({'A' : []})
        else:
            if isinstance(df, pd.DataFrame):
                self.df = df
            elif isinstance(df, str):
                self.df = self.import_csv(df, **kwargs)
            
        self.model_path = 'output/mymodel.py'
        self.df_path = 'output/mydf.csv'
        self.data_modification = {}
        if float == 16:
            self.float = tf.float16
            self.int = tf.int16
        if float == 32:
            self.float = tf.float32
            self.int = tf.int32
        if float == 64:
            self.float = tf.float64
            self.int = tf.int64

        # GPU configuration ----------------------------
        if gpu:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if formula is not None:
            self.formula(self.f)
            self.build_model()

    def build_model(self):
        # Gather formula input informations
        self.get_var()
        self.get_priors_names()
        self.get_model_type()
        self.get_mains_info() 
        
        #self.output_path = 'output/mydf.csv'
        #self.df.to_csv(self.output_path, index=False)

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
        
        self.posterior, self.trace, self.sample_stats = self.run_model(observed_data,
                       parallel_iterations=parallel_iterations,
                       num_results=num_results,
                       num_burnin_steps=num_burnin_steps,
                       step_size=step_size,
                       num_leapfrog_steps=num_leapfrog_steps,
                       num_adaptation_steps=num_adaptation_steps,
                       num_chains=num_chains)
        #self.posterior = posterior
        #self.trace = trace
        #self.sample_stats = sample_stats
