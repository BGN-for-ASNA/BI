#from code.model_diagnostic import *
#from code.model_fit import *
from code.class_write import*
from code.data_manip import data
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

def joineDis(tensor, text, df):
    df = df
    print(text)
    for key in text.keys():
        tensor[key] =  eval(f"{text[key]}")
    tensor = tfd.JointDistributionNamed(tensor)
    return tensor

class model(data, define, write):
    def __init__(self, 
                 formula = None, 
                 df = None,
                 float = 16,
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
        self.float = float
        
        # GPU configuration ----------------------------
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
        	tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        # If user setup directly a formula
        if formula is not None:
            self.formula(self.f)
            self.build_model()
        
    def build_model(self):
        self.get_var()
        self.get_priors_names()
        self.get_mains_info() 
        
        self.output_path = 'output/mydf.csv'
        self.df.to_csv(self.output_path, index=False)

        self.tensor_prior()
        self.write_main_text()
        for key in self.main_text.keys():
            self.tensor[key] = self.create_function_from_string(func_str = self.main_text[key], name = key)
        self.tensor = tfd.JointDistributionNamed(self.tensor)
        #new_tensor = joineDis(self.tensor, self.main_text, self.df)
        #self.tensor = new_tensor
        #print("Model builded")
        #return new_tensor
        
    def sample(self, *args, **kwargs):
        self.samples = self.tensor.sample(*args, **kwargs)
        return self.samples
        

