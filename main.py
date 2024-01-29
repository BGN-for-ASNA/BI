#from code.model_diagnostic import *
#from code.model_fit import *
from code.class_write import*
from code.data_manip import data
import pandas as pd
import numpy as np  

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

class model(data, define, write):
    def __init__(self, formula = None, float = 16):      
        self.f = formula
        self.df = pd.DataFrame({'A' : []})
        self.model_path = 'output/mymodel.py'
        self.df_path = 'output/mydf.csv'
        self.data_modification = {}
        self.float = float
    
    def build_model(self):
        self.get_var()
        self.get_priors_names()
        self.get_mains_info() 
        
        self.output_path = 'output/mydf.csv'
        self.df.to_csv(self.output_path, index=False)
        
        #self.write_header()
        #self.write_priors()
        self.tensor_prior()
        self.write_main_text()
        self.tensor_main()
        self.tensor =  tfd.JointDistributionNamed(self.tensor)
        #self.write_main2()       
        
        #import importlib
        #from output import mymodel
        #importlib.reload(mymodel)
        #from output.mymodel import m
        #self.tfp = m
        print("Model builded")
        
    def sample(self, *args, **kwargs):
        #sample = self.tfp.sample(*args, **kwargs)
        #print(sample)
        #result = {}
        #for key in sample.keys():
        #    result[key] = sample[key].numpy().reshape(sample[key].shape[1],sample[key].shape[0])[0]
        self.samples = self.tensor.sample(*args, **kwargs)
        return self.samples
        

