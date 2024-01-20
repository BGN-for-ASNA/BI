#from code.model_diagnostic import *
#from code.model_fit import *
from code.class_write import*
from code.data_manip import data
import pandas as pd
import numpy as np  


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
        
        self.write_header()
        self.write_priors()
        self.main_text()
        self.write_main2()
        
        
        import importlib
        from output import mymodel
        importlib.reload(mymodel)
        from output.mymodel import m
        self.tfp = m
        
        print("Model builded")

