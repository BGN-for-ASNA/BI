from code.model_diagnostic import *
from code.model_fit import *
from code.model_write import *
from code.data_manip import *
import tensorflow as tf
import arviz as az
from tensorflow_probability import distributions as tfd
import pandas as pd
import numpy as np  
import re

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
        print(self.undeclared_params)
        self.get_priors_names()
        self.get_mains_info() 
        self.write_header()
        self.write_priors()
        self.main_text()
        self.write_main2()

