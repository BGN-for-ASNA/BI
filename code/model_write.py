#%%
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
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

## Formula functions -----------------------------------------------------
def get_formula(formula = "y~Normal(0,1)", type = 'likelihood'):
    y, x = re.split(r'[~]',formula)
    y = y.replace(" ", "")
    x = x.replace(" ", "")
    if 'likelihood' in type: 
        if x.find('(') == -1: # If parenthesis then we concider the presence of a distribution
            args = re.split(r'[+*()*,]',x)
            new = []
            for i in range(len(args)):
                if args[i] != '':
                    new.append(args[i])
            return [y, new]     
        else:
            args = re.split(r'[+*()*,]',x)
            new = []
            dist = []
            for i in range(len(args)):
                if args[i] != '':
                    if args[i] in list(tf_classes.keys()):
                        dist.append('tf.' + str(args[i]))
                    else:
                        new.append(args[i])                        
            return [y,dist, new] 
            
    else:
        dist, args = x.split('(')
        dist = dist.replace(" ", "")
        args = args.replace("(", "")
        args = args.replace(")", "")
        args = args.split(",")
        return [y, dist, args]
    
def get_var(model):
    full_model = {}
    # Extract variables
    for  key in model.keys():
         full_model[key] = dict(
             input = model[key],
             var = get_formula(formula=model[key], type=key)
             ) 
    return full_model   

def get_undeclared_params(model, df):
    Vars = []
    params = []
    for key in model.keys():
        if 'main' in key:
            if df.empty: 
                tmp = model[key]['var'][2:]
            else:
                tmp = model[key]['var']
        else:
            tmp = model[key]['var']
  
        for a in range(len(tmp)):
            if isinstance(tmp[a], list):
                params.append(tmp[a])
            else:
                if a == 0:
                    Vars.append(tmp[0].replace(' ', ''))
                                    
    params = [item.replace(' ', '') for sublist in params for item in sublist]
    undeclared_params = list(set(Vars) ^ set(params))
    undeclared_params2 = []
    for a in range(len(undeclared_params)):
        try:
            x = float(undeclared_params[a])
        except ValueError:
            if undeclared_params[a].find('tf.') == -1:
                undeclared_params2.append(undeclared_params[a])
    
    if df.empty:  
        return undeclared_params2
    else:
        test = pd.Index(undeclared_params2).difference(df.columns).tolist()
        test2 =  list(set(undeclared_params2) & set(df.columns))
        return {'undeclared_params': test, 'params_in_data' : test2}  
        
## Write model functions-----------------------------------------------------
def get_likelihood(model, main_name):
    result = []
    for key in model.keys():
        if 'likelihood' in key:
            name = model[key]['var'][0]
            if name in model[main_name]['var'][2]:
                index = model[main_name]['var'][2].index(name)
                y, x = re.split(r'[~]',model[key]['input'])
                x = x.replace(" ", "")
                print(type(x) == str)
                print(x)
                
                if x.find('(') != -1:
                    x = 'tfd.Sample(tfd.' + x + ', sample_shape=1)'
                    
                result.append(x)
                result.append(index)
    if len(result) >= 1:
        return result
    else:
        return None
    
def write_header(output_file, float):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow as tf")    
        file.write('\n')
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')  
        file.write("from model_fit import *")    
        file.write('\n')
        file.write("from model_diagnostic import *")    
        file.write('\n')
        
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

def write_header_with_dataFrame(output_file, DFpath, data, float, sep):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow as tf")    
        file.write('\n')
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')        
        file.write("import pandas as pd")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        file.write("from model_fit import *")    
        file.write('\n')
        file.write("from model_diagnostic import *")    
        file.write('\n')
        
        file.write("d = pd.read_csv('" + DFpath +"', sep = '" + sep + "')")    
        file.write('\n')
        
    for a in range(len(data)):
        with open(output_file,'a') as file:
            #file.write(data[a] + "= tf.convert_to_tensor(d." + data[a] + ", dtype = tf.float" + str(float) + ")")  
            file.write(data[a] + "= d." + data[a] )  
            file.write('\n')
        
    with open(output_file,'a') as file:   
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

def write_priors(model, output_file, float):    
    p = [] # Store model parameters name
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['var']        
        if 'prior' in key.lower():
            p.append(var[0])
            with open(output_file,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + 
                           " = tfd.Sample(tfd." + var[1] + "(" + 
                            #str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]])) + "), sample_shape=1),")
                            str(', '.join([f'{item}' for item in var[2]])) + "), sample_shape=1),")

                file.write('\n')
    return p           

def write_main(model, output_file, p, float):    
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['var']
        if 'main' in key.lower():
            with open(output_file,'a') as file:
                file.write('\t')                
                
                file.write(str(var[0]) + " = lambda " + 
                            str(','.join(p)) + ":" +
                            " tfd.Independent(tfd."+ var[1] + "(")
                
                if 'likelihood' in model.keys():
                    formula = get_likelihood(model, key)
                    if formula is not None:
                        var[2][formula[1]] = formula[0]
                        #file.write(str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                        file.write(str(', '.join([f'{item}' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                    else:
                        #file.write(str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                        file.write(str(', '.join([f'{item}' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                else: # No fromula in likelihood
                    y, x = re.split(r'[~]',model['main']['input'])
                    x = x.replace(' ', '')
                    dist, args = x.split('(')
                    dist = dist.replace(" ", "")
                    args = args.replace("(", "")
                    args = args.replace(")", "")
                    args = args.split(",")
                    file.write(str(', '.join([f'{item}' for item in args]))+ "), reinterpreted_batch_ndims=1),")
                
                file.write('\n')
                
    with open(output_file,'a') as file:
        file.write('))')

def write_model(model, sep = ';',  path = 'mymodel.py', withDF = False, DFpath = None, data  = None, float = 16):
    if withDF == False:
        write_header(path, float)
    else:
        write_header_with_dataFrame(output_file = path, DFpath = DFpath, data = data, float =  float, sep = sep)
        
    p = write_priors(model, path, float)
    write_main(model, path, p, float)

## Mains -----------------------------------------------------
def build_model(model,
                path = None,
                df = None,
                float = 16,
                sep = ';'): 
    """
    Args:
        model (str): The name of the model to be built.
        path (str, optional): The path to the input data file. 
        If None, an empty DataFrame with column 'A' is created. Defaults to None.
        df (pandas.DataFrame, optional): A pandas DataFrame that contains the input data. 
        If None, the function reads the input data from the file specified by path. Defaults to None.

    Returns:
        Tensorflow probability model.

    Summary:
    This function takes three arguments: model, path, and df. It first checks if df is None. 
    If it is, it checks if path is None. If path is None, it creates an empty DataFrame with column 'A'. 
    If path is not None, it reads the input data from the file specified by path using the pd.read_csv() 
    function with any additional keyword arguments specified in kwargs. The function then gets the full model 
    specification using the get_var() function and checks for any undeclared parameters using the get_undeclared_params() 
    function. If there are no undeclared parameters, it writes the model specification to a file using the write_model() 
    function with the withDF parameter set to True, the DFpath parameter set to path, and the data parameter set to the 
    list of declared parameters in the input data. 
    If there are undeclared parameters, it prints a message indicating which parameters are missing and returns None.
    Finally, the function reloads the mymodel module and returns the built model.
    """
    if df is None :
        if  path is None:
            df = pd.DataFrame({'A' : []})
        else:        
            df = pd.read_csv(path, sep = sep)
    else:
        df.to_csv('mydf', index=False)
        path = 'mydf'
          
    full_model = get_var(model)
    print(full_model)
    issues = get_undeclared_params(full_model, df = df)

    if df.empty :
        if len(issues) == 0:
            print('Non missing variables')
            write_model(full_model, float = float, sep = sep)
        else:
            print("Arguments are missing: " + ''.join(issues))
            return None
    else:
        data = get_undeclared_params(full_model, df = df)
        if len(data['undeclared_params']) == 0:
            data = data['params_in_data']
            write_model(full_model, withDF = True, DFpath = path, data  = data, float = float, sep = sep)
        else:
           print("Arguments are missing: " + ''.join(data['undeclared_params'])) 
           return None        
    import importlib
    import mymodel
    importlib.reload(mymodel)
    from mymodel import m
    return m

#def write_fit(args, output_file = 'mymodel.py'):
#    file1 = open(output_file, 'w')
#    file1.write("\n")
#    file1.write

## test No data frame -----------------------------------------------------
#model = dict(main = 'y~Normal(m,s)',
#            likelihood = 'm = alpha + beta',
#            prior1 = 's~Exponential(1)',
#            prior2 = 'alpha ~ Normal(0,1)',
#            prior3 = 'beta ~ Normal(0,1)',
#            
#            main1 = 'z~Normal(m2,s2)',
#            likelihood2 = 'm2 = alpha2 + beta2',
#            prior4 = 's2~Exponential(1)',
#            prior5 = 'alpha2 ~ Normal(0,1)',
#            prior6 = 'beta2 ~ Normal(0,1)')    
#
#model = build_model(model, float = 16)        
#model

### test with data frame in path-----------------------------------------------------
#model = dict(main = 'weight~Normal(m,s)',
#            likelihood = 'm = alpha + beta * height',
#            prior1 = 's~Exponential(1)',
#            prior2 = 'alpha ~ Normal(0,1)',
#            prior3 = 'beta ~ Normal(0,1)')    
#
#model2 = build_model(model, 
#            path = "../data/Howell1.csv", sep = ';')
#model2

### test with data frame in function-----------------------------------------------------
#d = pd.read_csv('C:/Users/sebastian_sosa/OneDrive/Travail/Max Planck/Projects/python/rethinking-master/data/Howell1.csv', sep=';')
#d = d[d.age > 18]
#d.weight = d.weight - d.weight.mean()
#
#model = dict(main = 'height ~ Normal(m,s)',
#            likelihood = 'm = alpha + beta * weight',
#            prior1 = 's~Exponential(1)',
#            prior2 = 'alpha ~ Normal(0,1)',
#            prior3 = 'beta ~ Normal(0,1)')    
#
#build_model(model, path = None, df = d)
# %%
