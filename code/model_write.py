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
    mainOutput = []
    for key in model.keys():
        if 'main' in key:
            if df.empty: 
                mainOutput.append(model[key]['var'][0])
                tmp = model[key]['var'][2:]
            else:
                mainOutput.append(model[key]['var'][0])
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
    if any(ele in params for ele in mainOutput):
        print('model with main in other likelihood')
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
        file.write("# Import dependencies ----------------------------")    
        file.write('\n')
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
        file.write('\n')
        file.write("# Model ----------------------------")    
        file.write('\n')
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

def write_header_with_dataFrame(output_file, DFpath, data, float, sep):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("# Import dependencies ----------------------------")    
        file.write('\n')
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
        file.write('\n')
        file.write("# Import data (with modification if maded) ----------------------------")    
        file.write('\n')
        file.write("d = pd.read_csv('" + DFpath +"', sep = '" + sep + "')")    
        file.write('\n')
   
    for a in range(len(data)):
        with open(output_file,'a') as file:
            #file.write(data[a] + "= tf.convert_to_tensor(d." + data[a] + ", dtype = tf.float" + str(float) + ")")  
            file.write(data[a] + "= d." + data[a] )  
            file.write('\n')
       
    with open(output_file,'a') as file: 
        file.write("# Model ----------------------------")    
        file.write('\n')          
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
    mainsOutput = [] 
    for key in model.keys():
        input = model[key]['input']
        var = model[key]['var']
        if 'main' in key.lower():   
            mainsOutput.append(var[0])
            
    lParam = [] 
    for key in model.keys():
        input = model[key]['input']
        var = model[key]['var']
        if 'likelihood' in key.lower():   
            lParam.append(var[1]) 
    lParam = [item.replace(' ', '') for sublist in lParam for item in sublist]

    for key in model.keys():
        input = model[key]['input']
        var = model[key]['var']
        if 'main' in key.lower():           
            with open(output_file,'a') as file:  
                file.write('\n')                                           
                if 'likelihood' in model.keys():  
                    formula = get_likelihood(model, key)      
                    if formula is not None:
                        file.write('\t') 
                        params = re.split(r'[+***-]',formula[0])
                        params_in_prior =[x for x in params if x in p]
                        params_in_main =[x for x in params if x in mainsOutput]
                        params = params_in_prior + params_in_main + [var[2][1]]                     
                        var[2][formula[1]] = formula[0]
                        file.write(str(var[0]) + " = lambda " + 
                                    str(','.join(params)) + ":" +
                                    " tfd.Independent(tfd."+ var[1] + "(") 
                        #file.write(str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                        file.write(str(', '.join([f'{item}' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                    else:
                        file.write(str(var[0]) + " = lambda " + 
                            str(','.join(p)) + ":" +
                            " tfd.Independent(tfd."+ var[1] + "(")
                        #file.write(str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                        file.write(str(', '.join([f'{item}' for item in var[2]]))+ "), reinterpreted_batch_ndims=1),")
                
                else: # No fromula in likelihood
                    file.write(str(var[0]) + " = lambda " + 
                            str(','.join(p)) + ":" +
                            " tfd.Independent(tfd."+ var[1] + "(")
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

def write_HMC(model, 
              observed_data,
              parallel_iterations = 1,
              num_results = 2000, 
              num_burnin_steps=500,
              step_size = 0.065,
              num_leapfrog_steps = 5,
              num_adaptation_steps = 400,
              num_chains = 4,
              float = 32,
              inDF = True
             ):
    saved_args = locals()
    output_file = 'mymodel.py'
    with open(output_file,'a') as file:
        file.write('\n')
        file.write('\n')
        file.write('# Run HMC ----------------------------')
        file.write('\n')
        file.write("posterior, trace, sample_stats =  run_model(model = m,")
        file.write('\n')   
             
        for key in saved_args.keys():
            if key != "observed_data" and key != "inDF" and key != 'model' and key != 'float':
                file.write(key + '=' + str(saved_args[key]) + ',')
                file.write('\n')
        
        file.write("observed_data = dict(")        
        if saved_args['inDF']:
            for key in saved_args['observed_data'].keys():      
                file.write(key + ' = d.' + saved_args['observed_data'][key] + ".astype('float" + str(float) + "').values,")
        else:
             for key in saved_args['observed_data'].keys():
                file.write(key + ' = ' + observed_data[key])    
        file.write("))")
        
## Mains -----------------------------------------------------
def build_model(model,
                path = None,
                df = None,
                float = 16,
                sep = ';'): 
    if df is None :
        if  path is None:
            df = pd.DataFrame({'A' : []})
        else:        
            df = pd.read_csv(path, sep = sep)
    else:
        df.to_csv('mydf', index=False)
        path = 'mydf'
          
    full_model = get_var(model)
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