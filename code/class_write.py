from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
import numpy as np

class define():
    def __init__(self, formula = None):
        self.f = formula
    # Model definition----------------------------
    def find_index_position(self, text):
        pattern = r'\b(?:[^+\-*\/\(\)~]+|\([^)]+\]|[^[\]]+\])+'
        result = re.findall(pattern, text)
        position = []
        for a in range(len(result)):
            if "[" in result[a]:
                position.append(a-1)
        return position
    
    def get_model_type(self):
        type = {} 
        formula = self.f  
        indices = []
        var = []   
        if "likelihood" in self.f.keys():
           type["with_likelihood"] = True
        else:
            type["with_likelihood"] = False

        for key in formula.keys():
            if "likelihood" in key:
                if "[" in formula[key]:                    
                    type["with_indices"] = True
                    params = self.get_formula( formula = formula[key], type = 'likelihood') 
                    position = self.find_index_position(formula[key])  
                    id = [params[1][i] for i in position]
                    id_var = [params[1][i+1] for i in position]
                    type[key] = {"params": id} 
                    indices = indices + id
                    var = var + id_var
                else:
                    type["with_indices"] = False
            else:
                type["with_indices"] = False
        self.model_type = type
        # An index variable have a parameter and a variable associated (e.g. alpha[species])
        self.indices = indices
        self.indices_var = var
        return type

    def get_var(self):
        formula = self.f
        full_model = {}
        for  key in formula.keys():
             full_model[key] = dict(
                 input = formula[key],
                 var = self.get_formula(formula=formula[key], type=key)
                 )
        self.full_model = full_model
        return full_model     
    
    def get_priors_names(self):
        model = self.full_model
        priors = [] 
        for key in model.keys():
            input = model[key]['input']
            var = model[key]['var']      
            if 'prior' in key.lower():
                priors.append(var[0])
        self.priors = priors
        return priors
           
    def get_formula(self, formula = "y~Normal(0,1)", type = 'likelihood'):        
        y, x = re.split(r'[~]',formula)
        y = y.replace(" ", "")
        x = x.replace(" ", "")
        if 'likelihood' in type: 
            if x.find('(') == -1: # If parenthesis then we concider the presence of a distribution
                args = re.split(r'[+*()*[*,]',x)            
                new = []
                for i in range(len(args)):
                    if args[i] != '':
                        args[i] = args[i].replace("]", "")
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

    def get_undeclared_params(self):
        model = self.full_model
        # Name of variables
        Vars = []
        # Name of variables + values
        params = []
        # main ouput name
        mainOutput = []
        for key in model.keys():
            if 'main' in key:
                if self.df.empty: 
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

        params = [element for sublist in params for element in sublist]

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

        if self.df.empty:  
            self.undeclared_params =  {'undeclared_params': undeclared_params2}
            return {'undeclared_params': undeclared_params2}
        else:
            test = pd.Index(undeclared_params2).difference(self.df.columns).tolist()
            test2 =  list(set(undeclared_params2) & set(self.df.columns))
            test =  list(set(test).difference(mainOutput))
            self.undeclared_params = {'undeclared_params': test, 'params_in_data' : test2}  
            return {'undeclared_params': test, 'params_in_data' : test2}  

    def get_likelihood(self, main_name = "main"):
        model = self.full_model
        result = []
        for key in model.keys():
            if 'likelihood' in key.lower():
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
            self.likelihood = result
            return result
        else:
            print(None)
            self.likelihood = None
            return None
    
    def formula(self, f):
        self.f = f
        self.get_model_type()
        self.get_var()
        self.get_undeclared_params()
        self.get_priors_names()
        self.get_mains_info()
        self.get_likelihood(main_name = "main") #!make it adptative
        return self.undeclared_params
    
    def get_mains_info(self):
        tmp = self.full_model
        mains_infos = {}
        p = self.priors
        for key in tmp.keys():
            if 'main' in key.lower():
                main_name = key
                main_input = tmp[key]['input']
                infos = tmp[key]['var']
                main_output = infos[0]
                main_distribution = infos[1]
                main_params = infos[2]

                # Params
                ## Priors (get params in self.priors)
                main_priors = []
                for a in range(len(main_params)):
                    if main_params[a] in p:
                        main_priors.append(self.priors[self.priors.index(str(main_params[a]))])

                # likelihood
                if self.model_type['with_likelihood']:
                    main_with_likelihood = True
                    main_likelihood_name = infos[2][0]
                    main_likelihood_formula = self.get_likelihood(main_name = key)[0]
                    main_likelihood_input = main_likelihood_name + "~" + main_likelihood_formula
                    main_likelihood_params = self.get_formula(formula=main_likelihood_input, 
                                                              type = 'likelihood')[1]
                    # Index in formula
                    if "[" in main_likelihood_formula:                    
                        with_indices = True
                        position = self.find_index_position(main_likelihood_input)  
                        id = [main_likelihood_params[i] for i in position]
                        id_var = [main_likelihood_params[i+1] for i in position]
    
                    else:
                        with_indices =  False
                        id = None
                        id_var = None
                        position = None
                else:
                    with_indices =  False
                    id = None
                    id_var = None
                    position = None
                    main_with_likelihood = None
                    main_likelihood_name = None
                    main_likelihood_formula = None
                    main_likelihood_input = None
                    main_likelihood_params = None

                mains_infos[main_name] = {'ouput': main_output, 
                                          'input': main_input,
                                          'distribution': main_distribution, 
                                          'params' : main_params,
                                          'priors' : main_priors,
                                          'with_likelihood': main_with_likelihood,
                                          'with_indices' : with_indices,
                                          'likelihood_names' : main_likelihood_name,
                                          'likelihood_input' : main_likelihood_input,
                                          'likelihood_formula': main_likelihood_formula,
                                          'likelihood_params': main_likelihood_params,                                          
                                          'indices_prior' : id,
                                          'indices_position': position,
                                          'indices_var': id_var,
                                          }

        self.mains_infos = mains_infos
        return mains_infos

class write():
    def write_header(self):
        output_file = self.model_path
        self.undeclared_params['undeclared_params']
        data = self.undeclared_params['params_in_data']
        if self.df.empty : 
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
                file.write("from code.model_fit import *")    
                file.write('\n')
                file.write("from code.model_diagnostic import *")    
                file.write('\n')
                file.write('\n')
                file.write("# GPU configuration ----------------------------")  
                file.write('\n')  
                file.write("physical_devices = tf.config.experimental.list_physical_devices('GPU')") 
                file.write('\n')   
                file.write("if len(physical_devices) > 0:")
                file.write('\n')
                file.write('\t')
                file.write("tf.config.experimental.set_memory_growth(physical_devices[0], True)")
                file.write('\n')
                file.write('\n')
                file.write("# Model ----------------------------")    
                file.write('\n')
                file.write("m = tfd.JointDistributionNamed(dict(")
                file.write('\n')
        else:
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
                file.write("from code.model_fit import *")    
                file.write('\n')
                file.write("from code.model_diagnostic import *")    
                file.write('\n')
                file.write('\n')        
                file.write("# GPU configuration ----------------------------")  
                file.write('\n')  
                file.write("physical_devices = tf.config.experimental.list_physical_devices('GPU')")
                file.write('\n')    
                file.write("if len(physical_devices) > 0:")
                file.write('\n')
                file.write('\t')
                file.write("tf.config.experimental.set_memory_growth(physical_devices[0], True)")
                file.write('\n')    
                file.write('\n')
                file.write("# Import data (with modification if maded) ----------------------------")    
                file.write('\n')
                file.write("d = pd.read_csv('" + self.df_path +"')")    
                file.write('\n')
                file.write('\n')
                file.write("# Set up parameters ----------------------------")    
                file.write('\n')
            for a in range(len(data)):
                with open(output_file,'a') as file:
                    file.write(data[a] + "= d." + data[a] )                      
                    file.write('\n')
                    if data[a] in self.indices_var:
                        file.write("len_" + data[a] + "= len(set(d." + data[a] + ".values))")                   
                        file.write('\n')

            with open(output_file,'a') as file: 
                file.write('\n')
                file.write("# Model ----------------------------")    
                file.write('\n')          
                file.write("m = tfd.JointDistributionNamed(dict(")
                file.write('\n')

    def write_priors(self): 
        model = self.full_model
        output_file = self.model_path
        p = [] 
        for key in model.keys():
            input = model[key]['input']
            var = model[key]['var']      
            if 'prior' in key.lower():
                p.append(var[0])
                # Get indices shape
                if self.model_type["with_indices"]:
                    if var[0] in self.indices:
                        idI = self.indices.index(var[0])
                        shape = "len_" + self.indices_var[idI]
                    else:
                        shape = 1
                else:
                    shape = 1
                with open(output_file,'a') as file:
                    file.write('\t')
                    file.write(str(var[0]) + 
                               " = tfd.Sample(tfd." + var[1] + "(" + 
                                #str(', '.join([f'tf.cast([{item}], dtype=tf.float{float})' for item in var[2]])) + "), sample_shape=1),")
                                str(', '.join([f'{item}' for item in var[2]])) + "), sample_shape= "+ str(shape)+ "),")

                    file.write('\n')
        self.priors = p
        return p           

    def main_text(self):
        result = []
        for key in self.mains_infos.keys():
            text = ""
            text = text + self.mains_infos[key]['ouput'] + " = lambda "
            text = text + "".join(self.mains_infos[key]['priors']) + ", "
            text = text + ", ".join(self.mains_infos[key]["likelihood_params"])

            # Lambda params
            if self.mains_infos[key]['indices_prior'] != None:
                text = text + ''.join(self.mains_infos[key]['indices_prior'])

            # likelihood distribution
            text = text + ": tfd.Independent(tfd." + self.mains_infos[key]['distribution'] + "(" 

            # likelihood formula
            if self.mains_infos[key]['with_indices']:
                likelihood_formula = self.mains_infos[key]['likelihood_formula']
                likelihood_params = self.mains_infos[key]['likelihood_params']
                indices_prior = self.mains_infos[key]['indices_prior']
                indices_var = self.mains_infos[key]['indices_var']
                new_formula = likelihood_formula
                for a in range(len(indices_prior)):
                    item = indices_prior[a]
                    print(item)            
                    char_param_index = "tf.transpose(tf.gather(tf.transpose(" + item + "), tf.cast("
                    new_formula = new_formula.replace(item, char_param_index)

                    char_var_index =  indices_var[a] + ", dtype= tf.int32)))"
                    new_formula = new_formula.replace("[" + indices_var[a] + "]", char_var_index)
                    print(new_formula)
                    text = text + new_formula

                text = text + ',' + ','.join(self.mains_infos[key]["params"][1:]) + ")"
            else:
                text = text + self.mains_infos[key]['likelihood_formula'] + ','
                text = text + ','.join(self.mains_infos[key]["params"][1:]) + ")"
            text = text + ", reinterpreted_batch_ndims=1)"
            result.append(text)
        
        
        self.main_text = result
        return result
    
    def write_main2(self):
        output_file = self.model_path
        with open(output_file,'a') as file: 
            for a in self.main_text:
                file.write(a) 
                file.write(",")
                file.write("\n")
            file.write("))")
    def write_main(self):   
        model = self.full_model
        output_file = self.model_path
        p = self.priors

        # Get ouput of main(s)
        mainsOutput = [] 
        for key in model.keys():
            input = model[key]['input']
            var = model[key]['var']
            if 'main' in key.lower():   
                mainsOutput.append(var[0])
        #print(mainsOutput)

        # Find all main(s)
        for key in model.keys():
            input = model[key]['input']
            var = model[key]['var']
            #print(var)
            if 'main' in key.lower():           
                with open(output_file,'a') as file:  
                    # Find next likelihood                                       
                    if 'likelihood' in model.keys():                      
                        formula = self.get_likelihood()   
                        #print(formula)
                        if formula is not None:
                            file.write('\t') 
                            # Find parameters corresponding to the main 
                            ## param = values, var[1] = distribution, var[2] = names
                            ## formula[0] = likelihood parameters name
                            #print(formula[0])
                            params = re.split(r'[+***-]',formula[0])
                            
                            params_in_prior =[x for x in params if x in p]
                            params_in_main =[x for x in params if x in mainsOutput]
                            #print(params_in_prior)  
                            #print(params_in_main)  
                            if len(var[2]) > 1:
                                params = params_in_prior + params_in_main + [var[2][1]]   
                            else:
                                params = params_in_prior + params_in_main 

                            # get potential prior 
                            #print(params)            
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
