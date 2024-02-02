from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
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
        type["with_likelihood"] = False
        type["with_indices"]  = False

        formula = self.f  
        indices = []
        var = []   
        
        if "likelihood" in self.f.keys():
           type["with_likelihood"] = True
        else:
            type["with_likelihood"] = False

        for key in formula.keys():
            if "likelihood" in key :
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
        self.likelihood = {}       
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
            self.likelihood[main_name] = result
            return result
        else:
            self.likelihood = None
            return None
    
    def formula(self, f):
        self.f = f
        self.get_model_type()
        self.get_var()
        self.get_undeclared_params()
        self.get_priors_names()
        self.get_mains_info()
        self.likelihood = {}
        for key in self.mains_infos.keys():
            if 'main' in key:
                 self.get_likelihood(main_name = key)

       
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
                    if  len(self.undeclared_params) > 1:
                        main_likelihood_params_in_df = [x for x in main_likelihood_params if x in self.undeclared_params['params_in_data']]     
                        main_likelihood_params = [x for x in main_likelihood_params if x not in self.undeclared_params['params_in_data']]
                    else:
                        main_likelihood_params_in_df = None

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
                    main_likelihood_params_in_df = None


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
                                      'likelihood_params_in_df' : main_likelihood_params_in_df,
                                      'indices_prior' : id,
                                      'indices_position': position,
                                      'indices_var': id_var,
                                      }

        self.mains_infos = mains_infos
        return mains_infos

class write():
    
    def create_distribution(self, distribution_name, *args, **kwargs):
        distribution_class = getattr(tfd, distribution_name, None)

        if distribution_class is None or not callable(distribution_class):
            raise ValueError(f"Invalid distribution name: {distribution_name}")

        distribution_instance = distribution_class(*args, **kwargs)
        return distribution_instance
    
    def convert_to_numeric(self, lst):
        numeric_list = []
        for item in lst:
            try:
                myFloat = float(item)
                myFloat = tf.cast(myFloat, self.float)
                numeric_list.append(myFloat)
            except ValueError:
                try:
                    myInt = int(item)
                    myInt = tf.cast(myInt, self.int)
                    numeric_list.append(myInt)

                except ValueError:
                    numeric_list.append(item)  # Keep as is if not numeric
        return numeric_list

    def write_main_text(self):
        result = {}
        for key in self.mains_infos.keys():
            text = ""
            #text = text + self.mains_infos[key]['ouput'] + " = lambda "
            text = text +  "lambda "

            if len(self.mains_infos[key]['priors']) > 0:
                text = text + ",".join(self.mains_infos[key]['priors']) + ", "

            if self.mains_infos[key]["likelihood_params"] is not None:
                text = text + ", ".join(self.mains_infos[key]["likelihood_params"])

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
                    char_param_index = "tf.transpose(tf.gather(tf.transpose(" + item + "), tf.cast("
                    new_formula = new_formula.replace(item, char_param_index)

                    if indices_var[a] in self.mains_infos[key]['likelihood_params_in_df']:
                        char_var_index = 'df.' + indices_var[a] + ", dtype= tf.int32)))"
                    else:
                        char_var_index = indices_var[a] + ", dtype= tf.int32)))"
                    new_formula = new_formula.replace("[" + indices_var[a] + "]", char_var_index)
                    text = text + new_formula

                text = text + ',' + ','.join(self.mains_infos[key]["params"][1:]) + ")"
            else:
                if self.mains_infos[key]['with_likelihood'] is not None:
                    if self.mains_infos[key]['likelihood_params_in_df'] is not None:
                         for a in range(len(self.mains_infos[key]['likelihood_params_in_df'])):
                              self.mains_infos[key]['likelihood_formula'] = self.mains_infos[key]['likelihood_formula'].replace(self.mains_infos[key]['likelihood_params_in_df'][a], 'df.' + self.mains_infos[key]['likelihood_params_in_df'][a])
                         text = text + self.mains_infos[key]['likelihood_formula'] + ','
                         text = text + ','.join(self.mains_infos[key]["params"][1:]) + ")"
                    else:
                        text = text + self.mains_infos[key]['likelihood_formula'] + ','
                        text = text + ','.join(self.mains_infos[key]["params"][1:]) + ")"
                else:
                     text = text + ','.join(self.mains_infos[key]['params']) + ")"

            text = text + ", reinterpreted_batch_ndims=1)"
            result[self.mains_infos[key]['ouput']] = text
        
        self.main_text = result
        return result

    def create_function_from_string(self, func_str, name):
        # Define required imports and namespace for exec
        imports = {'tfd': tfd, 'tf': tf, 'df': self.df}
        namespace = {}
        # Execute the string as Python code within the specified namespace
        exec( name + ' = ' + func_str, imports, namespace)

        # Extract the function from the namespace
        return namespace[name]
    
    def tensor_prior(self):
        self.tensor = {}
        # prior -------------------------------
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
                        shape = self.df[self.indices_var[idI]].nunique()
                    else:
                        shape = 1
                else:
                    shape = 1

                self.tensor[model[key]['var'][0]] = tfd.Sample(
                    self.create_distribution(model[key]['var'][1],
                    *self.convert_to_numeric(model[key]['var'][2])), sample_shape = shape)
            
        self.priors = p
   
    def tensor_main(self):
        for key in self.main_text.keys():
            self.tensor[key] = eval(f"{self.main_text[key]}")