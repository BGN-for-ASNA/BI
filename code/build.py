from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

class define():
    def __init__(self, formula = None):
        self.f = formula        
    # Basic functions----------------------------    
    def separate_args_kwargs(self,input_list):
        args = []
        kwargs = {}

        for item in input_list:
            if '=' in item:
                key, value = item.split('=')
                kwargs[key] = value
            else:
                args.append(item)

        return {'args': args, 'kwargs': kwargs}
    
    def extract_indices_patterns(self, input_string):
        pattern = r'\b(\w+)\s*\[(.*?)\]'
        matches = re.findall(pattern, input_string)
        patterns_dict = {match[0]: match[1] for match in matches}
        return patterns_dict
    
    def find_index_position(self, text):
        pattern = r'\b(?:[^+\-*\/\(\)~]+|\([^)]+\]|[^[\]]+\])+'
        result = re.findall(pattern, text)
        position = []
        for a in range(len(result)):
            if "[" in result[a]:
                position.append(a-1)
        return position
    
    def which_key_have_value(self, dict, value) :
        matching_keys = ''
        for k, v in dict.items():
            if v == value:
                matching_keys = k
        return matching_keys
    
    def add_entry_if_not_none(self, variable, new_entry):
        if variable is not None:
            if not isinstance(variable, list):
                variable = [variable]  # Convert variable to a list if it's not already a list
            variable.append(new_entry)  # Add new entry to the list
        else:
            variable = new_entry
        return variable
    
    def join_elements(self, input):
        if isinstance(input, list):
            return ' '.join(map(str, input))
        else:
            return str(input)
        
    # Get model informations----------------------------    
    def get_model_type(self):
        type = {} 
        type["with_likelihood"] = False
        type["with_indices"]  = False

        formula = self.f  
        self.indices = {}
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
                    indices_dict = self.extract_indices_patterns(formula[key])
                    id = [params[1][i] for i in position]
                    id_var = [params[1][i+1] for i in position]
                    type[key] = {"params": id} 
                    var = var + id_var
                    self.indices.update(indices_dict)
                else:
                    type["with_indices"] = False

        self.model_type = type
        # An index variable have a parameter and a variable associated (e.g. alpha[species])
        self.indices_var = var
        return type

    def get_var(self):
        formula = self.f
        full_model = {}
        names = {}
        for  key in formula.keys():
             full_model[key] = dict(
                 input = formula[key],
                 var = self.get_formula(formula=formula[key], type=key)
                 )
             names[key] = full_model[key]['var'][0]

        self.full_model = full_model
        self.model_names = names
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
                    result.append(index) # return a list with likelihood formula and its position in model[main_name]['var'][2]

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
    
    # Get main informations----------------------------    
    def get_main_info_likelihood_indices(self, dict):
        dict['with_indices'] = True
        dict['indices_position'] = self.find_index_position(dict['likelihood_formula'])  
        dict['indices_patterns'] = self.extract_indices_patterns(dict['likelihood_formula'])  
        dict['indices_prior'] = [dict['likelihood_params'][i] for i in dict['indices_position']]
        dict['indices_var'] = [dict['likelihood_params'][i+1] for i in dict['indices_position']]
        return dict
    
    def get_main_info_likelihood_args(self, dict):
        dict['new_params'] = dict['params'] 
        # Find likelihood associated to this main
        for a in range(len(dict['params']['args'])):
            if dict['params']['args'][a] in self.model_names.values():
                #check if self.model_names.values() name have likelihhod in it
                key_name = self.which_key_have_value(self.model_names, dict['params']['args'][a])

                if 'prior' in key_name:
                    dict['priors'] = self.add_entry_if_not_none(dict['priors'], self.model_names[key_name])
                    
                if 'likelihood' in key_name:
                    dict['likelihood_ouput'] = dict['params']['args'][a]     
                    
                   # Handle multiple likelihood (to do)
                    if dict['likelihood_name'] is not None: 
                        dict['multiple_likelihoods'] = True
                        self.model_type['multiple_likelihoods'] = True # To work on
                        dict['likelihood_name'] = self.add_entry_if_not_none(dict['likelihood_name'],key_name)
                    else:
                        dict['likelihood_name'] = self.add_entry_if_not_none(dict['likelihood_name'],key_name)

                    dict['multiple_likelihoods'] = False
                    self.model_type['multiple_likelihoods'] = False            
                    dict['likelihood_formula'] = self.full_model[dict['likelihood_name']]['input']
                    dict['likelihood_output'] = self.full_model[dict['likelihood_name']]['var'][0]
                    dict['likelihood_params'] = self.full_model[dict['likelihood_name']]['var'][1] 
                    x = re.split(r'[~]',dict['likelihood_formula'])[1]
                    x = x.replace(" ", "")
                    dict['params']['args'][a] = x
    
                    # Index in formula
                    if "[" in dict['likelihood_formula']:                    
                        dict = self.get_main_info_likelihood_indices(dict)
    
                    if  len(self.undeclared_params) > 1:
                        dict['likelihood_params_in_df'] = [x for x in dict['likelihood_params'] if x in self.undeclared_params['params_in_data']]                             
                        dict['likelihood_params_not_in_df'] = [x for x in dict['likelihood_params'] if x not in self.undeclared_params['params_in_data']]
                        dict['likelihood_params'] = [item for item in dict['likelihood_params']  if item not in dict['likelihood_params_in_df']]

                    else:
                        dict['main_likelihood_params_in_df'] = False
        return dict
    
    def get_main_info_likelihood_kargs(self, dict):
        dict['new_params'] = dict['params'] 
        for key in dict['params']['kwargs']:
            if dict['params']['kwargs'][key] in self.model_names.values():               
                #check if self.model_names.values() name have likelihhod in it
                key_name = self.which_key_have_value(self.model_names, dict['params']['kwargs'][key])

                if 'prior' in key_name:
                    dict['priors'] = self.add_entry_if_not_none(dict['priors'], self.model_names[key_name])

                if 'likelihood' in key_name:
                    dict['likelihood_ouput'] = dict['params']['kwargs'][key]

                # Handle multiple likelihood (to do)
                if dict['likelihood_name'] is not None: 
                    dict['multiple_likelihoods'] = True
                    self.model_type['multiple_likelihoods'] = True # To work on
                    dict['multiple_likelihoods'].append = [k for k, v in self.model_names.items() if v == dict['params']['kwargs'][key]]   
                else:
                    dict['likelihood_name'] = [k for k, v in self.model_names.items() if v == dict['params']['kwargs'][key]] 

                dict['likelihood_name'] = dict['likelihood_name'][0]
                dict['multiple_likelihoods'] = False
                self.model_type['multiple_likelihoods'] = False            
                dict['likelihood_formula'] = self.full_model[dict['likelihood_name']]['input']
                dict['likelihood_output'] = self.full_model[dict['likelihood_name']]['var'][0]
                dict['likelihood_params'] = self.full_model[dict['likelihood_name']]['var'][1] 
                x = re.split(r'[~]',dict['likelihood_formula'])[1]
                x = x.replace(" ", "")
                dict['new_params']['kwargs'][key] = x

                # Index in formula
                if "[" in dict['likelihood_formula']:                    
                    self.get_main_info_likelihood_indices(dict)

                if  len(self.undeclared_params) > 1:
                    dict['main_likelihood_params_in_df'] = [x for x in dict['likelihood_params'] if x in self.undeclared_params['params_in_data']]     
                    dict['main_likelihood_params'] = [x for x in dict['likelihood_params'] if x not in self.undeclared_params['params_in_data']]
                    dict['likelihood_params'] = [item for item in dict['likelihood_params']  if item not in dict['likelihood_params_in_df']]
                else:
                    dict['main_likelihood_params_in_df'] = False
        return dict

    def get_main_info_with_likelihood(self, dict):            
            if len(dict['params']['args']) > 0:
                dict = self.get_main_info_likelihood_args(dict)
            if len(dict['params']['kwargs']) > 0:
                dict = self.get_main_info_likelihood_kargs(dict)        
            return  dict
    
    def get_main_info_likelihood_arguments_in_df(self, dict):
        if len(dict['params']['args']) > 0:
            for a in range(len(dict['likelihood_params_in_df'])):
                for b in range(len(dict['params']['args'])):
                    if dict['likelihood_params_in_df'][a] in dict['params']['args'][b]:
                        dict['params']['args'][b] = dict['params']['args'][b].replace(dict['likelihood_params_in_df'][a], 
                                                                                      'df.' + dict['likelihood_params_in_df'][a])
        if len(dict['params']['kwargs']) > 0:
            for a in range(len(dict['likelihood_params_in_df'])):
                for key in dict['params']['kwargs'].keys():
                    if dict['likelihood_params_in_df'][a] in dict['params']['kwargs'][key]:
                        dict['params']['kwargs'][key] = dict['params']['kwargs'][key].replace(dict['likelihood_params_in_df'][a], 
                                                                                              'df.' + dict['likelihood_params_in_df'][a])
        return dict   
    
    # Build dictionary to store main informations
    def get_main_dict(self):
        dict = {'ouput': None, 
                'input': None,
                'distribution': None, 
                'params' : None,
                'priors' : None,
                'with_likelihood': None,
                'multiple_likelihoods': None,
                'with_indices' : False,
                'likelihood_name' : None,
                'likelihood_output' : None,
                'likelihood_formula': None,
                'likelihood_params': None,                                          
                'likelihood_params_in_df' : None,
                'indices_prior' : None,
                'indices_position': None,
                'indices_patterns': None,
                'indices_var': None,
                } 
                
        return dict
    
    def get_mains_info(self):
        tmp = self.full_model
        mains_infos = {}
        p = self.priors
        # For each main retrieve informations
        for key in tmp.keys():
            if 'main' in key.lower():                
                main_name = key
                infos = tmp[key]['var']
                dict = self.get_main_dict()
                dict['ouput'] = infos[0]
                dict['distribution'] = infos[1]
                dict['params'] = infos[2]
                dict['params'] = self.separate_args_kwargs(dict['params'])
                dict['input'] = tmp[key]['input']

                # likelihood
                if self.model_type['with_likelihood']:
                    dict['with_likelihood'] = True
                    dict = self.get_main_info_with_likelihood(dict)
                    
                    # Add params in df to LK
                    if dict['likelihood_params_in_df'] is not None:
                        dict['likelihood_params_in_df'] = list(set(dict['likelihood_params_in_df']))
                        dict = self.get_main_info_likelihood_arguments_in_df(dict)

                    dict['new_params']['args'] = self.convert_to_numeric(dict['new_params']['args'])
                    dict['new_params']['kwargs']= self.convert_to_numeric_dict(dict['new_params']['kwargs'])
            mains_infos[main_name] = dict


        self.mains_infos = mains_infos
        return mains_infos

class write():
    
    def create_distribution(self, distribution_name, *args, **kwargs):
        distribution_class = getattr(tfd, distribution_name, None)

        if distribution_class is None or not callable(distribution_class):
            raise ValueError(f"Invalid distribution name: {distribution_name}")

        distribution_instance = distribution_class(*args, **kwargs)
        return distribution_instance
    
    def convert_to_numeric_prior(self, lst):
        numeric_list = []
        for item in lst:
            try:
                myFloat = float(item)
                myFloat = tf.cast(myFloat, self.float_prior)
                numeric_list.append(myFloat)
            except ValueError:
                try:
                    myInt = int(item)
                    myInt = tf.cast(myInt, self.int)
                    numeric_list.append(myInt)

                except ValueError:
                    numeric_list.append(item)  # Keep as is if not numeric
        return numeric_list

    def convert_to_numeric(self, lst):
        numeric_list = []
        for item in lst:
            try:
                myFloat = "tf.cast(" + str(float(item)) + ", dtype = tf.float" + str(self.float) + ')'
                #myFloat = tf.cast(myFloat, self.float)
                numeric_list.append(myFloat)
            except ValueError:
                try:
                    myInt = "tf.cast(" + str(int(item))  + ", dtype = tf.int" + str(self.float) + ')'
                    myInt = tf.cast(myInt, self.int)
                    numeric_list.append(myInt)

                except ValueError:
                    numeric_list.append(item) 
                     # Keep as is if not numeric
        return numeric_list
    
    def convert_to_numeric_dict(self, lst):
        for key in lst.keys():
            try:
                #myFloat = tf.cast(myFloat, self.float)
                lst[key] = "tf.cast(" + str(float(lst[key] )) + ", dtype = tf.float" + str(self.float) + ')'
            except ValueError:
                try:
                    lst[key] =  "tf.cast(" + str(int(lst[key] ))  + ", dtype = tf.float" + str(self.float) + ')'
                except ValueError:
                    lst[key] = lst[key]
                     # Keep as is if not numeric
        return lst

    # Wirte main functions -------------------------------------
    def convert_indices(self, input_string, dtype='32'):
        pattern = r'(\w+)\[(.*?)\]'
        output_string = re.sub(pattern, rf"tf.transpose(tf.gather(tf.transpose(\1), tf.cast(\2, dtype=tf.int{dtype})))", input_string)
        return output_string

    #def write_main_text_indices(self, mains_infos, text):
    #    likelihood_formula = mains_infos['likelihood_formula']
    #    likelihood_params = mains_infos['likelihood_params']
    #    indices_prior = mains_infos['indices_prior']
    #    indices_var = mains_infos['indices_var']
    #    new_formula = likelihood_formula
    #    new_formula = new_formula.split('~')[1]
    #    new_formula = new_formula.replace(" ", "")

    #    # Manage indices-------
    #    for a in range(len(indices_prior)):
    #        item = indices_prior[a]
    #        char_param_index = "tf.transpose(tf.gather(tf.transpose(" + item + "), tf.cast("
    #        new_formula = new_formula.replace(item, char_param_index)
 #
    #        if mains_infos['likelihood_params_in_df'] is not None:
    #            if indices_var[a] in mains_infos['likelihood_params_in_df']:
    #                char_var_index = 'df.' + indices_var[a] + ", dtype= tf.int32)))"
    #            else:
    #                char_var_index = indices_var[a] + ", dtype= tf.int32))"
    #            new_formula = new_formula.replace("[" + indices_var[a] + "]", char_var_index)
#
    #        text = text + new_formula 

    #    # Manage non indices-------
    #    if mains_infos['likelihood_params_in_df'] is not None:
    #        for a in range(len(mains_infos['likelihood_params_in_df'])):
    #            if mains_infos['likelihood_params_in_df'][a] not in indices_var:
    #                if mains_infos['likelihood_params_in_df'][a] in mains_infos['likelihood_params_in_df']:
    #                    text = text.replace(mains_infos['likelihood_params_in_df'][a], 'df.' + mains_infos['likelihood_params_in_df'][a] )
    #    
    #    if len(mains_infos['params']['args']) > 0:
    #        no_indices_param = [mains_infos['params']['args'][i] for i in range(len(mains_infos['params']['args'])) if i not in mains_infos['indices_position']]
    #        text = text + ', ' +','.join(no_indices_param) + ")"
    #    return text
    
    def write_main_text_no_indices(self, mains_infos, text):
        if mains_infos['with_likelihood']:
            if len(mains_infos['params']['args']) > 0:
               text = text + ','.join(mains_infos['params']['args']) 
            if len(mains_infos['params']['kwargs']) > 0:
                text = text + ','.join(mains_infos['params']['kwargs'])                 
            #else:
            #    text = text + ','.join(mains_infos["params"]['args']) + ","
            #    text = text + ', '.join([f"{key} = {value}" for key, value in mains_infos["params"]['kwargs'].items()]) + ")"
            text = text + ')'
        else:
             text = text + ','.join(mains_infos['params']) + ")"
        return text
    
    def write_main_text(self):
        result = {}
        for key in self.mains_infos.keys():
            text = ""
            text = text +  "lambda "

            # Params
            ## likelihood priors 
            if self.mains_infos[key]['priors'] is not None:
                text = text + self.join_elements(self.mains_infos[key]["priors"]) + ', '

            # likelihood params
            if self.mains_infos[key]["likelihood_params"] is not None:
                text = text + ", ".join(self.mains_infos[key]["likelihood_params"])

            # likelihood distribution
            text = text + ": tfd.Independent(tfd." + self.mains_infos[key]['distribution'] + "(" 

            # likelihood formula
            if self.mains_infos[key]['with_indices']:
                #text = self.write_main_text_indices(self.mains_infos[key], text)
                if len(self.mains_infos[key]['params']['args'])>0:
                    for item in self.mains_infos[key]['params']['args']:
                        text = text + self.convert_indices(item, self.float)+ ','

                if len(self.mains_infos[key]['params']['kwargs'])>0:
                    for key in self.mains_infos[key]['params']['kwargs'].keys():
                        text = text + self.convert_indices(self.mains_infos[key]['params']['kwargs'][key], self.float) + ','
                
                text = text + "), reinterpreted_batch_ndims=1)"

            else:
                text = self.write_main_text_no_indices(self.mains_infos[key], text)
                text = text + ", reinterpreted_batch_ndims=1)"

            result[self.mains_infos[key]['ouput']] = text
        
        self.main_text = result
        return result

    # Write tensorflow probability model -------------------------
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
        #output_file = self.model_path
        p = [] 
        for key in model.keys():
            input = model[key]['input']
            var = model[key]['var']      
            if 'prior' in key.lower():
                p.append(var[0])
                # Get indices shape
                if self.model_type["with_indices"]:
                    if var[0] in self.indices.keys():
                        shape = self.df[self.indices[var[0]]].nunique()
                    else:
                        shape = 1
                else:
                    shape = 1

                self.tensor[model[key]['var'][0]] = tfd.Sample(
                    self.create_distribution(model[key]['var'][1],
                    *self.convert_to_numeric_prior(model[key]['var'][2])), sample_shape = shape)
            
        self.priors = p
   
    def tensor_main(self):
        for key in self.main_text.keys():
            self.tensor[key] = eval(f"{self.main_text[key]}")