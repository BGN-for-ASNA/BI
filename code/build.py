from tensorflow_probability import distributions as tfd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

def get_distribution_classes():
    # Get all names defined in the distributions module
    all_names = dir(tfd)
    
    # Filter to include only classes
    class_names = [name for name in all_names if isinstance(getattr(tfd, name), type)]
    
    # Create a dictionary of class names and corresponding classes
    class_dict = {name: getattr(tfd, name) for name in class_names}
    
    return class_dict
def exportTFD(tfd_classes):
    for key in tfd_classes.keys():
        globals()[key] = tfd_classes[key]
tfd_classes = get_distribution_classes()


def get_tensorflow_classes():
    # Get all names defined in the distributions module
    all_names = dir(tf)
    
    # Create a dictionary with all names
    class_dict = {name: getattr(tf, name) for name in all_names}
    
    return class_dict
def exportTF(tf_classes):
    for key in tf_classes.keys():
        globals()[key] = tf_classes[key]
tf_classes = get_tensorflow_classes()


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
        
    def unlist(self, L):
        newList = []
        for item in L:
            if isinstance(item, list):
                for i in item:
                    newList.append(i)
            else:
                newList.append(item)
        return newList
    
    def is_float(self, char):
        try:
            float(char)
            return True
        except ValueError:
            return False

    def remove_numeric_priors(self, characters):
     return [char for char in characters if not self.is_float(char)]

    # Get model informations----------------------------    
    def get_model_type(self):
        type = {} 
        type["with_likelihood"] = False
        type["with_indices"]  = False
        type["with_tf"] = self.Tensoflow
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
                    print(formula[key])
                    params = self.get_formula( formula = formula[key], type = 'likelihood') 
                    print(params)
                    position = self.find_index_position(formula[key])  
                    print(position)
                    indices_dict = self.extract_indices_patterns(formula[key])
                    print(indices_dict)
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
                #args = re.split(r'[+*()*[*,]',x)          
                args = re.split(r'[+*()*[*^*-*/*,]',x)            
                new = []
                for i in range(len(args)):
                    if args[i] != '':
                        args[i] = args[i].replace("]", "")
                        new.append(args[i])
                return [y, new]     
            else:
                #args = re.split(r'[+*()*[*,]',x)  
                args = re.split(r'[+*()*[*^*-*/*,]',x)
                new = []
                dist = []
                Tensorflow = []
                for i in range(len(args)):
                    if args[i] in list(tf_classes.keys()):

                        Tensorflow.append('tf.' + str(args[i]))
                        new.append('tf.' + str(args[i]) + '(')
                        self.Tensoflow = True

                    elif args[i] in list(tfd_classes.keys()):    
                        dist.append('tfd.' + str(args[i]))
                        new.append('tfd.' + str(args[i]) + '(')

                    elif args[i] not in list(tf_classes.keys()) and args[i] in list(tfd_classes.keys()):
                        new.append(args[i])                  
                return [y,dist, new, Tensorflow] 

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
            undeclared_params2 = [x for x in undeclared_params2 if '=' not in x]
            self.undeclared_params =  {'undeclared_params': undeclared_params2}
            return {'undeclared_params': undeclared_params2}
        
        else:
            test = pd.Index(undeclared_params2).difference(self.df.columns).tolist()
            test2 =  list(set(undeclared_params2) & set(self.df.columns))
            test =  list(set(test).difference(mainOutput))
            test = [x for x in test if '=' not in x]
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
    ## Build dictionary to store main informations
    def get_main_dict(self):
        dict = {'output': None, 
                'input': None,
                'distribution': None, 
                'params' : None,
                'priors' : None,
                'with_likelihood': None,
                'multiple_likelihoods': None,
                'with_indices' : False,
                'params_in_df': None,
                'params_not_in_df': None,
                'likelihood(s)': {}
                } 
                
        return dict
    
    def get_likelihood_dict(self):
        LK_dict = {}
        return LK_dict

    def get_main_info_likelihood_indices(self,dict):
        dict['with_indices'] = True
        dict['indices_position'] = self.find_index_position(dict['formula'])  
        dict['indices_patterns'] = self.extract_indices_patterns(dict['formula'])  
        dict['indices_prior'] = [dict['params'][i] for i in dict['indices_position']]
        dict['indices_var'] = [dict['params'][i+1] for i in dict['indices_position']]
        return dict

    def get_main_info_with_likelihood(self,dict, LK_dict):         
            if len(dict['params']['args']) > 0:
                LK_dict = self.get_main_info_likelihood_args(dict, LK_dict)
            if len(dict['params']['kwargs']) > 0:
                LK_dict = self.get_main_info_likelihood_kwargs(dict, LK_dict)        
            return  LK_dict

    def get_main_info_likelihood_kwargs(self, main_dict, LK_dict):
        for key in main_dict['params']['kwargs']:
            if main_dict['params']['kwargs'][key] in self.model_names.values():               
                #check if self.model_names.values() name have likelihood in it
                key_name = self.which_key_have_value(self.model_names, main_dict['params']['kwargs'][key])
                if 'prior' in key_name:
                    main_dict['priors'] = self.add_entry_if_not_none(main_dict['priors'], self.model_names[key_name])

                if 'likelihood' in key_name: # Get basic information of likelihood
                    name = [k for k, v in self.model_names.items() if v == main_dict['params']['kwargs'][key]] 
                    name = name[0]
                    LK_dict[name] = {}   
                    LK_dict[name]['main_params']= {'args':[], "kwargs":{}} 
                    LK_dict[name]['formula'] = self.full_model[name]['input']
                    LK_dict[name]['output'] = self.full_model[name]['var'][0]
                    LK_dict[name]['params'] = self.full_model[name]['var'][1]
                    x = re.split(r'[~]',LK_dict[name]['formula'])[1]
                    x = x.replace(" ", "")
                    LK_dict[name]['main_params']['kwargs'][key] = str(x)

                    # Add information to main to build LK within lambda
                    main_dict['params']['kwargs'][key] = LK_dict[name]['main_params']['kwargs'][key]
                    main_dict['priors'] = self.add_entry_if_not_none(main_dict['priors'], LK_dict[name]['params'].copy())

                    if "[" in LK_dict[name]['formula']: 
                        main_dict['with_indices']   = True  
                        LK_dict[name] = self.get_main_info_likelihood_indices(LK_dict[name])

                    if len(self.undeclared_params) > 1:
                        LK_dict[name]['params_in_df'] = [x for x in  LK_dict[name]['params'] if x in self.undeclared_params['params_in_data']]                          
                        LK_dict[name]['params_not_in_df'] = [x for x in  LK_dict[name]['params']  if x not in self.undeclared_params['params_in_data']]

                        if len(LK_dict[name]['params_in_df']) > 0:
                            main_dict['params_in_df'] = self.add_entry_if_not_none(main_dict['params_in_df'], LK_dict[name]['params_in_df'])
                        main_dict['params_not_in_df'] = self.add_entry_if_not_none(main_dict['params_not_in_df'], LK_dict[name]['params_not_in_df'])

                    else:
                        LK_dict[name]['params_in_df'] = None
                        main_dict['params_in_df'] = {}
                        main_dict['params_in_df'] = None

        #main_dict['likelihood(s)'].update(LK_dict)
        return LK_dict

    def get_main_info_likelihood_args(self,main_dict, LK_dict):
        # Find likelihood associated to this main
        for a in range(len(main_dict['params']['args'])):
            if main_dict['params']['args'][a] in self.model_names.values():
                #check if self.model_names.values() name have likelihhod in it
                key_name = self.which_key_have_value(self.model_names, main_dict['params']['args'][a])
                if 'prior' in key_name:
                    main_dict['priors'] = self.add_entry_if_not_none(main_dict['priors'], self.model_names[key_name])

                if 'likelihood' in key_name:    
                    name = key_name
                    LK_dict[name] = {}
                    LK_dict[name]['main_params'] =  main_dict['params'] # get corresponding main prior
                    LK_dict[name]['formula'] = self.full_model[name]['input']
                    LK_dict[name]['output'] = main_dict['params']['args'][a]
                    LK_dict[name]['params'] = self.full_model[name]['var'][1] 
                    x = re.split(r'[~]',LK_dict[name]['formula'])[1]
                    x = x.replace(" ", "")
                    LK_dict[name]['main_params']['args'][a] = x
                    # Add information to main to build LK within lambda
                    main_dict['params']['args'][a] = LK_dict[name]['main_params']['args'][a]
                    main_dict['priors'] = self.add_entry_if_not_none(main_dict['priors'], LK_dict[name]['params'])

                    # Index in formula
                    if "[" in LK_dict[name]['formula']:   
                        main_dict['with_indices']   = True             
                        LK_dict[name] = self.get_main_info_likelihood_indices(LK_dict[name])

                    if  len(self.undeclared_params) > 1:
                        LK_dict[name]['params_in_df'] = [x for x in  LK_dict[name]['params'] if x in self.undeclared_params['params_in_data']]                          
                        LK_dict[name]['params_not_in_df'] = [x for x in  LK_dict[name]['params']  if x not in self.undeclared_params['params_in_data']]

                        if len(LK_dict[name]['params_in_df']) > 0:
                            main_dict['params_in_df'] = self.add_entry_if_not_none(main_dict['params_in_df'], LK_dict[name]['params_in_df'])
                        main_dict['params_not_in_df'] = self.add_entry_if_not_none(main_dict['params_not_in_df'], LK_dict[name]['params_not_in_df'])
                    else:
                        LK_dict[name]['params_in_df'] = None

            if len(self.undeclared_params) > 1:# POurquoi n'est pas présent dans kwargs?
                if main_dict['params']['args'][a] in self.undeclared_params['params_in_data']:
                        main_dict['params']['args'][a] = "df."+ main_dict['params']['args'][a] + """.astype('float""" + str(self.float) + """').values"""

        return LK_dict

    def get_main_info_likelihood_arguments_in_df(self,dict):
        if len(dict['params']['args']) > 0:
            for a in range(len(dict['params_in_df'])):
                for b in range(len(dict['params']['args'])):
                    if dict['params_in_df'][a] in dict['params']['args'][b]:
                         dict['params']['args'][b] = dict['params']['args'][b].replace(dict['params_in_df'][a], 
                                                                                      'df.' + dict['params_in_df'][a] + '.values')
                        #dict['params']['args'][b] = dict['params']['args'][b].replace(dict['likelihood_params_in_df'][a], 
                        #                                                              ' tf.cast(df.' + dict['likelihood_params_in_df'][a] + ',dtype=tf.float' + str(self.float)+ ')')
        if len(dict['params']['kwargs']) > 0:
            for a in range(len(dict['params_in_df'])):
                for key in dict['params']['kwargs'].keys():
                    if dict['params_in_df'][a] in dict['params']['kwargs'][key]:
                        dict['params']['kwargs'][key] = dict['params']['kwargs'][key].replace(dict['params_in_df'][a], 
                                                                                              'df.' + dict['params_in_df'][a] + '.values')
                        #dict['params']['kwargs'][key] = dict['params']['kwargs'][key].replace(dict['likelihood_params_in_df'][a], 
                        #                                                                      ' tf.cast(df.' + dict['likelihood_params_in_df'][a] + ',dtype=tf.float' + str(self.float) + ')')
        return dict   

    def get_mains_info(self):    
        tmp = self.full_model
        mains_infos = {}
        # For each main retrieve informations
        for key in tmp.keys():
            if 'main' in key.lower():                
                main_name = key
                infos = tmp[key]['var']
                main_dict = self.get_main_dict()
                main_dict['output'] = infos[0]
                main_dict['distribution'] = infos[1]
                main_dict['params'] = infos[2]
                main_dict['params'] = self.separate_args_kwargs(main_dict['params'])
                main_dict['input'] = tmp[key]['input']
                main_dict['priors'] = self.priors

                # Handle Likelihood
                if self.model_type['with_likelihood']:
                    main_dict['with_likelihood'] = True
                    LK_dict = {}
                    LK_dict = self.get_main_info_with_likelihood(main_dict, LK_dict)
                else:
                    LK_dict = {}
                
                main_dict['likelihood(s)'].update(LK_dict)

                if len(LK_dict) > 1 :
                    main_dict['multiple_likelihoods'] = True
                
                # Handle data frame data
                if main_dict['params_in_df'] is not None:
                    main_dict['params_in_df'] = list(set(main_dict['params_in_df']))
                    main_dict = self.get_main_info_likelihood_arguments_in_df(main_dict)

                # Clean main arguments
                main_dict['params']['args'] = self.convert_to_numeric(main_dict['params']['args'])
                main_dict['params']['kwargs']= self.convert_to_numeric_dict(main_dict['params']['kwargs'])

                # Clean priors
                if main_dict['priors'] is not None:
                    main_dict['priors'] = self.unlist(main_dict['priors'])
                    main_dict['priors'] = self.remove_numeric_priors(main_dict['priors'])

                    if main_dict['params_in_df'] is not None:
                        main_dict['priors'] =[char for char in main_dict['priors'] if char not in main_dict['params_in_df']]

                    main_dict['priors'] = list(np.unique(main_dict['priors']))

                #  Update model info
                if main_dict["with_indices"] == True:
                    self.model_type["with_indices"] = main_dict["with_indices"]

                mains_infos[main_name] = main_dict

        self.mains_infos = mains_infos

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
                lst[key] = " tf.cast(" + str(float(lst[key] )) + ", dtype = tf.float" + str(self.float) + ')'
            except ValueError:
                try:
                    lst[key] =  " tf.cast(" + str(int(lst[key] ))  + ", dtype = tf.float" + str(self.float) + ')'
                except ValueError:
                    lst[key] = lst[key]
                     # Keep as is if not numeric
        return lst

    # Wirte main functions -------------------------------------
    def convert_indices(self, input_string, dtype='32'):
        pattern = r'(\w+)\[(.*?)\]'
        #output_string = re.sub(pattern, rf"tf.transpose(tf.gather(tf.transpose(\1), tf.cast(\2, dtype=tf.int{dtype})))", input_string)
        output_string = re.sub(pattern, rf" tf.squeeze(tf.gather(\1,tf.cast(\2, dtype=tf.int{dtype}), axis = -1))", input_string)
        #output_string = re.sub(pattern, rf" tf.gather(\1,\2, dtype=tf.int{dtype}), axis = -1)", input_string)
        return output_string
    
    def write_main_text_no_indices2(self, mains_infos, text):
        print("No indices in main" )
        if mains_infos['with_likelihood']:
            print("With LK")
            if len(mains_infos['params']['args']) > 0:
               text = text + ','.join(mains_infos['params']['args']) + ','
            if len(mains_infos['params']['kwargs']) > 0:
                for key in mains_infos['params']['kwargs'].keys():                    
                    text = text + str(key) + '=' + str(mains_infos['params']['kwargs'][key])  + ','              
            text = text + ')'
        else:
            print("Without LK")
            if len(mains_infos['params']['args'])>0:                     
                    text = text +  ','.join(mains_infos['params']['args'])+ ','
            if len(mains_infos['params']['kwargs'])>0: 
                    for k in mains_infos['params']['kwargs'].keys():  
                        text = text + str(k)+ ' = ' +  mains_infos['params']['kwargs'][k] + ','
        return text
    
    def write_main_text_no_indices(self, mains_infos, text):
        print("No indices in main" )
        if mains_infos['with_likelihood']:
            print("With LK")
            if len(mains_infos['params']['args']) > 0:
               text = text + ','.join(mains_infos['params']['args']) + ','
            if len(mains_infos['params']['kwargs']) > 0:
                tmp = [] 
                for k in mains_infos['params']['kwargs'].keys():                                       
                    tmp.append(str(k)+ ' = ' +  mains_infos['params']['kwargs'][k])
                text = text + ','.join(tmp)         
            text = text + ')'
        else:
            print("Without LK")
            if len(mains_infos['params']['args'])>0:                     
                    text = text +  ','.join(mains_infos['params']['args'])+ ','
            if len(mains_infos['params']['kwargs'])>0: 
                tmp = []
                for k in mains_infos['params']['kwargs'].keys():  
                    tmp.append(str(k)+ ' = ' +  mains_infos['params']['kwargs'][k])
                text = text + ','.join(tmp)
            text = text + ')'
        return text  
    
    def write_main_text(self):
        result = {}
        for key in self.mains_infos.keys():
            text = ""
            text = text +  "lambda "

            # Params
            ## likelihood priors 
            if self.mains_infos[key]['priors'] is not None:
                text = text + ','.join(self.mains_infos[key]["priors"]) + ', '

            ## likelihood params
            #if self.mains_infos[key]["likelihood_params"] is not None:
            #    text = text + ", ".join(self.mains_infos[key]["likelihood_params"])

            # likelihood distribution
            text = text + ": tfd.Independent(tfd." + self.mains_infos[key]['distribution'] + "(" 

            # likelihood formula
            if self.mains_infos[key]['with_indices']:
                #text = self.write_main_text_indices(self.mains_infos[key], text)
                if len(self.mains_infos[key]['params']['args'])>0:
                    for a in range(len(self.mains_infos[key]['params']['args'])):                        
                        text = text + self.convert_indices(self.mains_infos[key]['params']['args'][a], self.float)+ ','

                if len(self.mains_infos[key]['params']['kwargs'])>0:
                    for k in self.mains_infos[key]['params']['kwargs'].keys():
                        text = text + str(k)+ ' = ' +  self.convert_indices(self.mains_infos[key]['params']['kwargs'][k], self.float) + ','
                
                text = text + 'name =' + "'" + str(key) + "'" + "), reinterpreted_batch_ndims=1)"

            else:
                text = self.write_main_text_no_indices(self.mains_infos[key], text)
                text = text + ", reinterpreted_batch_ndims=1)"

            result[self.mains_infos[key]['output']] = text
        
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
                    *self.convert_to_numeric_prior(model[key]['var'][2]), 
                    **{'name': str(key)}), sample_shape = shape)
            
        self.priors = p
   
    def tensor_main(self):
        for key in self.main_text.keys():
            self.tensor[key] = eval(f"{self.main_text[key]}")