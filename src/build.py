import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import ast

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
    def __init__(self, formula = None, df = None):
        self.f = formula   
    # Utility functions----------------------------    
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
    
    def convert_indices(self, input_string, dtype='32'):
        pattern = r'(\w+)\[(.*?)\]'
        #output_string = re.sub(pattern, rf"tf.transpose(tf.gather(tf.transpose(\1), tf.cast(\2, dtype=tf.int{dtype})))", input_string)
        output_string = re.sub(pattern, rf" tf.squeeze(tf.gather(\1,tf.cast(\2, dtype=tf.int{dtype}), axis = -1))", input_string)
        #output_string = re.sub(pattern, rf" tf.gather(\1,\2, dtype=tf.int{dtype}), axis = -1)", input_string)
        return output_string
    
    def unlist(self, data, remove_chars=[" ", "]"], remove_empty=True):
        """
        Unlists nested lists of strings into a single, flat list, handling any level of nesting.
        Optionally removes empty strings and specified characters.

        Args:
            data: A list containing strings or nested lists of strings.
            remove_chars: A string or a list of characters to remove from each string (default: "").
            remove_empty: Whether to remove empty strings (default: True).

        Returns:
            A single list containing all the strings from the input data,
            with empty strings and specified characters removed if applicable.
        """

        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(unlist(item, remove_chars, remove_empty))  # Recursively unlist nested lists
            else:
                if remove_empty and not item:
                    continue  # Skip empty strings

                if remove_chars:
                    # Convert a list of characters to a translation table if necessary
                    if isinstance(remove_chars, list):
                        remove_chars = str.maketrans('', '', ''.join(remove_chars))

                    item = item.translate(remove_chars)  # Remove specified characters

                result.append(item)
        return result

    def args_add_df(self, main):
        if not self.df.empty:
            if len(main['params']['args']) > 0:            
                for b in range(len(main['params']['args'])):
                    if any(self.df.columns.str.strip().str.fullmatch(main['params']['args'][b])):
                        main['input'] = main['input'].replace(main['params']['args'][b], "df." + main['params']['args'][b])
                        main['params']['args'][b] = "df." + main['params']['args'][b]                    

            if len(main['params']['kwargs']) > 0:              
                for k in main['params']['kwargs'].keys():
                    if any(self.df.columns.str.strip().str.fullmatch(main['params']['kwargs'][k])):
                        main['input'] = main['input'].replace(main['params']['kwargs'][k], "df." + main['params']['kwargs'][k])
                        main['params']['kwargs'][k] = "df" + main['params']['kwargs'][k]


        return main

    def split_formula(self, formula):
        y, x = re.split(r'[~]',formula)
        y = y.replace(" ", "")
        x = x.replace(" ", "")
        return y, x

    def replace_exact_match(self, pattern, replacement, input_string):
        # Escape special characters in the pattern
        pattern = re.escape(pattern)
        # Define the regular expression pattern with word boundaries
        pattern = r'\b' + pattern + r'\b'
        # Use re.sub to replace only the exact match of the pattern with the replacement value
        modified_string = re.sub(pattern, replacement, input_string)
        return modified_string
    
    def which_prior_in_LinearOperatorDiag(self, input_string, output, dimMulti = False):
        match = re.search(r'LinearOperatorDiag\((.*?)\)', input_string)
        if match:
            prior_diag = match.group(1)
        
            match2 = re.search(r'concat\(\[(.*?)\]', input_string)
            if match2:
                extracted_string = match2.group(1)
                elements = [e.strip() for e in extracted_string.split(',')]
                self.model_info['Multilevel_diag'][prior_diag] = len(elements)
                if dimMulti:
                    self.model_info['Multilevel_indices_dim'][output] =  len(elements)

    def extract_mvn_dimetion(self):
        for key in self.model_info['Multilevel_indices'].keys():
            for k in self.mains.keys():
                for inner_dict in self.mains[k]['likelihood(s)'].values():
                    if key in inner_dict['formula']:
                        #print(f'find {key} in {k} formula')
                        if key in list(self.model_info['Multilevel_indices'].keys()):
                                dim = self.model_info['Multilevel_indices_dim'][key]
                                #print(f'find {key} in Multilevel_indices_dim with a dim of {dim}')
                                Multilvel_occurrences = [m.start() for m in re.finditer(key, self.mains[k]['formula'])]
                                Multilvel_occurrences_length = len(Multilvel_occurrences)
                                if Multilvel_occurrences_length == dim:
                                    input_string =  self.mains[k]['formula']
                                    offset = 0
                                    replacement_pattern = "tf.gather({}, {}, axis=-1)"
                                    #print(f'{key} is observed {Multilvel_occurrences_length} times in formula which correspond to declared dim: {dim}' )
                                    for i, index in enumerate(Multilvel_occurrences):
                                        index += offset  # Adjust index by current offset
                                        replacement = replacement_pattern.format(key, i)
                                        input_string = input_string[:index] + replacement + input_string[index + len(key):]
                                        offset += len(replacement)  - len(key)# Update offset by the difference in lengths
                                    self.mains[k]['formula'] = input_string

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

    # Extract informations functions --------
    def get_general_info(self, key, lk, is_main = False):
        lk[key] = {}
        lk[key]['input'] = self.f[key]
    
        # Extracting params
        lk[key]['output'], lk[key]['formula'] = self.split_formula(lk[key]['input'])
        args = re.split(r'[+*()*[*^*-*/*,]',lk[key]['formula'] )  
 
        lk[key]['args'] = self.unlist(args)

        lk[key]['args'] = np.unique(lk[key]['args']) 

        lk[key]['likelihood(s)'] = {}
        lk[key]['prior(s)'] = {}
    
        lk[key]['with_tensorflow'] = False
        lk[key]['with_distribution'] = False    
        lk[key]['distribution(s)'] = [] 
        return lk
    
    def get_params_info(self, key, lk, diagnostic = False):
        # Evaluate arguments
        to_remove = []
        for i in range(len(lk[key]['args'])):
            # For the moment we replace tf and tfd functions, but we may also ask user to directly write tf and tfd
            # replace tf function
            if lk[key]['args'][i] in list(tf_classes.keys()):   
                lk[key]['formula'] = self.replace_exact_match(lk[key]['args'][i], 'tf.' + lk[key]['args'][i], lk[key]['formula'] )
                lk[key]['with_tensorflow'] = True
    
            # replace tfd function
            elif lk[key]['args'][i]  in list(tfd_classes.keys()):        
                lk[key]['formula'] = self.replace_exact_match(lk[key]['args'][i], 'tfd.'+ lk[key]['args'][i], lk[key]['formula'])
                lk[key]['with_distribution'] = True
                lk[key]['distribution(s)'].append('tfd.'+ lk[key]['args'][i]) # Add distribution to 
                to_remove.append(i)

            if 'LinearOperatorDiag' in lk[key]['args'][i]:                
                lk[key]['formula'] = lk[key]['formula'].replace('LinearOperatorDiag', 'tf.linalg.LinearOperatorDiag')
                if 'MultivariateNormalTriL' in lk[key]['input']:
                    dimMulti = True
                else:
                    dimMulti = False
                self.which_prior_in_LinearOperatorDiag(lk[key]['formula'], output = lk[key]['output'], dimMulti = dimMulti)
                

            # replace df arguments
            if not self.df.empty:
                replace = False
                # If kwarg
                if '=' in lk[key]['args'][i]:
                    tmp = lk[key]['args'][i].split('=')
                    tmp1 = tmp[0]
                    tmp2 = tmp[1]

                    if any(self.df.columns.str.strip().str.fullmatch(tmp2)): #if item in columns
                        lk[key]['formula'] = self.replace_exact_match(lk[key]['args'][i], 
                                                tmp1 + '= df.' +tmp2 + """.astype('float""" + str(self.float) + """').values""",
                                                lk[key]['formula'])

                else:
                    if any(self.df.columns.str.strip().str.fullmatch(lk[key]['args'][i])) :
                        lk[key]['formula'] = self.replace_exact_match(lk[key]['args'][i], 
                                                                       'df.' + lk[key]['args'][i] + """.astype('float""" + str(self.float) + """').values""",
                                                                        lk[key]['formula'])

    
        #lk[key]['args'] = np.delete(lk[key]['args'], to_remove)
        # check for indices        
        lk[key]['with_indices'] = "[" in lk[key]['input'] 
        lk[key]['indices'] = self.model_info['indices'] | self.extract_indices_patterns(lk[key]['input'])
        lk[key]['formula'] = self.convert_indices(lk[key]['formula'], dtype = self.float)

         # check for Multilevel mdoel    
        if 'CholeskyLKJ' in lk[key]['formula']:
            self.model_info['Multilevel'] = True          

        if 'MultivariateNormalTriL' in lk[key]['formula']:   
                self.model_info['Multilevel'] = True  
                # Priors are treqted qfter likelihoods, we thus have lk indices storeed in  self.model_info['indices'].
                # We can thus have variable to which mvn is liked, e.g. the specify random effect. This will be used after merge functions to change the lk formula to retrieve the mvn columns
                self.model_info['Multilevel_indices'][lk[key]['output']] = self.model_info['indices'][lk[key]['output']]
              
        lk[key]['params'] = self.separate_args_kwargs(lk[key]['args']) 
    
        if "[" in lk[key]['input'] :
            self.model_info["with_indices"] = True 
            self.model_info["indices"] = lk[key]['indices']
    
        ## Cast numeric (can't be done before, otherwise separate_args_kwargs will concider casts as kwargs )
        #if len(lk[key]['params']['args']) > 0:
        #    for a in range(len(lk[key]['params']['args'])):            
        #        try:
        #            tmp = "tf.cast(" + str(float(lk[key]['params']['args'][a])) + ", dtype = tf.float" + str(self.float) + ')'
        #            lk[key]['formula'] =  self.replace_exact_match(lk[key]['params']['args'][a], tmp, lk[key]['formula'])
        #            #lk[key]['params']['args'][a] = tmp
        #        except ValueError:
        #            try:
        #                tmp = "tf.cast(" + str(int(lk[key]['params']['args'][a]))  + ", dtype = tf.float" + str(self.float) + ')'
        #                lk[key]['formula'] = self.replace_exact_match(lk[key]['params']['args'][a], tmp, lk[key]['formula'])
        #                #lk[key]['params']['args'][a] =  tmp
    #
        #            except ValueError:
        #                lk[key]['params']['args'][a] =  lk[key]['params']['args'][a]
    #
        #if len(lk[key]['params']['kwargs']) > 0:
        #    for k in lk[key]['params']['kwargs'].keys():            
        #        try:
        #            tmp = "tf.cast(" + str(float(lk[key]['params']['kwargs'][k])) + ", dtype = tf.float" + str(self.float) + ')'
        #            lk[key]['formula'] = self.replace_exact_match(lk[key]['params']['kwargs'][k], tmp, lk[key]['formula'])
        #            #lk[key]['params']['kwargs'][k] = tmp
        #        except ValueError:
        #            try:
        #                tmp = "tf.cast(" + str(int(lk[key]['params']['kwargs'][k]))  + ", dtype = tf.float" + str(self.float) + ')'
        #                lk[key]['formula'] = self.replace_exact_match(lk[key]['params']['kwargs'][k], tmp, lk[key]['formula'])
        #                #lk[key]['params']['kwargs'][k] =  tmp
    #
        #            except ValueError:
        #               lk[key]['params']['kwargs'][k] =  lk[key]['params']['kwargs'][k]
        if diagnostic:
            return lk # return more info for diagnostic
        else:
            new = {}
            new = {a: lk[key][a] for a in ["input", "output", "formula", "args", "likelihood(s)", "params", 'distribution(s)'] if a in lk[key]}
            lk[key] = new
            return  lk        
    
    def get_model_info(self, diagnostic = False):
        mains = {}
        lks = {}
        priors = {}
        for key in self.f.keys():
            if 'main' in key.lower():
                mains = self.get_general_info(key, mains)
                mains = self.get_params_info(key, mains, diagnostic)
                mains[key]["prior(s)"] = {}
    
            elif 'likelihood' in key.lower():
                lks =self.get_general_info(key, lks)
                lks = self.get_params_info(key, lks, diagnostic)
                lks[key]["prior(s)"] = {}
    
            elif 'prior' in  key.lower():
                priors = self.get_general_info(key, priors)
                priors = self.get_params_info(key, priors, diagnostic)
                priors[key]["prior(s)"] = {}

        self.mains = mains
        self.lks = lks
        self.priors = priors

    # Merge mains, lks, priors-------------------------------
    def merge_main_lks(self, mains, lks):         
        # find where lks are 
        for key in lks.keys():
            for i in mains.keys():  
                if len(mains[i]['params']["args"]) > 0:
                    for j, item  in  enumerate(mains[i]['params']["args"]):
                        if lks[key]["output"] == item:
                            mains[i]["likelihood(s)"][key] = lks[key] 
                            mains[i]['params']["args"][j] = lks[key]['formula'] # repalce main arg by lk formula
                            mains[i]['formula'] = self.replace_exact_match(item, lks[key]['formula'], mains[i]['formula'] )

                if len(mains[i]['params']["kwargs"]) > 0:
                     for k in  mains[i]['params']["kwargs"].keys():
                         if lks[key]["output"] == mains[i]['params']["kwargs"][k]:
                            mains[i]["likelihood(s)"][key] = lks[key]
                            #mains[i]['params']["kwargs"][k] = lks[key]['formula']# repalce main kwarg by lk formula
                            mains[i]['formula'] = self.replace_exact_match(mains[i]['params']["kwargs"][k], lks[key]['formula'], mains[i]['formula'] )
                            
                if len(mains[i]["likelihood(s)"]) > 1:
                     mains[i]["multiple_lks"] = True
                     self.model_info["multiple_lks"] = True
                else:
                    mains[i]["multiple_lks"] = False

        return mains

    def merge_main_priors(self,mains,priors):
        # find where lks are 
        for key in priors.keys():
            for i in mains.keys():
                if len(mains[i]['params']["args"]) > 0:
                    for item in  mains[i]['params']["args"]:
                        if priors[key]["output"] == item and priors[key]["output"] :
                            mains[i]["prior(s)"][key] = priors[key]

                if len(mains[i]['params']["kwargs"]) > 0:
                     for k in  mains[i]['params']["kwargs"].keys():
                         if priors[key]["output"] == mains[i]['params']["kwargs"][k] :
                             mains[i]["prior(s)"][key] = priors[key]
        return mains

    def merge_lks_priors(self, lks,priors):
        # find where lks are 
        to_remove = []
        for key in self.priors.keys():
            for i in lks.keys():  
                if len(lks[i]["args"]) > 0:
                    for item in  lks[i]["args"]:
                        if self.priors[key]["output"] ==  item:
                            lks[i]["prior(s)"][key] = self.priors[key]
                            to_remove.append(key)

        return lks

    def merge_priors_priors(self, priors):
        # find where lks are 
        for key in priors.keys():
            for i in priors.keys():  
                if len(priors[i]['params']["args"]) > 0:
                    for item in  priors[i]['params']["args"]:
                        if priors[key]["output"] == item:
                            priors[i]["prior(s)"][key] = priors[key]

                if len(priors[i]['params']["kwargs"]) > 0:
                     for k in  priors[i]['params']["kwargs"].keys():
                         if priors[key]["output"] == priors[i]['params']["kwargs"][k]:
                             priors[i]["prior(s)"][key] = priors[key]
        return priors

    def merge(self):        
        self.priors = self.merge_priors_priors(self.priors) # Priors of priors (need recursive?)
        self.lks = self.merge_lks_priors(self.lks, self.priors) # Priors of likelihoods 
        self.mains = self.merge_main_lks(self.mains, self.lks) # likelihoods of mains 
        self.mains = self.merge_main_priors(self.mains, self.priors) # Priors of mains     

        if len(self.mains) > 1 :
            self.model_info['multiple_mains'] = True
        else:
            self.model_info['multiple_mains'] = False

        if len(self.lks) > 1 :
            self.model_info['multiple_lks'] = True
        else:
            self.model_info['multiple_lks'] = False

        if len(self.priors) > 1 :
            self.model_info['multiple_priors'] = True
        else:
            self.model_info['multiple_priors'] = False

class write():
    def __init__(self, model_dict):
        self.model_dict = model_dict
    # Utility functions -------------------------
    def eval_expression(self, expr):
        # Parse the expression as an abstract syntax tree (AST)
        expr_ast = ast.parse(expr, mode='eval')

        # Traverse the AST and evaluate the expression
        expr_value = eval(compile(expr_ast, filename="<ast>", mode="eval"))
        return expr_value
    
    def create_function_from_string(self, func_str, name):
        # Define required imports and namespace for exec
        imports = {'tfp':tfp, 'tfd': tfd, 'tf': tf, 'df': self.df}
        namespace = {}
        # Execute the string as Python code within the specified namespace
        exec( name + ' = ' + func_str, imports, namespace)

        # Extract the function from the namespace
        return namespace[name]
    
    # Priors entries
    def write_priors(self):
        # For non Multilevel models
        for key in self.priors.keys():
           
            self.priors_name.append(self.priors[key]["output"])
            text = ''

            # Setup prior shape------------------------
            ## Prior for Multilevel model
            if self.priors[key]["output"] in list(self.model_info['Multilevel_diag'].keys()):
                shape = self.model_info['Multilevel_diag'][ self.priors[key]["output"]]
            
            ## Prior with CholeskyLKJ
            elif 'CholeskyLKJ' in self.priors[key]['formula'] :
                shape = ()

            elif 'LKJ' in self.priors[key]['formula'] :
                shape = ()
                
            # #Default prior shape
            else:
                shape = 1

            # Setup tensor text---------------------
            # For prior without hyperpriors
            if len(self.priors[key]['prior(s)']) == 0:
                text = "tfd.Sample("
                text = text + self.priors[key]["formula"] 
                text = text[:-1]
                text = text + """, name = '""" + key + """')"""
                text = text + ", sample_shape = "

                ## Change shape if prior in indices
                if self.model_info['with_indices']:
                    if self.priors[key]["output"]  in self.model_info["indices"].keys():
                        shape = self.df[self.model_info["indices"][self.priors[key]["output"]]].nunique()

                text =  text = text + str(shape) + ")"
            
            # For prior with hyperpriors
            else :
                text = 'lambda '
                ## Get hyperpriors
                for k in self.priors[key]['prior(s)'].keys():
                    text = text +  self.priors[key]['prior(s)'][k]['output'] + ', '
                text = text[:-2] + ':'
                text = text +' tfd.Sample(' + self.priors[key]['formula']
                
                ## Change shape if prior in indices
                if self.model_info['with_indices']:
                    if self.priors[key]["output"]  in self.model_info["indices"].keys():
                        shape = self.df[self.model_info["indices"][self.priors[key]["output"]]].nunique()
                
                text = text + ', sample_shape = ' +  str(shape) + ')'

            self.prior_dict[self.priors[key]["output"]] = text
            self.model_dict[self.priors[key]["output"]] = text   

        # Evaluate the expressions in the dictionary values
        self.model_dict = {key: self.eval_expression(value) for key, value in self.model_dict.items()}

    # Mains entries
    def write_mains(self):
        self.main_dict = {}
        for key in self.mains:
            if len(self.mains[key]) > 0 :
                # Mains distribution 
                text = "lambda "

                #Get main priors
                lambda_params = []
                if len(self.mains[key]['prior(s)']) > 0 :
                    for k in self.mains[key]['prior(s)'].keys():
                        text = text + self.mains[key]['prior(s)'][k]["output"] + ', '
                        lambda_params.append(self.mains[key]['prior(s)'][k]["output"])

                # Get main lk priors
                if len(self.mains[key]['likelihood(s)']) > 0:
                    # For each LK
                    for a in self.mains[key]['likelihood(s)'].keys():
                        # Check its priors and add them to lambda
                        if len(self.mains[key]['likelihood(s)'][a]['prior(s)']) > 0:
                            for b in  self.mains[key]['likelihood(s)'][a]['prior(s)'].keys():
                                if  self.mains[key]['likelihood(s)'][a]['prior(s)'][b]["output"] not in lambda_params:
                                    text = text + self.mains[key]['likelihood(s)'][a]['prior(s)'][b]["output"]  + ', '


                #text = text[:-2] + " : tfd.Independent("+ self.mains[key]['distribution(s)'][0] + "(" # First argument is the distribution
                
                #if len(self.mains[key]['params']['args']) > 0:
                #    text = text + ", ".join(self.mains[key]['params']['args']) + "," # Remove first argument 

                #if len(self.mains[key]['params']['kwargs']) > 0:
                #    for k, v in  self.mains[key]['params']['kwargs'].items():
                #        text = text + k + " = " + v + ","
                #text = text[:-1] +  ', name =' + "'" + str(key) + "'" + "), reinterpreted_batch_ndims=1)"
                                    
                text = text[:-2] + " : tfd.Independent("+ self.mains[key]['formula'] + ')'

                text = text[:-2] +  ', name =' + "'" + str(key) + "'" + "), reinterpreted_batch_ndims=1)"

                self.main_dict[self.mains[key]["output"]] = text

    def build_tensor(self):
        for key in  self.main_dict.keys():
                self.model_dict[key] = self.create_function_from_string(func_str =  self.main_dict[key], name = key)

    def write_tensor(self):
        self.write_priors()
        self.write_mains()
        self.build_tensor()
        self.tensor = tfd.JointDistributionNamed(self.model_dict)
        self.priors_dict = self.priors
        self.priors = list(self.priors.keys())