#%%
from code.model_diagnostic import *
from code.model_fit import *
from code.model_write import *
from code.data_manip import *
import pandas as pd

d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

#%% Test No data frame single likelihood -----------------------------------------------------
model = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)')    


model = build_model(model, float = 16)        


posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(y =  d.weight.astype('float32').values),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test No data frame multiple likelihood-----------------------------------------------------
model = dict(main = 'y~Normal(m,s)',
            likelihood = 'm ~  alpha + beta',
            prior1 = 's~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,1)',
            prior3 = 'beta ~ Normal(0,1)',
            
            main1 = 'z~Normal(m2,s2)',
            likelihood2 = 'm2 ~ alpha2 + beta2',
            prior4 = 's2~Exponential(1)',
            prior5 = 'alpha2 ~ Normal(0,1)',
            prior6 = 'beta2 ~ Normal(0,1)')    

model = build_model(model, float = 16)        


posterior, trace, sample_stats =  run_model(model, 
                                            observed_data = dict(y = d.weight.astype('float32').values),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Test with data frame in likelihood-----------------------------------------------------
d = pd.read_csv('./data/Howell1.csv', sep=';')
d = d[d.age > 18]
d.weight = d.weight - d.weight.mean()
weight = d.weight
d.age = d.age - d.age.mean()
age = d.age

formula = dict(main = 'height ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * weight',
            prior1 = 'sigma~Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,20)',
            prior3 = 'beta ~ Normal(0,1)')    

model = build_model(formula, df = d, sep = ',', float=32)


posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(height =  'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Test  with data frame only as output-----------------------------------------------------
model = dict(main = 'height ~ Normal(mu,sigma)',
            prior1 = 'mu ~ Normal(178.0, 0.1)',            
            prior2 = 'sigma ~ Uniform(0.0, 50.0)')    


model = build_model(model, path = None, df = d, sep = ',', float=32)

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(height = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test with multiple likelihood-----------------------------------------------------
model = dict(main = 'z ~ Poisson(alpha)',
            likelihood = 'alpha ~ a + beta * weight',
            prior1 = 'a ~ Normal(178,20)',
            prior2 = 'beta ~ Normal(0,1)',
            
            main2 = 'y ~ Normal(mu, sigma)',
            likelihood2 = 'mu ~ alpha2 + beta * age',
            prior4 = 'sigma ~ Normal(0.0, 1.0)',
            prior5 = 'alpha2 ~ Normal(0,1)'
            )    

model = build_model(model, path = None, df = d, sep = ',', float=32)

model.sample()

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Test error modeling-----------------------------------------------------
# Initialize a single 2-variate Gaussian.
mvn = tfd.MultivariateNormalTriL(
    loc=[1., 10, -10],
    scale_tril = [[1., 0.9, -0.8],
                  [0.9, 1, 0.4],
                  [-0.8, 0.4, 1]])


df = pd.DataFrame(mvn.sample(30).numpy())
df.columns = ['y', 'att', 'exposure']
df.plot()
df.corr()

model = dict(main = 'y ~ Normal(mu, sigma)',
            likelihood = 'mu ~ a + b * att + exposureM',
            prior1 = 'a ~ Normal(0,1)',
            prior2 = 'b ~ Normal(0,1)',
            prior3 = 'sigma~Normal(0,1)',
            
            main2 = 'exposureM ~ Normal(mu2, sigma2)',
            likelihood2 = 'mu2 ~ a2 + b2 * att + b3 * exposure',
            prior4 = 'a2 ~ Normal(0,1)',
            prior5 = 'b2 ~ Normal(0,1)',
            prior6 = 'b3 ~ Normal(0,1)',
            prior7 = 'sigma2~Normal(0,1)'
            )

model = build_model(model, path = None, df = df, sep = ',', float=32)

posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'y'),
                                            num_chains = 4, inDF = True)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Categorical models with OHE code 5.45
d = pd.read_csv('./data/Howell1.csv', sep=';')
formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + beta * male',
            prior1 = 'sigma~Uniform(0,50)',
            prior2 = 'alpha ~ Normal(178,100)',
            prior3 = 'beta ~ Normal(0,10)')     

model = build_model(formula, df = d, sep = ',', float=32)
posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'height'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

#%% Categorical models with OHE multiple categories code 5.52
d = pd.read_csv('./data/milk.csv', sep=';')
d = OHE(d, ['clade'])
d.columns

formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha + B1*clade_New_World_Monkey + B2*clade_Old_World_Monkey + B3*clade_Strepsirrhine',
            prior1 = 'sigma~Uniform(0,10)',
            prior2 = 'alpha ~ Normal(0.6,10)',
            prior3 = 'B1 ~ Normal(0,1)',
            prior4 = 'B2 ~ Normal(0,1)',
            prior5 = 'B3 ~ Normal(0,1)',)     

model = build_model(formula, df = d, sep = ',', float=32)
posterior, trace, sample_stats =  fit_model(model, 
                                            observed_data = dict(y = 'kcal_per_g'),
                                            num_chains = 4)

az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)


#%% Categorical models with index and multiple categories code 5.45
d = pd.read_csv('./data/milk.csv', sep=';')
d = index(d,["clade"])
d["kcal_per_g"] = d["kcal_per_g"].pipe(lambda x: (x - x.mean()) / x.std())
CLADE_ID_LEN = len(set(d.index_clade.values))
formula = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha[index_clade]',
            prior1 = 'sigma~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,0.5)')   
#%%
# index must be within main equation to handle symbolic tensor
m = tfd.JointDistributionNamed(dict(
    sigma = tfd.Sample(tfd.Exponential(1), sample_shape=1),
    alpha = tfd.Sample(tfd.Normal(0, 0.5), sample_shape=4),
    y = lambda alpha, sigma: tfd.Independent(tfd.Normal(
        loc=tf.transpose(tf.gather(tf.transpose(alpha), tf.cast(d.index_clade , dtype= tf.int32))),
        scale=sigma
    ), reinterpreted_batch_ndims=1),
))
#%%
posterior, trace, sample_stats =  run_model(model = m,
parallel_iterations=1,
num_results=2000,
num_burnin_steps=500,
step_size=0.065,
num_leapfrog_steps=5,
num_adaptation_steps=400,
num_chains=4,
observed_data = dict(y = d.kcal_per_g.astype('float32').values,))

#%%
# alpha indeces must be gather
ape_alpha, nwm_alpha, owm_alpha, strep_alpha = tf.split(
    posterior["alpha"][0], 4, axis=1
)

updated_posterior = {
    "ape_alpha": ape_alpha.numpy(),
    "nwm_alpha": nwm_alpha.numpy(),
    "owm_alpha": owm_alpha.numpy(),
    "strep_alpha": strep_alpha.numpy(),
    "sigma": posterior["sigma"][0],
}


az.summary(trace, round_to=2, kind="stats", hdi_prob=0.89)

  
# %%
model_type = check_index(formula)
sep = ';', 
path = 'output/mymodel.py', 
withDF = True, 
DFpath = None,
float = 16
df = d
df.to_csv('output/mydf.csv', index=False)
path = 'output/mydf.csv'
model = formula
full_model = get_var(model)

issues = get_undeclared_params(full_model, df = df)
issues
data = issues['params_in_data']
sep= ","

# %%
path = 'output/mymodel.py'
if model_type == "model_index":
    write_header_with_dataFrame(output_file = path, DFpath = 'output/mydf.csv', data = data, float =  float, sep = sep)
    p = write_priors(full_model, path, float)

#%%
model = full_model


mainsOutput = [] 
for key in model.keys():
    input = model[key]['input']
    var = model[key]['var']
    if 'main' in key.lower():   
        mainsOutput.append(var[0])

# Get all params     
lParam = [] 
for key in model.keys():
    input = model[key]['input']
    var = model[key]['var']
    if 'likelihood' in key.lower():   
        lParam.append(var[1]) 
lParam = [item.replace(' ', '') for sublist in lParam for item in sublist]

# Fint all main(s)
for key in model.keys():
    input = model[key]['input']
    var = model[key]['var']
    if 'main' in key.lower():           
        with open(output_file,'a') as file:  
            file.write('\n')    
            # Find next likelihood                                       
            if 'likelihood' in model.keys():                      
                formula = get_likelihood(model, key)   
                if formula is not None:
                    file.write('\t') 
                    params = re.split(r'[+***-]',formula[0])
                    params_in_prior =[x for x in params if x in p]
                    params_in_main =[x for x in params if x in mainsOutput]
                    if len(var[2]) > 1:
                        params = params_in_prior + params_in_main + [var[2][1]]   
                    else:
                        params = params_in_prior + params_in_main                 
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


#%%
from tensorflow_probability import distributions as tfd
import pandas as pd
import numpy as np  
import re
class model:
    def __init__(self, formula = None, float = 16):      
        self.f = formula
        self.df = pd.DataFrame({'A' : []})
        self.model_path = 'output/mymodel.py'
        self.df_path = 'output/mydf.csv'
        self.data_modification = {}
        self.float = float

    # Import data----------------------------
    def import_csv(self, path, **kwargs):
        self.df_original_path = path
        self.df_args = kwargs
        self.df = pd.read_csv(path, **kwargs)
        return self.df
   
    # Data manipulation----------------------------
    def OHE(self, cols = 'all'):
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            OHE = pd.get_dummies(self.df, columns=colCat, dtype=int)
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            OHE = pd.get_dummies(self.df, columns=cols, dtype=int)

        OHE.columns = OHE.columns.str.replace('.', '_')
        OHE.columns = OHE.columns.str.replace(' ', '_')


        self.df = pd.concat([self.df , OHE], axis=1)
        self.data_modification['OHE'] = cols
        return OHE

    def index(self, cols = 'all'):
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            for a in range(len(colCat)):
                self.df["index_"+ colCat[a]] =  self.df.loc[:,colCat[a]].astype("category").cat.codes
                self.df["index_"+ colCat[a]] = self.df["index_"+ colCat[a]].astype(np.int64)
        else:
            if isinstance(cols, list) == False:
                cols = [cols]
            for a in range(len(cols)):
                self.df["index_"+ cols[a]] =  self.df.loc[:,cols[a]].astype("category").cat.codes
                self.df["index_"+ cols[a]] = self.df["index_"+ cols[a]].astype(np.int64)

        self.df.columns = self.df.columns.str.replace('.', '_')
        self.df.columns = self.df.columns.str.replace(' ', '_')

        self.data_modification['index'] = cols
        return self.df
  
    # Model definition----------------------------
    def find_index_position(self, text):
        pattern = r'\b(?:[^+\-*\/\(\)~]+|\([^)]+\]|[^[\]]+\])+'
        result = re.findall(pattern, text)
        position = []
        for a in range(len(result)):
            if "[" in result[a]:
                position.append(a-1)
        return position
    
    def model_type(self):
        model.type = {} 
        formula = self.f  
        indices = []
        var = []   
        if "likelihood" in f.keys():
            model.type["with_likelihood"] = True
        else:
            model.type["with_likelihood"] = False

        for key in formula.keys():
            if "likelihood" in key:
                if "[" in formula[key]:                    
                    model.type["with_indices"] = True
                    params = self.get_formula( formula = formula[key], type = 'likelihood') 
                    position = self.find_index_position(formula[key])  
                    id = [params[1][i] for i in position]
                    id_var = [params[1][i+1] for i in position]
                    model.type[key] = {"params": id} 
                    indices = indices + id
                    var = var + id_var
                else:
                    model.type["with_indices"] = False
            else:
                model.type["with_indices"] = False
        self.model_type = model.type
        # An index variable have a parameter and a variable associated (e.g. alpha[species])
        self.indices = indices
        self.indices_var = var
        return model.type

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
        self.model_type()
        self.get_var()
        self.get_undeclared_params()
        self.get_priors_names()
        self.get_mains_info()
        self.get_likelihood(main_name = "main") #!make it adptative
        return self.undeclared_params
    
    def get_mains_info(self):
        tmp = self.full_model
        mains_infos = {}
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
    
    # write model----------------------------
    def write_header(self):
        output_file = self.model_path
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
        for key in self.mains_infos:
            text = ""
            text = text + self.mains_infos[key]['ouput'] + " = lambda "
            text = text + "".join(self.mains_infos[key]['priors']) + ", "

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

                    #text = text + indices_var[a] + ", dtype= tf.int32)))"
            else:
                text = text + self.mains_infos[key]['likelihood_formula']

                #text = text + "tf.transpose(tf.gather(tf.transpose(alpha), tf.cast(d.index_clade , dtype= tf.int32)))"
        
        text = text + "))"
        self.main_text = text
        return text
    
    def write_main2(self):
        output_file = self.model_path
        text = self.main_text()

        with open(output_file,'a') as file: 
            file.write(text) 
    
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

        # Fint all main(s)
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

self = model()
self.import_csv(path = './data/milk.csv', sep = ';')
self.index(cols = "clade")
f = dict(main = 'y ~ Normal(mu,sigma)',
            likelihood = 'mu ~ alpha[index_clade]',
            prior1 = 'sigma~Exponential(1)',
            prior2 = 'alpha ~ Normal(0,0.5)')  
#f = dict(main = 'y ~ Normal(mu,sigma)',
#            prior1 = 'sigma~Exponential(1)',
#            prior2 = 'mu ~ Normal(0,0.5)')  


self.formula(f)
self.get_mains_info()


#%%
self.write_header()
self.write_priors()


self.write_main2()

#%%


