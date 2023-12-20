import pandas as pd
import flet as ft
def get_formula(formula = "y~Normal(0,1)", type = 'likelihood'):
    import re
    y, x = re.split(r'[~=]',formula)
    y = y.replace(" ", "")

    if type != 'likelihood':
        dist, args = x.split('(')
        dist = dist.replace(" ", "")
        args = args.replace("(", "")
        args = args.replace(")", "")
        args = args.split(",") 
        #args = args.replace(" ", "")
        return [y, dist, args]
    else:
        y, x = re.split(r'[=]',formula)
        y = y.replace(" ", "")    
        args = re.split(r'[+*()]',x)
        for i in range(len(args)):
            args[i] = args[i].replace(" ", "") 
        return [y, args]        
   
def headers(df : pd.DataFrame) -> list:
    return [ft.DataColumn(ft.Text(header)) for header in df.columns]

def rows(df : pd.DataFrame) -> list:
    rows = []
    for index, row in df.iterrows():
        rows.append(ft.DataRow(cells = [ft.DataCell(ft.Text(row[header])) for header in df.columns]))
    return rows  

def textInput_to_list(value):
    result = []
    current_word = ""
    for char in value:
        if char != ',':
            current_word  += char
        else:
            if current_word:
                result.append(current_word)
            current_word = ""
    result.append(current_word)   
    return result
def undeclared_params(model, df):
    Vars = []
    params = []
    for key in model.keys():
        tmp = model[key]['model']
        for a in range(len(tmp)):
            if isinstance(tmp[a], list):
                params.append(tmp[a])
            else:
                if a == 0:
                    Vars.append(tmp[0].replace(' ', ''))                
    params = [item.replace(' ', '') for sublist in params for item in sublist]

    #del Vars[0]  # First var is concider as ouput 
    undeclared_params = list(set(Vars) ^ set(params))

    undeclared_params2 = []
    for a in range(len(undeclared_params)):
        if undeclared_params[a].isdigit() != True:
            undeclared_params2.append(undeclared_params[a])
    test = pd.Index(undeclared_params2).difference(df.columns).tolist()

    test2 =  list(set(undeclared_params2) & set(df.columns))
    return {'undeclared_params': test, 'params_in_data' : test2}


def write_header(output_file):
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        file.write("import tensorflow as tf")    
        file.write('\n')
        file.write("import pandas as pd")    
        file.write('\n')
        file.write("from bayesian.distribution import dist")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')
        
def write_priors(model, output_file):    
    p = [] # Store model parameters name
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'prior' in key.lower():
            p.append(var[0])
            with open(output_file,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + 
                           " = tfd.Sample(tfd." + var[1] + "(" + 
                            str(','.join(var[2])) + ")),")
                file.write('\n')
                
def write_main(model, output_file):    
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'main' in key.lower():
            with open(output_file,'a') as file:
                file.write('\t')
                
                
                file.write(str(var[0]) + " = lambda " + 
                            str(','.join(p)) + ":" +
                            " tfd.Independent(tfd."+ var[1] + "(" +
                            str(','.join(var[2]))+ "))"
                            )
                file.write('\n')
    with open(output_file,'a') as file:
        file.write('))')
               
def write_nonVar_model(model):
    output_file = 'output/model.py'
    write_header(output_file)
    write_priors(model, output_file)
    write_main(model, output_file)

def write_nonVar_withL(model):
    output_file = 'output/model.py'
    with open(output_file,'w') as file:
        pass
    with open(output_file,'w') as file:
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        file.write("import tensorflow as tf")    
        file.write('\n')
        file.write("import pandas as pd")    
        file.write('\n')
        file.write("from bayesian.distribution import dist")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        file.write("m = tfd.JointDistributionNamed(dict(")
        file.write('\n')

    p = [] # Store model parameters name
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'prior' in key.lower():
            p.append(var[0])
            with open(output_file,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + 
                        " = tfd.Sample(tfd." + var[1] + "(" + 
                            str(','.join(var[2])) + ")),")
                file.write('\n')
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'likelihood' in key.lower():
            if len(p) == 1: 
                param = str(p[0])
            else:
                param = str(','.join(p))   
        
            if len(var[1]) == 1:
                var2 = str(var[1][0])
            else:
                var2 = str(','.join(var[1]))
                
            with open(output_file,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + " = lambda " + 
                            str(param) + ":" +
                            " tfd.Independent(tfd."+ 
                            model["Main"]['model'][1] + "(" +
                            var2 + ")),"
                            )
                file.write('\n')
                
    with open(output_file,'a') as file:
        file.write('))')      
    
def write_model(model, path = None):
    print(model)
    model2 = 'output/model.py'
    with open(model2,'w') as file:
        pass

    with open(model2,'w') as file:
        file.write("import tensorflow_probability as tfp")    
        file.write('\n')
        file.write("import tensorflow as tf")    
        file.write('\n')
        file.write("import pandas as pd")    
        file.write('\n')
        file.write("from bayesian.distribution import dist")    
        file.write('\n')
        file.write("tfd = tfp.distributions")    
        file.write('\n')
        
        
    #if path is not None:
    #    with open(model2,'w') as file:
    #        print(path)
    #        print(model)
    #        file.write("data = pd.read_csv(")
    #        file.write(str(path))   
    #        file.write(", sep = ';')")
    #        file.write('\n')
    #        file.write("m = tfd.JointDistributionNamed(dict(")
    #        file.write('\n')

    # Write priors
    p = []
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'prior' in key.lower():
            p.append(var[0])
            with open(model2,'a') as file:
                file.write('\t')
                file.write(str(var[0]) + 
                           " = dist( **{'distribution':'" +
                           str(var[1]) +"', 'loc' : " + 
                           str(float(var[2][0])) + ", 'scale' : " +
                           str(float(var[2][1])) + "}),")
                file.write('\n')
    # Write likelihood
    for key in model.keys():
        tmp = model[key]
        input = model[key]['input']
        var = model[key]['model']
        if 'likelihood' in key.lower():
            with open(model2,'a') as file:
                file.write('\t')
                file.write(str(var[0]) +
                    " = lambda ")
            for a in range(len(p)):
                if a != (len(p)-1):
                    with open(model2,'a') as file:
                        file.write(str(p[a]) + ", ")
                else:
                     with open(model2,'a') as file:
                        file.write(str(p[a]) + ": tfd.Independent(\n \t\t tfd.Normal(")               
            with open(model2,'a') as file:
                file.write('\n')  
                file.write('\t\t\t\t' + "loc = " + input.split("=")[1] + ",")
                file.write('\n') 
                file.write('\t\t\t\t' + "scale = 0 \n \t\t),")
                file.write("reinterpreted_batch_ndims=1)")
                file.write('\n') 
                file.write('), validate_args=True)') 
    print("Model created")