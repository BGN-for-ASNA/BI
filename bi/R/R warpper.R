
library(magrittr)
library(reticulate)
inspect <- import("inspect")
bi <- import("main")
foos = c('setup', 'data', 'OHE', 'index', 'scale', 'data_to_model', 
         'run',
         'summary', 
         'diag_prior_dist', 
         'diag_posterior', 
         'diag_traces',
         'diag_rank',
         'diag_forest',
         'diag_waic',
         'diag_compare',
         'diag_rhat',
         'diag_ess',
         'diag_pair',
         'diag_density',
         'diag_plot_ess',
         'model_checks')

extract_values <- function(param) {
  # Convert to string and extract the text within quotes
  as.character(param) %>%
    sub(".*?\"(.*?)\".*", "\\1", .)
}




  
build_function = function(foo,  
                          func_name,
                          name_file,
                          signature,
                          output_dir="G:/OneDrive/Travail/Max Planck/Projects/BI/bi/R"){
  # Create directory if it doesn't exist
  dir.create(output_dir, showWarnings = FALSE)

  params <- signature$parameters
  param_names <- reticulate:::py_dict_get_keys_as_str(params)
  named_list <- as.list(params$copy())
  extracted_terms <- lapply(named_list, extract_values)
  extracted_terms <- extracted_terms[!names(extracted_terms) %in% c("self")]
  extracted_terms <- extracted_terms[!names(extracted_terms) %in% c("args")]
  extracted_terms <- lapply(extracted_terms, function(x) gsub("\\(", "c(", gsub("\\]", ")", x)))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("\\[", "c(", gsub("\\]", ")", x)))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("None", "py_none()", x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("True", paste(T), x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("False", paste(F), x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("<function\\s+(\\w+)\\s+at\\s+0x[0-9A-Fa-f]+>", "'numpyro.\\1'", x))
  default_params <- paste(extracted_terms, collapse = ", ")  

  
  if('kwargs' %in% names(extracted_terms)){
    default_paramsR = gsub("\\*\\*kwargs", '...', default_params)
    default_paramsP = gsub("\\*\\*kwargs", 'list(...)', default_params)
    shape_inside=FALSE
    
    if(grepl('shape',default_params)){
      shape_inside=TRUE
    }    
    
  }else{
    shape_inside=FALSE
    if(grepl('shape',default_params)){
      shape_inside=TRUE
    }
    
    default_paramsR = default_params
    tmp=strsplit(default_params,',')
    for (a in 1:length(tmp[[1]])) {
      tmp[[1]][a]=gsub("([^=]+)=.*", "\\1=\\1", tmp[[1]][a])
    }
    tmp=paste(unlist(tmp),collapse = ', ')

    default_paramsP = tmp
  }
  
  # Generate the new R function dynamically
  if(shape_inside){
    func_body <- paste0("function(", paste(default_paramsR), ") {",
                        " shape=do.call(tuple, as.list(as.integer(shape)))",
                        "    bi$", foo, "(", 
                        paste(default_paramsP), ")",
                        "}")   
  }else{
    func_body <- paste0("function(", paste(default_paramsR), ") {",
                      "    bi$", foo, "(", 
                      paste(default_paramsP), ")",
                      "}")
  }

  
  # Assign the function as before
  eval(parse(text = func_body))
  assign(func_name, eval(parse(text = func_body)))
  
  # Write the function to a file in the specified directory
  # Construct the full file path
  file_path <- file.path(output_dir, paste0(name_file, ".R"))
  print(file_path)

  # Write the function to the file
  file_con <- file(file_path, "w")
  writeLines(func_body, file_con)
  close(file_con)
}

for( a in 1: length(foos)){
  build_function(foo=foos[a],
                 name_file=foos[a],
                 func_name =paste0('bi.', gsub('_', '.', foos[a])),
                 signature = inspect$signature(bi$bi[foos[a]]))
}

# Call distributions----------------------
attrs <- py_list_attributes(bi$bi$dist)
no=c("__class__",
     "__delattr__",
     "__dict__",
     "__dir__",
     "__doc__",
     "__eq__",
     "__format__",
     "__ge__",
     "__getattribute__",
     "__getstate__",
     "__gt__",
     "__hash__",
     "__init__",
     "__init_subclass__",
     "__le__",
     "__lt__",
     "__module__",
     "__ne__",
     "__new__",
     "__reduce__",
     "__reduce_ex__",
     "__repr__",
     "__setattr__",
     "__sizeof__",
     "__str__",
     "__subclasshook__",
     "__weakref__",
     "sineskewed")

for (a in attrs){
  if(!a %in% no){
    obj <- tryCatch(bi$bi$dist[[a]], error = function(e) NULL)
    if (!is.null(obj)) {
      py_has_attr(bi$bi$dist[[a]], "__call__")
      func_name = gsub("<function\\s+[\\w\\.]+\\.(\\w+)\\s+at\\s+0x[0-9A-Fa-f]+>", "bi.dist.\\1", as.character(bi$bi$dist[[a]]), perl=TRUE)
      func_name2=gsub("\\.", "\\$",func_name)
      func_name3=gsub("\\.", "",func_name)
      build_function(foo=func_name2,  
                     name_file=a,
                     func_name = func_name,
                     signature = inspect$signature(bi$bi$dist[[a]]))
    } else {
      FALSE
    }    
  }

}

