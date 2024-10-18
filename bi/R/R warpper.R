library(magrittr)
library(reticulate)
inspect <- import("inspect")
bi <- import("main")


foos = c('setup', 'data', 'OHE', 'index', 'scale', 'data_to_model', 
         'run',
         'summary', 
         'model_output_to_df', 
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
  
for( a in 1: length(foos)){
  func_name <- paste0('bi.', gsub('_', '.', foos[a]))
  signature <- inspect$signature(bi$bi[foos[a]])
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
  extracted_terms <- lapply(extracted_terms, function(x) gsub("<function init_to_uniform at 0x0000012F7E192160>", 'numpyro.init_to_mean()', x))
  default_params <- paste(extracted_terms, collapse = ", ")  
  if('kwargs' %in% names(extracted_terms)){
    default_paramsR = gsub("\\*\\*kwargs", '...', default_params)
    default_paramsP = gsub("\\*\\*kwargs", 'list(...)', default_params)
  }else{
    default_paramsR = default_paramsP = default_params
  }
    
    
    
  # Generate the new R function dynamically
  func_body <- paste0("function(", paste(default_paramsR), ") {",
                        "    reticulate::py_call(bi$bi$", foos[a], ", list(", 
                        paste(default_paramsP), "));",
                        "}")

  
  eval(parse(text = func_body))
  assign(func_name, eval(parse(text = func_body)))
}
