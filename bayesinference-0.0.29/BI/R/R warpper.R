# This file contain the process to wrap the python code into R functions programatically
library(magrittr)
library(reticulate)
library(jsonlite)
reticulate::py_install('BayesInference')
inspect <- import("inspect")
jnp = import("jax.numpy")
dirout_path = "./R"
#setwd(dirout_path)
bi <- import("BI")
data <- fromJSON("Doc_R.json")

extract_values <- function(param) {
  # Convert to string and extract the text within quotes
  as.character(param) %>%
    sub(".*?\"(.*?)\".*", "\\1", .)
}


build_function = function(foo,
                          func_name,
                          name_file,
                          signature,
                          output_dir=dirout_path,
                          docString = ""){
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

    func_body <- paste0(func_name,
                        "=function(", paste(default_paramsR), ") { \n",
                        #"     bi=importbi(platform='cpu')\n",
                        "     shape=do.call(tuple, as.list(as.integer(shape)))\n",
                        "     seed=as.integer(seed);\n",
                        "     ", foo, "(",paste(default_paramsP), ")\n",
                        "}")

    func_body = gsub("loc: jax.Array", "loc=0.0", func_body)
    func_body = gsub("covariance_row:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_row=None,", func_body)
    func_body = gsub("covariance_row: jax.Array = covariance_row: jax.Array ,", "covariance_row = covariance_row,", func_body)
    func_body = gsub("covariance_rfft:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_rfft=None,", func_body)
    func_body = gsub(" covariance_rfft: jax.Array = covariance_rfft: jax.Array ,", "covariance_rfft=covariance_rfft,", func_body)

  }else{
    func_body <- paste0("function(", paste(default_paramsR), ") {",
                      "    bi$", foo, "(",
                      paste(default_paramsP), ")",
                      "}")
    func_body = gsub("loc: jax.Array", "loc=0.0", func_body)
    func_body = gsub("covariance_row:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_row=None,", func_body)
    func_body = gsub("covariance_rfft:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_rfft=None,", func_body)
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
  writeLines(doc, file_con)
  writeLines(func_body, file_con)
  close(file_con)
}


# Call distributions----------------------
attrs <- py_list_attributes(bi$dist)
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
attrs = attrs[30:length(attrs)]

for (a in attrs){
  if(!a %in% no){
    obj <- tryCatch(bi$dist[[a]], error = function(e) NULL)
    if (!is.null(obj)) {
      py_has_attr(bi$dist[[a]], "__call__")
      func_name = gsub("<function\\s+[\\w\\.]+\\.(\\w+)\\s+at\\s+0x[0-9A-Fa-f]+>", "bi.dist.\\1", as.character(bi$dist[[a]]), perl=TRUE)
      func_name2=paste(".",gsub("\\.", "\\$",func_name),sep='')
      func_name3=gsub("\\.", "",func_name)
      docName = gsub("bi.dist.", "", func_name)
      doc = data[[docName]]
      if (is.null(doc)){doc = ""}
      build_function(foo=func_name2,
                     name_file=a,
                     func_name = func_name,
                     signature = inspect$signature(bi$dist[[a]]),
                     docString = doc)
    } else {FALSE}
  }

}




