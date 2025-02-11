bi.setup=function(platform='cpu', cores=py_none(), deallocate=FALSE) {   
  bi <- import("main")
  bi$setup(platform=platform,  cores= cores,  deallocate= deallocate)
  return(bi)
  }
