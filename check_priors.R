.libPaths(c("/home/sebastian_sosa/R/x86_64-pc-linux-gnu-library/4.6", "/home/sebastian_sosa/R/x86_64-pc-linux-gnu-library/4.3", .libPaths()))
library(bisonR)
priors <- get_default_priors("binary")
print("Binary priors:")
print(priors)

priors_duration <- get_default_priors("duration")
print("Duration priors:")
print(priors_duration)
