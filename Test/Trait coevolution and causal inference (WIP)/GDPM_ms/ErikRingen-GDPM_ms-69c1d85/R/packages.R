options(tidyverse.quiet = TRUE)

library(targets)
library(tidyverse)
library(tarchetypes)

tar_option_set(
  packages = c(
    "ape","cmdstanr","coevolve","knitr","posterior","SBC","tidyverse", "patchwork", "ggdist", "grid", "tinytable", "bayesplot", "phytools", "future", "DiagrammeR", "DiagrammeRsvg"
  )
)