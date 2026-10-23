library(brms)
library(ape)

# Load data
data_slopes <- read.table("data_slopes.txt", header = TRUE)
phylo <- read.nexus("phylo_slopes.nex")
A <- vcv.phylo(phylo)

# Define formula
# Using (1 + x | gr(phylo, cov = A)) for varying intercept and slope
formula <- bf(y ~ x + (1 + x | gr(phylo, cov = A)))

# Get Stan code
cat("Generating Stan code for Model 6...\n")
stancode <- make_stancode(
  formula = formula,
  data = data_slopes,
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), class = Intercept),
    prior(normal(0, 10), class = b),
    prior(student_t(3, 0, 10), class = sd),
    prior(lkj(2), class = cor)
  )
)

writeLines(stancode, "model_slopes.stan")
cat("Stan code saved to model_slopes.stan\n")
