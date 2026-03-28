library(brms)
library(ape)

# Load data
phylo <- read.nexus("phylo.nex")
data_repeat <- read.table("data_repeat.txt", header = TRUE)

# Calculate species-level mean of cofactor
data_repeat$spec_mean_cf <- with(data_repeat, tapply(cofactor, species, mean)[species])

# Covariance matrix
A <- vcv.phylo(phylo)

# Formula
# (1 | species) is a non-phylogenetic varying intercept
# (1 | gr(phylo, cov = A)) is the phylogenetic varying intercept
formula <- bf(phen ~ spec_mean_cf + (1 | gr(phylo, cov = A)) + (1 | species))

# Generate Stan code
stan_code <- make_stancode(
  formula,
  data = data_repeat,
  family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0,10), "b"),
    prior(normal(0,50), "Intercept"),
    prior(student_t(3,0,20), "sd"),
    prior(student_t(3,0,20), "sigma")
  )
)

# Save Stan code
writeLines(stan_code, "model_repeat.stan")

# Generate Stan data to inspect mapping if needed
stan_data <- make_standata(
  formula,
  data = data_repeat,
  family = gaussian(),
  data2 = list(A = A)
)

# Print a summary of standata for reference
cat("Stan data names:\n")
print(names(stan_data))
cat("\nN:", stan_data$N)
cat("\nN_1:", stan_data$N_1)
cat("\nN_2:", stan_data$N_2)
cat("\nM_1:", stan_data$M_1)
cat("\nM_2:", stan_data$M_2)
