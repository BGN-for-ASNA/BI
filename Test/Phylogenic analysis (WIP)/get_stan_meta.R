library(brms)
library(ape)

# Load data
phylo <- read.nexus("phylo.nex")
data_fisher <- read.table("data_effect.txt", header = TRUE)
data_fisher$obs <- 1:nrow(data_fisher)

# Calculate standard error
data_fisher$se <- sqrt(1 / (data_fisher$N - 3))

# Covariance matrix
A <- vcv.phylo(phylo)

# Formula
formula <- bf(Zr | se(se) ~ 1 + (1|gr(phylo, cov = A)) + (1|obs))

# Generate Stan code
stan_code <- make_stancode(
  formula,
  data = data_fisher,
  family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), "Intercept"),
    prior(student_t(3, 0, 10), "sd")
  )
)

# Save Stan code
writeLines(stan_code, "model_meta.stan")

# Save Cholesky of A for BI
L <- t(chol(A))
write.csv(L, "L_meta.csv", row.names=FALSE)

cat("Generated model_meta.stan and L_meta.csv\n")
