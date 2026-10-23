# %%
library(brms)
library(ape)

# Load data
phylo <- read.nexus("phylo.nex")
data_simple <- read.table("data_simple.txt", header = TRUE)

# Prepare covariance matrix
A <- vcv.phylo(phylo)

# Fit model
model_simple <- brm(
  phen ~ cofactor + (1 | gr(phylo, cov = A)),
  data = data_simple,
  family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), "b"),
    prior(normal(0, 50), "Intercept"),
    prior(student_t(3, 0, 20), "sd"),
    prior(student_t(3, 0, 20), "sigma")
  ),
  chains = 2, iter = 2000, warmup = 1000, cores = 2
)

# Save summary
sum_simple <- summary(model_simple)
print(sum_simple)

# Save posteriors for comparison
post_simple <- as.data.frame(model_simple)
write.csv(post_simple, "brms_post_simple.csv", row.names = FALSE)
saveRDS(model_simple, "model_simple.rds")

# %%
