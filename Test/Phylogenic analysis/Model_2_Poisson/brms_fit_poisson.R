# %%
library(brms)
library(ape)

# Load data
phylo <- read.nexus("phylo.nex")
data_pois <- read.table(
    "https://paul-buerkner.github.io/data/data_pois.txt",
    header = TRUE
)
data_pois$obs <- 1:nrow(data_pois)
head(data_pois)

# Prepare covariance matrix
A <- vcv.phylo(phylo)

# Fit model
model_pois <- brm(
    phen_pois ~ cofactor + (1 | gr(phylo, cov = A)) + (1 | obs),
    data = data_pois, family = poisson("log"),
    data2 = list(A = A),
    chains = 2, cores = 2, iter = 4000,
    control = list(adapt_delta = 0.95)
)

# Save summary
sum_pois <- summary(model_pois)
print(sum_pois)

# Save posteriors for comparison
post_pois <- as.data.frame(model_pois)
write.csv(post_pois, "brms_post_pois.csv", row.names = FALSE)
saveRDS(model_pois, "model_pois.rds")

# %%
