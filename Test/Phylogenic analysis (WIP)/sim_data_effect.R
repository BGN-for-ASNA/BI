library(ape)
library(MASS)

set.seed(1234)
N_species <- 50

# 1. Simulate Tree
tree <- rcoal(N_species)
tree$tip.label <- paste0("sp_", 1:N_species)
write.nexus(tree, file = "phylo_slopes.nex")
A <- vcv.phylo(tree)
A <- A / max(A) # Standardize

# 2. Simulate Group-Level Effects (Varying Intercepts and Slopes)
# Covariance for intercept and slope (rho = 0.5)
sd_intercept <- 1.5
sd_slope <- 0.8
rho <- 0.5
Sigma <- matrix(c(sd_intercept^2, rho * sd_intercept * sd_slope,
                  rho * sd_intercept * sd_slope, sd_slope^2), 2, 2)

# Kronecker product A %x% Sigma
# But we can simulate U_species as (L_A %*% Z %*% t(L_Sigma))
L_A <- t(chol(A))
L_Sigma <- t(chol(Sigma))
Z <- matrix(rnorm(N_species * 2), N_species, 2)
U <- L_A %*% Z %*% t(L_Sigma)

u_intercept <- U[, 1]
u_slope <- U[, 2]

# 3. Simulate Data
b0 <- 2.0 # Population Intercept
b1 <- 1.2 # Population Slope
sigma <- 1.0 # Error SD

x <- rnorm(N_species, mean = 5, sd = 2) # Continuous predictor
mu <- (b0 + u_intercept) + (b1 + u_slope) * x
y <- rnorm(N_species, mean = mu, sd = sigma)

data_slopes <- data.frame(
  phylo = tree$tip.label,
  x = x,
  y = y
)

write.table(data_slopes, "data_slopes.txt", row.names = FALSE, quote = FALSE)

cat("Simulated data for Model 6 saved to data_slopes.txt and phylo_slopes.nex\n")
