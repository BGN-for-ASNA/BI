#!/usr/bin/env Rscript
# Runs STRAND simulation + numpyro fit, saves arrays for Python/BI comparison

library(posterior)
library(igraph)
library(STRAND)

set.seed(1)

N_id     <- 50
N_layers <- 7

# Covariates
Kinship   <- standardize_strand(rlkjcorr(1, N_id, eta=1.5))
Dominance <- ceiling(rlkjcorr(1, N_id, eta=1.5) - 0.1)
Mass      <- rbern(N_id, 0.4)
Age       <- rnorm(N_id, 0, 1)
Strength  <- rnorm(N_id, 0, 1)

dyadic_preds      <- array(NA, c(N_id, N_id, 2))
dyadic_preds[,,1] <- Kinship
dyadic_preds[,,2] <- Dominance

sr_mu    <- rep(0, N_layers*2)
sr_sigma <- c(0.2, 0.7, 1.1, 0.3, 0.7, 1.8, 1.5,
              0.5, 1.7, 0.7, 0.9, 0.7, 0.8, 1.5)
sr_Rho   <- rlkjcorr(1, N_layers*2, eta=1.5)
dr_mu    <- rep(0, N_layers)
dr_sigma <- c(0.9, 1.1, 1.2, 0.5, 1.6, 2.1, 2.9)
error_sigma <- c(0.35, 2.1, 1.2, 0.2, 0.9, 1.5, 2.5)

# Use rlkjcorr directly — skip Stan-based generate_structured_correlation_matrix
repeat {
  dr_Rho_raw <- rlkjcorr(1, N_layers*2, eta=1)
  dr_Rho <- if (length(dim(dr_Rho_raw)) == 3) dr_Rho_raw[1,,] else as.matrix(dr_Rho_raw)
  if (inherits(try(chol(dr_Rho), silent=TRUE), "matrix")) break
}
cat("dr_Rho is positive definite, dim:", dim(dr_Rho), "\n")

sr_1 <- matrix(NA,2,3); sr_1[1,]=c(-0.5,1.0,-0.7); sr_1[2,]=c(0.7,-1.1,-1)
sr_2 <- matrix(NA,2,3); sr_2[1,]=c(-0.1,-1.0,0.7); sr_2[2,]=c(-0.7,-0.6,-0.01)
sr_3 <- matrix(NA,2,3); sr_3[1,]=c(0.1,0.3,-0.5);  sr_3[2,]=c(0.1,0.4,0)
sr_4 <- matrix(NA,2,3); sr_4[1,]=c(0.7,1.3,-1.5);  sr_4[2,]=c(0.2,0.4,1.0)
sr_5 <- matrix(NA,2,3); sr_5[1,]=c(0.1,-0.3,0.5);  sr_5[2,]=c(-0.6,2.4,-1.3)
sr_6 <- matrix(NA,2,3); sr_6[1,]=c(1.1,0.8,-2.5);  sr_6[2,]=c(2.1,0.8,0)
sr_7 <- matrix(NA,2,3); sr_7[1,]=c(1.1,0,-0.9);    sr_7[2,]=c(1.1,0,0.9)
sr_effects <- list(sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7)

dr_effects <- list(c(0.6,0.3), c(-0.2,-0.7), c(-1.1,1.7),
                   c(1.2,-0.3), c(0.2,0.7), c(-1.2,-1.7), c(0.2,-0.7))

group_probs_block_size <- c(0.25, c(0.25,0.75)*(1-0.25))
groups_1 <- rep("Any", N_id)
groups_2 <- sample(c("Mottled","Striped","Spotted"), N_id, replace=TRUE,
                   prob=group_probs_block_size)
groups_3 <- sample(c("Male","Female"), N_id, replace=TRUE, prob=c(0.5,0.5))

B_1a <- matrix(-0.7,1,1); B_2a <- matrix(0.5,1,1);  B_3a <- matrix(-3.5,1,1)
B_4a <- matrix(-2.5,1,1); B_5a <- matrix(-1.5,1,1); B_6a <- matrix(-3.5,1,1)
B_7a <- matrix(-2.5,1,1)

set.seed(1)
B_1b <- matrix(rnorm(9,0,1),3,3);   B_2b <- matrix(rnorm(9,-2,1),3,3)
B_3b <- matrix(rnorm(9,0.5,1),3,3); B_4b <- matrix(rnorm(9,0.5,1),3,3)
B_5b <- matrix(rnorm(9,0.5,1),3,3); B_6b <- matrix(rnorm(9,0.5,1),3,3)
B_7b <- matrix(rnorm(9,0.5,1),3,3)
diag(B_2b) <- diag(B_2b) + 1.5

B_1c <- matrix(rnorm(4,0,1),2,2);  B_2c <- matrix(rnorm(4,-3,1),2,2)
B_3c <- matrix(rnorm(4,-2,1),2,2); B_4c <- matrix(rnorm(4,-2,1),2,2)
B_5c <- matrix(rnorm(4,-2,1),2,2); B_6c <- matrix(rnorm(4,-2,1),2,2)
B_7c <- matrix(rnorm(4,-2,1),2,2)
diag(B_1c) <- diag(B_1c) - 0.5
diag(B_2c) <- diag(B_2c) + 1

B <- list(
  list(B_1a,B_1b,B_1c), list(B_2a,B_2b,B_2c), list(B_3a,B_3b,B_3c),
  list(B_4a,B_4b,B_4c), list(B_5a,B_5b,B_5c), list(B_6a,B_6b,B_6c),
  list(B_7a,B_7b,B_7c))

groups   <- data.frame(Intercept=as.numeric(factor(groups_1)),
                       Pattern=as.numeric(factor(groups_2)),
                       Sex=as.numeric(factor(groups_3)))
groups_f <- data.frame(Pattern=factor(groups_2), Sex=factor(groups_3))
indiv    <- data.frame(Mass=Mass, Age=Age, Strength=Strength)
dyad_list <- list(Kinship=Kinship, Dominance=Dominance)

labels <- paste("Ind", 1:N_id)

G <- simulate_multiplex_network(
  N_id=N_id, N_layers=N_layers, B=B, V=3, groups=groups,
  sr_mu=sr_mu, sr_sigma=sr_sigma, sr_Rho=sr_Rho,
  dr_mu=dr_mu, dr_sigma=dr_sigma, dr_Rho=dr_Rho,
  outcome_mode="gaussian", link_mode="identity",
  individual_predictors=indiv, dyadic_predictors=dyadic_preds,
  individual_effects=sr_effects, dyadic_effects=dr_effects,
  error_sigma=error_sigma)

outcome <- list(
  Feeding=G$network[1,,], Fighting=G$network[2,,], Grooming=G$network[3,,],
  Hunting=G$network[4,,], Smilling=G$network[5,,], Barking=G$network[6,,],
  Killing=G$network[7,,])

for (nm in names(outcome)) {
  rownames(outcome[[nm]]) <- colnames(outcome[[nm]]) <- labels
}
rownames(dyad_list$Kinship)   <- colnames(dyad_list$Kinship)   <- labels
rownames(dyad_list$Dominance) <- colnames(dyad_list$Dominance) <- labels
rownames(indiv)    <- labels
rownames(groups_f) <- labels

dat <- make_strand_data(outcome=outcome, block_covariates=groups_f,
                        individual_covariates=indiv, dyadic_covariates=dyad_list,
                        outcome_mode="gaussian", link_mode="identity",
                        multiplex=TRUE)

# STRAND numpyro fit (ground truth)
fit_numpyro <- fit_multiplex_model(
  data=dat,
  block_regression = ~ Pattern + Sex,
  focal_regression = ~ Mass + Age + Strength,
  target_regression = ~ Mass + Age + Strength,
  dyad_regression   = ~ Kinship + Dominance,
  mode="numpyro",
  mcmc_parameters=list(chains=1, parallel_chains=1, refresh=1, seed=42,
                       iter_warmup=2000, iter_sampling=2000, init=0.25,
                       max_treedepth=12, adapt_delta=0.95, cores=4,
                       chain_method="vectorized"))

# Extract posterior samples from numpyro fit
strand_np_samples <- fit_numpyro$fit$get_samples()

# Extract internal data arrays used by numpyro_multiplex
d      <- fit_numpyro$data
N_id_d <- d$N_id
N_lay  <- d$N_responses
locs   <- make_dyadic_edgelist(N_id_d)
N_dyads_r <- nrow(locs)

long_focal_set  <- focal_set_to_long(as.matrix(d$focal_set))
long_target_set <- target_set_to_long(as.matrix(d$target_set))
long_dyad_set   <- dyadic_set_to_long(d$dyad_set)

block_dat      <- block_set_to_dyadic_block_set(d$block_set, priors=d$priors)
long_block_set <- dyadic_set_to_long(block_dat$Y)
block_mu       <- block_dat$Mu
block_sigma    <- block_dat$Sigma

# Multiplex outcomes: (N_layers, N_dyads, 2)
# Use G$network directly (N_layers x N_id x N_id), already has correct indices
locs_cols <- colnames(locs)
cat("locs columns:", paste(locs_cols, collapse=", "), "\n")
cat("locs head:\n"); print(head(locs))

# Determine column names for i/j indices
col_i <- locs_cols[1]; col_j <- locs_cols[2]
cat("Using locs cols: i=", col_i, " j=", col_j, "\n")

long_outcome <- array(NA, c(N_lay, N_dyads_r, 2))
for (l in 1:N_lay) {
  om <- G$network[l,,]  # N_id x N_id matrix
  # locs[col_j, col_i] = from j to i = "i receives from j"
  idx_mat  <- cbind(locs[[col_j]], locs[[col_i]])  # (row, col) = (j, i) -> om[j,i]
  idx_mat2 <- cbind(locs[[col_i]], locs[[col_j]])  # (row, col) = (i, j) -> om[i,j]
  long_outcome[l,,1] <- om[idx_mat]
  long_outcome[l,,2] <- om[idx_mat2]
}

# Extract bandage bindings from data object (build them if needed)
dat2 <- build_multiplex_bindings_dr_multiplex(dat)
dr_bind_out1 <- dat2$numpyro_dr_bindings_out_1
dr_bind_out2 <- dat2$numpyro_dr_bindings_out_2
dr_bind_in1  <- dat2$numpyro_dr_bindings_in_1
dr_bind_in2  <- dat2$numpyro_dr_bindings_in_2
bandage_penalty <- if (!is.null(d$bandage_penalty)) d$bandage_penalty else 0.01

# Save STRAND posterior samples as .npy files (bypass reticulate serialization issues)
np <- reticulate::import("numpy")
sample_names <- names(strand_np_samples)
cat("Saving STRAND posteriors:", paste(sample_names, collapse=", "), "\n")
for (nm in sample_names) {
  np$save(paste0("strand_post_", nm, ".npy"), strand_np_samples[[nm]])
}
cat("Saved", length(sample_names), "posterior arrays as .npy files\n")

# Save all other arrays
save(long_focal_set, long_target_set, long_dyad_set, long_block_set,
     block_mu, block_sigma, long_outcome, locs,
     dr_bind_out1, dr_bind_out2, dr_bind_in1, dr_bind_in2,
     bandage_penalty, N_id_d, N_lay,
     file='multiplex_arrays.RData')

cat("strand_sim.R complete. Arrays saved to multiplex_arrays.RData\n")
