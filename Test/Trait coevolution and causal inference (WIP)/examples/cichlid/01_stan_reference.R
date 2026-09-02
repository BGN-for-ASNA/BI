# Cichlid example (Ringen 2026, §3.6) — Stan reference
# DGP = the paper's own generator: stan/synthetic_model.stan run prior-only with
# A, Q, b fixed to known truth. Fitting model = coevolve-generated GDPM.
suppressPackageStartupMessages({
  library(cmdstanr); library(posterior); library(jsonlite)
})

REPO <- "/home/sebastian_sosa/phylo/GDPM_ms/ErikRingen-GDPM_ms-69c1d85"
OUT  <- "/home/sebastian_sosa/phylo/examples/cichlid"
set.seed(123)

tree <- phytools::readNexus(file.path(REPO, "data/05_BEAST_RAxML.tre"), format = "raxml")
variables <- c("Promiscuity", "SpermSize", "Predation")
N <- length(tree$tip.label)

# effects_mat as in R/cichlid_sim_data.R: SpermSize does not cause Promiscuity or Predation
effects_mat <- matrix(TRUE, 3, 3, dimnames = list(variables, variables))
effects_mat[1, 2] <- FALSE
effects_mat[3, 2] <- FALSE

d <- data.frame(Promiscuity = rnorm(N), SpermSize = rnorm(N), Predation = rnorm(N),
                species = tree$tip.label)
vars <- list(Promiscuity = "normal", SpermSize = "normal", Predation = "normal")

# ---- 1. simulate from the paper's generative model (prior_only, fixed truth) ----
gen_dat <- coevolve::coev_make_standata(
  data = d, variables = vars, effects_mat = effects_mat,
  estimate_correlated_drift = FALSE, id = "species", tree = tree, prior_only = TRUE
)
gen <- cmdstan_model(file.path(REPO, "stan/synthetic_model.stan"))
sim <- gen$sample(data = gen_dat, chains = 1, seed = 123, refresh = 0,
                  iter_warmup = 50, iter_sampling = 1)
yrep <- posterior::draws_of(as_draws_rvars(sim)$yrep)
d_sim <- data.frame(Promiscuity = yrep[1, 1, 1:N, 1],
                    SpermSize   = yrep[1, 1, 1:N, 2],
                    Predation   = yrep[1, 1, 1:N, 3],
                    species     = tree$tip.label)
write.csv(d_sim, file.path(OUT, "data/d_sim.csv"), row.names = FALSE)

# true parameter values, hardcoded in stan/synthetic_model.stan
# A[2,1]=3, A[1,3]=-2, A[2,3]=-2, A[3,1]=1.5, diagonal -0.5 (Stan is 1-based [row,col])
truth <- list(
  A = matrix(c(-0.5, 0, -2,   3, -0.5, -2,   1.5, 0, -0.5), 3, 3, byrow = TRUE),
  Q = diag(2.0, 3), b = rep(0, 3), eta_anc = rep(0, 3)
)
write_json(truth, file.path(OUT, "data/true_params.json"), auto_unbox = TRUE, digits = 10)

# ---- 2. fitting model: coevolve-generated Stan, priors as in R/cichlid_fit.R ----
prior <- list(A_offdiag = "normal(0, 2)", Q_sigma = "normal(0, 2)")
stancode <- coevolve::coev_make_stancode(
  data = d_sim, variables = vars, prior = prior, effects_mat = effects_mat,
  scale = FALSE, estimate_correlated_drift = FALSE, id = "species", tree = tree)
writeLines(as.character(stancode), file.path(OUT, "stan/cichlid_gdpm.stan"))

standata <- coevolve::coev_make_standata(
  data = d_sim, variables = vars, prior = prior, effects_mat = effects_mat,
  scale = FALSE, estimate_correlated_drift = FALSE, id = "species", tree = tree)
write_stan_json(standata, file.path(OUT, "data/standata.json"))

cat("N_tips", standata$N_tips, "N_seg", standata$N_seg, "N_obs", standata$N_obs,
    "J", standata$J, "num_effects", standata$num_effects, "\n")

# ---- 3. fit ----
mod <- cmdstan_model(file.path(OUT, "stan/cichlid_gdpm.stan"))
fit <- mod$sample(data = standata, seed = 42, chains = 4, parallel_chains = 4,
                  iter_warmup = 500, iter_sampling = 500, adapt_delta = 0.95,
                  refresh = 100)

pars <- c("A", "Q", "b", "eta_anc", "Q_sigma", "A_offdiag", "A_diag")
draws <- as_draws_df(fit$draws(variables = pars))
write.csv(draws, file.path(OUT, "results/stan_draws.csv"), row.names = FALSE)
write.csv(fit$summary(variables = pars), file.path(OUT, "results/stan_summary.csv"),
          row.names = FALSE)
diag <- fit$diagnostic_summary()
write_json(diag, file.path(OUT, "results/stan_diagnostics.json"), auto_unbox = TRUE)
cat("divergences:", sum(diag$num_divergent), " max_treedepth hits:",
    sum(diag$num_max_treedepth), "\n")
cat("time (s):", fit$time()$total, "\n")
