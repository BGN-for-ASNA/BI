# Primates example (Ringen 2026, §3.7) — Stan reference
# Empirical GDPM: 2 latent traits (life history, brain allometry) mapped by a
# factor matrix onto 4 gamma-distributed observed traits, with missing data.
# Data prep transcribed from R/prepare_primates_GDPM_standata.R.
suppressPackageStartupMessages({
  library(cmdstanr); library(posterior); library(jsonlite); library(dplyr)
})

REPO <- "/home/sebastian_sosa/phylo/GDPM_ms/ErikRingen-GDPM_ms-69c1d85"
OUT  <- "/home/sebastian_sosa/phylo/examples/primates"
set.seed(123)

d <- read.csv(file.path(REPO, "data/primates_data.csv"))
tree <- ape::read.tree(file.path(REPO, "data/primate_consensus_tree.tre"))

stopifnot(length(setdiff(d$taxon, tree$tip.label)) == 0)
d_matched <- d[match(tree$tip.label, d$taxon), ]
stopifnot(identical(d_matched$taxon, tree$tip.label))

# clade labels are only used for posterior predictive checks
d_matched <- left_join(d_matched, coevolve::primates$data,
                       by = c("taxon" = "species"))

body <- d_matched$body
brain <- d_matched$brain
longevity <- d_matched$max_longevity; longevity[is.na(longevity)] <- -99
maturity  <- d_matched$fem_maturity;  maturity[is.na(maturity)]  <- -99

# coevolve builds the tree segment structure; the latent columns are dummies
data_list <- coevolve::coev_make_standata(
  data = d_matched %>% mutate(life_history = rnorm(nrow(d_matched)),
                              diet = rnorm(nrow(d_matched)),
                              beta = rnorm(nrow(d_matched))),
  variables = list(life_history = "normal", beta = "normal"),
  id = "taxon", tree = tree
)

data_list$N_latent <- 2
data_list$y <- as.matrix(data.frame(body = body, brain = brain,
                                    longevity = longevity, maturity = maturity))
data_list$J <- ncol(data_list$y)
data_list$miss <- as.matrix(data_list$y == -99) * 1
data_list$y_mean <- apply(data_list$y, 2, function(x) mean(x[x != -99]))
data_list$clade <- d_matched$clade

# GDPM_primate.stan does not declare `clade`, and it is a character vector that
# write_stan_json cannot serialise, so drop it (it is kept in the matched csv
# below for the posterior predictive checks that use it)
standata <- data_list
standata$clade <- NULL
write_stan_json(standata, file.path(OUT, "data/standata.json"))
write.csv(d_matched, file.path(OUT, "data/primates_matched.csv"), row.names = FALSE)

cat("N_tips", standata$N_tips, "N_seg", standata$N_seg, "N_obs", standata$N_obs,
    "J", standata$J, "N_latent", standata$N_latent,
    "num_effects", standata$num_effects, "\n")
cat("missing cells per trait:", colSums(standata$miss), "\n")

file.copy(file.path(REPO, "stan/GDPM_primate.stan"),
          file.path(OUT, "stan/GDPM_primate.stan"), overwrite = TRUE)

mod <- cmdstan_model(file.path(OUT, "stan/GDPM_primate.stan"))
fit <- mod$sample(data = standata, seed = 42, chains = 4, parallel_chains = 4,
                  iter_warmup = 500, iter_sampling = 500, adapt_delta = 0.95,
                  refresh = 100)

pars <- c("A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma",
          "alpha", "shape", "lambda_free", "cor_R")
draws <- as_draws_df(fit$draws(variables = pars))
write.csv(draws, file.path(OUT, "results/stan_draws.csv"), row.names = FALSE)
write.csv(fit$summary(variables = pars), file.path(OUT, "results/stan_summary.csv"),
          row.names = FALSE)
diag <- fit$diagnostic_summary()
write_json(diag, file.path(OUT, "results/stan_diagnostics.json"), auto_unbox = TRUE)
cat("divergences:", sum(diag$num_divergent), " max_treedepth hits:",
    sum(diag$num_max_treedepth), "\n")
cat("time (s):", fit$time()$total, "\n")
