# Stan arm of the benchmark. Reports sampling wall clock plus the sampler
# diagnostics needed to compare like with like: leapfrog counts, treedepth
# saturation, E-BFMI, divergences, adapted step size.
suppressPackageStartupMessages({library(cmdstanr); library(posterior); library(jsonlite)})

a <- commandArgs(trailingOnly = TRUE)
example <- a[1]; seed <- as.integer(a[2])
warmup <- as.integer(a[3]); samples <- as.integer(a[4]); chains <- as.integer(a[5])

EX  <- file.path("/home/sebastian_sosa/phylo/examples", example)
OUT <- "/home/sebastian_sosa/phylo/examples/benchmark"

stan_file <- if (example == "cichlid") file.path(EX, "stan/cichlid_gdpm.stan") else
                                       file.path(EX, "stan/GDPM_primate.stan")
standata <- jsonlite::fromJSON(file.path(EX, "data/standata.json"))
# fromJSON collapses the leading tree dimension on these; restore it
for (nm in c("node_seq", "parent", "ts", "tip", "length_index", "tip_to_seg")) {
  if (!is.null(standata[[nm]]) && is.null(dim(standata[[nm]])))
    standata[[nm]] <- matrix(standata[[nm]], nrow = 1)
}
standata$effects_mat <- as.matrix(standata$effects_mat)
standata$y    <- as.matrix(standata$y)
standata$miss <- as.matrix(standata$miss)

mod <- cmdstan_model(stan_file)
fit <- mod$sample(data = standata, seed = seed, chains = chains,
                  parallel_chains = chains, iter_warmup = warmup,
                  iter_sampling = samples, adapt_delta = 0.95, refresh = 0,
                  show_messages = FALSE)

pars <- if (example == "cichlid")
  c("A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma") else
  c("A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma",
    "alpha", "shape", "lambda_free", "cor_R")

# chain-structured draws -> npz, so ESS is computed by the same estimator
# numpyro uses rather than by two different implementations
# Index the 3-d array explicitly: `as.matrix(d[, , v])` silently collapses the
# chain dimension (giving 1 x draws*chains), which would break both ESS and Rhat.
d <- posterior::as_draws_array(fit$draws(variables = pars))
n_iter <- dim(d)[1]; n_chain <- dim(d)[2]
arrs <- list()
for (v in dimnames(d)[[3]]) {
  m <- matrix(as.numeric(d[, , v]), nrow = n_iter, ncol = n_chain)  # iter x chain
  arrs[[v]] <- t(m)                                                # chain x iter
}
stopifnot(all(vapply(arrs, function(x) all(dim(x) == c(n_chain, n_iter)), TRUE)))
np <- reticulate::import("numpy", convert = FALSE)
# arrs is already (chains, draws); do not transpose again
do.call(np$savez, c(list(file.path(OUT, paste0(example, "_stan_s", seed, ".npz"))),
                    lapply(arrs, function(x) np$array(x))))

sp <- fit$sampler_diagnostics(format = "draws_df")
diag <- fit$diagnostic_summary()
tt <- fit$time()
info <- list(
  sampling_wall = max(tt$chains$total),          # slowest chain = wall clock
  total_leapfrog = sum(sp$n_leapfrog__),
  mean_leapfrog  = mean(sp$n_leapfrog__),
  pct_treedepth  = 100 * mean(sp$treedepth__ >= 10),
  divergences    = sum(diag$num_divergent),
  min_ebfmi      = min(diag$ebfmi),
  step_size      = mean(sp$stepsize__)
)
write_json(info, file.path(OUT, paste0(example, "_stan_s", seed, ".json")),
           auto_unbox = TRUE)
cat("stan done:", example, "seed", seed, "wall", round(info$sampling_wall, 1), "\n")
