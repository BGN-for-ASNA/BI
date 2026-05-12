library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

cat("Testing m$link$inv_logit...\n")
x <- jnp$array(c(-1.0, 0.0, 1.0))
tryCatch({
  res <- m$link$inv_logit(x)
  cat("Success! Results:", as.numeric(res), "\n")
}, error = function(e) {
  cat("m$link$inv_logit FAILED:", conditionMessage(e), "\n")
})

cat("\nTesting jax.nn.sigmoid...\n")
jax_nn <- import("jax.nn")
tryCatch({
  res <- jax_nn$sigmoid(x)
  cat("Success! Results:", as.numeric(res), "\n")
}, error = function(e) {
  cat("jax.nn.sigmoid FAILED:", conditionMessage(e), "\n")
})
