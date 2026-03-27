library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

model <- function() {
  # This works?
  x <- m$dist$left_truncated_distribution(
    m$dist$student_t(3.0, 0.0, 1.0, create_obj = TRUE),
    low = 0.0, name = "x"
  )
  print(x)
}

tryCatch({
  m$fit(model, num_samples = 10L, num_warmup = 10L)
}, error = function(e) {
  print(py_last_error())
})
