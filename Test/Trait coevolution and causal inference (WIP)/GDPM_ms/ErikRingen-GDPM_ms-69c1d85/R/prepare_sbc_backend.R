prepare_sbc_backend <- function(sbc_model) {
  # prepare backend
  SBC::SBC_backend_cmdstan_sample(
    model = sbc_model,
    iter_warmup = 250,
    iter_sampling = 1000,
    chains = 2,
    adapt_delta = 0.95
  )
}
