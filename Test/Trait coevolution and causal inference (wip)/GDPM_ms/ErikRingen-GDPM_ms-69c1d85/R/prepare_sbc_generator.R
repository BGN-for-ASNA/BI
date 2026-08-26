prepare_sbc_generator <- function(sbc_model) {
  # prepare generator
  SBC::SBC_generator_function(
    f = sbc_generator_function,
    model = sbc_model,
    N = 20
  )
}
