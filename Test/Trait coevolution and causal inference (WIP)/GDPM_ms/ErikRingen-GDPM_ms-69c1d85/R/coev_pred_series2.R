# simplified version of the coevolve function, useable with model for primate model not fit by the package
coev_pred_series2 <- function (fit, var_names, eta_anc = NULL, intervention_values = NULL, stochastic = F, ntimes = 30, tmax = 1) {
  post <- extract_samples2(fit)
  J <- length(var_names)
  nsamps <- length(post$lp__)
  preds <- array(NA, dim = c(nsamps, ntimes + 1, J), dimnames = list(samps = 1:nsamps, 
                                                                     time = 1:(ntimes + 1), response = var_names))
  if (!is.null(intervention_values)) {
    x_hat <- unlist(intervention_values[var_names])
    held_indices <- which(!is.na(x_hat))
    free_indices <- which(is.na(x_hat))
    if (length(free_indices) == 0) {
      stop2(paste0("At least one variable must be declared as a free variable ", 
                   "(with NA in 'intervention_values')."))
    }
  }
  else {
    x_hat <- rep(NA, J)
    held_indices <- integer(0)
    free_indices <- 1:J
  }
  if (is.null(intervention_values)) {
    initial_values <- rep(NA, J)
  }
  else {
    initial_values <- unlist(intervention_values[var_names])
  }
  if (!is.null(eta_anc)) {
    eta_anc <- unlist(eta_anc[var_names])
    conflicting_vars <- intersect(names(eta_anc)[!is.na(eta_anc)], 
                                  names(x_hat)[!is.na(x_hat)])
    if (length(conflicting_vars) > 0) {
      message(paste0("Note: For variable(s) ", paste(conflicting_vars, 
                                                     collapse = ", "), ", both 'eta_anc' and 'intervention_values' specify non-NA values. ", 
                     "The 'intervention_values' will take precedence for these ", 
                     "variable(s)."))
      eta_anc[conflicting_vars] <- x_hat[conflicting_vars]
    }
    initial_values <- eta_anc
  }
  if (any(is.na(initial_values))) {
    eta_anc_long <- post$eta_anc
    ntrees <- dim(eta_anc_long)[2]
    eta_anc_long <- eta_anc_long[, sample(1:ntrees, size = ntrees, 
                                          replace = FALSE), ]
    if (ntrees > 1) {
      eta_anc_long2 <- eta_anc_long[, 1, ]
      for (t in 2:ntrees) {
        eta_anc_long2 <- rbind(eta_anc_long2, eta_anc_long[, 
                                                           t, ])
      }
      eta_anc_long <- eta_anc_long2
    }
  }
  for (j in 1:J) {
    for (i in 1:nsamps) {
      if (is.na(initial_values[j])) {
        preds[i, 1, j] <- eta_anc_long[i, j]
      }
      else {
        preds[i, 1, j] <- initial_values[j]
      }
    }
  }
  for (i in 1:nsamps) {
    A <- post$A[i, , ]
    b <- post$b[i, ]
    A_free_free <- A[free_indices, free_indices, drop = FALSE]
    if (length(held_indices) > 0) {
      A_free_held <- A[free_indices, held_indices, drop = FALSE]
    }
    else {
      A_free_held <- matrix(0, nrow = length(free_indices), 
                            ncol = 0)
    }
    b_free <- b[free_indices]
    if (length(held_indices) > 0) {
      c <- A_free_held %*% x_hat[held_indices] + b_free
    }
    else {
      c <- b_free
    }
    A_delta_free_free <- as.matrix(Matrix::expm(A_free_free * 
                                                  tmax/ntimes))
    if (nrow(A_free_free) != ncol(A_free_free)) {
      stop2("Matrix A_free_free must be square.")
    }
    inv_A_free_free <- tryCatch(solve(A_free_free), error = function(e) {
      stop2("Matrix A_free_free is singular and cannot be inverted.")
    })
    I_free_free <- diag(rep(1, length(free_indices)))
    if (stochastic == TRUE) {
      Q_inf <- post$Q_inf[i, , ]
      VCV <- Q_inf - ((A_delta_free_free) %*% Q_inf %*% 
                        t(A_delta_free_free))
      chol_VCV <- t(chol(Matrix::nearPD(VCV)$mat))
    }
    preds_free <- preds[i, 1, free_indices]
    for (t in 1:ntimes) {
      preds_free <- (A_delta_free_free %*% preds_free + 
                       (inv_A_free_free %*% (A_delta_free_free - I_free_free) %*% 
                          c))[, 1]
      if (stochastic == TRUE) {
        preds_free <- preds_free + (chol_VCV %*% stats::rnorm(length(free_indices), 
                                                              0, 1))
      }
      preds[i, t + 1, free_indices] <- preds_free
      if (length(held_indices) > 0) {
        preds[i, t + 1, held_indices] <- x_hat[held_indices]
      }
    }
  }
  return(preds)
}
