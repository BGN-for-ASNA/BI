functions {
  #include util.stanfunctions
  #include cjs.stanfunctions
}

data {
  int<lower=1> I, J;  // number of individuals and surveys
  int<lower=2> S;  // number of alive states
  vector<lower=0>[J - 1] tau;  // survey intervals
  array[I, J] int<lower=0, upper=S> y;  // detection history
  int<lower=0, upper=1> ind;  // survey (0) or individual-level (1) parameters
  int<lower=0> grainsize;  // threading
}

transformed data {
  int Jm1 = J - 1, Sm1 = S - 1;
  array[I, 2] int f_l = first_last(y);
  array[I] int seq = linspaced_int_array(I, 1, I);
  vector[Jm1] tau_scl = tau / exp(mean(log(tau)));
}

parameters {
  vector<lower=0>[S] h;  // mortality hazard rates
  row_vector<lower=0>[S * Sm1] q;  // transition rates
  matrix<lower=0, upper=1>[S, Jm1] p;  // detection probabilities
}

transformed parameters {
  // priors
  real lprior = gamma_lpdf(h | 1, 3) + gamma_lpdf(q | 1, 3)
                + beta_lpdf(to_vector(p) | 1, 1);
}

model {
  target += lprior;
  
  // log TPMs and detection logits
  matrix[S, S] Q = rate_matrix(h, q)[:S, :S];
  array[Jm1] matrix[S, S] log_H_j;
  for (j in 1:Jm1) {
    log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
  }
  matrix[S, Jm1] logit_p_j = logit(p);
  
  // likelihood with individual or survey-level parameters
  if (ind) {
    array[I, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, I);
    array[I] matrix[S, Jm1] logit_p_i = rep_array(logit_p_j, I);
    target += grainsize ?
              reduce_sum(partial_cjs_ms, seq, grainsize, y, f_l, log_H_i, 
                         logit_p_i)
              : sum(cjs_ms(y, f_l, log_H_i, logit_p_i));
  } else {
    target += cjs_ms(y, f_l, log_H_j, logit_p_j);
  }
}

generated quantities {
  vector[I] log_lik;
  {
    matrix[S, S] Q = rate_matrix(h, q)[:S, :S];
    array[Jm1] matrix[S, S] log_H_j;
    for (j in 1:Jm1) {
      log_H_j[j] = log(matrix_exp(Q * tau_scl[j]));
    }
    matrix[S, Jm1] logit_p_j = logit(p);
    if (ind) {
      array[I, Jm1] matrix[S, S] log_H_i = rep_array(log_H_j, I);
      array[I] matrix[S, Jm1] logit_p_i = rep_array(logit(p), I);
      log_lik = cjs_ms(y, f_l, log_H_i, logit_p_i);
    } else {
      array[Jm1] matrix[S, S] log_H = log_H_j;
      matrix[S, Jm1] logit_p = logit(p);
      log_lik = cjs_ms(y, f_l, log_H, logit_p);
    }
  }
}
