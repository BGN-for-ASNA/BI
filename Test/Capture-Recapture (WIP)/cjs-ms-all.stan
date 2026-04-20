functions {
  /** 
   * Elementwise natural logarithm of the product of the elementwise 
   * exponentiation of two matrices
   */
  matrix log_prod_exp(matrix A, matrix B) {
    int I = rows(A);
    int J = cols(A);
    int K = cols(B);
    matrix[J, I] A_tr = A';
    matrix[I, K] C;
    for (k in 1:K) {
      for (i in 1:I) {
        C[i, k] = log_sum_exp(A_tr[:, i] + B[:, k]);
      }
    }
    return C;
  }
  vector log_prod_exp(matrix A, vector B) {
    int I = rows(A);
    int J = cols(A);
    matrix[J, I] A_tr = A';
    vector[I] C;
    for (i in 1:I) {
      C[i] = log_sum_exp(A_tr[:, i] + B);
    }
    return C;
  }
  
  /** 
   * Create transition rate matrix from mortality and transition rates
   */
  matrix rate_matrix(vector h, row_vector q) {
    int S = size(h), Sm1 = S - 1, Sp1 = S + 1;
    matrix[Sp1, Sp1] Q = rep_matrix(0, Sp1, Sp1);
    row_vector[Sm1] q_s;
    int idx = 1;
    for (s in 1:S) {
      q_s = segment(q, idx, Sm1);
      Q[s, 1:s - 1] = head(q_s, s - 1);
      Q[s, s] = -h[s] - sum(q_s);
      Q[s, s + 1:S] = tail(q_s, S - s);
      idx += Sm1;
    }
    Q[:S, Sp1] = h;
    return Q;
  }

  /** 
   * Get first and last survey of detection for detection history.
   */
  array[,] int first_last(data array[,] int y) {
    int I = size(y), J = size(y[1]);
    array[I, 2] int f_l = rep_array(0, I, 2);
    for (i in 1:I) {
      for (j in 1:J) {
        if (y[i, j]) {
          f_l[i] = rep_array(j, 2);
          break;
        }
      }
      if (f_l[i, 1] > 0) {
        int JJ = J - f_l[i, 1];
        if (JJ > 0) {
          array[JJ] int idx = linspaced_int_array(JJ, f_l[i, 1] + 1, J);
          for (j in reverse(idx)) {
            if (y[i, j]) {
              f_l[i, 2] = j;
              break;
            }
          }
        }
      }
    }
    return f_l;
  }

  /** 
   * Multistate Cormack-Jolly-Seber individual log likelihoods.
   */
  vector cjs_ms(data array[,] int y, data array[,] int f_l, array[] matrix log_H, 
                matrix logit_p) {
    int I = size(y), J = size(y[1]), Jm1 = J - 1, S = rows(log_H[1]);
    matrix[S, Jm1] log_1mp = log1m_inv_logit(logit_p),
                   log_p = log_1mp + logit_p;
    vector[S] Omega;
    vector[Jm1] log1m_chi;
    matrix[S, Jm1] log_chi;
    for (l in 1:Jm1) {
      for (s in 1:S) {
        Omega = log_H[l, s]' + log_1mp[:, l];
        log1m_chi[l] = log_sum_exp(Omega + logit_p[:, l]);
        for (j in l + 1:Jm1) {
          Omega = log_prod_exp(log_H[j]', Omega) + log_1mp[:, j];
          log1m_chi[j] = log_sum_exp(Omega + logit_p[:, j]);
        }
        log_chi[s, l] = log1m_exp(log_sum_exp(log1m_chi[l:]));
      }
    }
    array[J] int y_i;
    vector[I] log_lik = rep_vector(0, I);
    for (i in 1:I) {
      int f = f_l[i, 1], l = f_l[i, 2];
      if (f == 0) continue;
      y_i[f:l] = y[i, f:l];
      int y_j = y_i[f];
      if (f < l) {
        Omega = rep_vector(negative_infinity(), S);
        Omega[y_j] = 0;
        for (j in f + 1:l) {
          int jm1 = j - 1, y_jm1 = y_j;
          y_j = y_i[j];
          if (y_j) {
            real lp_j = y_jm1 ?
                        Omega[y_jm1] + log_H[jm1, y_jm1, y_j]
                        : log_sum_exp(Omega + log_H[jm1, :, y_j]);
            Omega = rep_vector(negative_infinity(), S);
            Omega[y_j] = lp_j + log_p[y_j, jm1];
          } else {
            Omega = y_jm1 ?
                    Omega[y_jm1] + log_H[jm1, y_jm1]'
                    : log_prod_exp(log_H[jm1]', Omega);
            Omega += log_1mp[:, jm1];
          }
        }
        log_lik[i] += Omega[y_j];
      }
      if (l < J) {
        log_lik[i] += log_chi[y_j, l];
      }
    }
    return log_lik;
  }
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
  
  // likelihood (non-individual for this study)
  target += cjs_ms(y, f_l, log_H_j, logit_p_j);
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
    log_lik = cjs_ms(y, f_l, log_H_j, logit_p_j);
  }
}
