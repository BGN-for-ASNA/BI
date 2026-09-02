functions {
  // Charles Driver's optimized way of solving for the asymptotic Q matrix
  matrix ksolve (matrix A, matrix Q) {
    int d = rows(A);
    int d2 = (d * d - d) %/% 2;
    matrix [d + d2, d + d2] O;
    vector [d + d2] triQ;
    matrix[d,d] AQ;
    int z = 0;         // z is row of output
    for (j in 1:d) {   // for column reference of solution vector
      for (i in 1:j) { // and row reference...
        if (j >= i) {  // if i and j denote a covariance parameter (from upper tri)
          int y = 0;   // start new output row
          z += 1;      // shift current output row down
          for (ci in 1:d) {   // for columns and
            for (ri in 1:d) { // rows of solution
              if (ci >= ri) { // when in upper tri (inc diag)
                y += 1;       // move to next column of output
                if (i == j) { // if output row is for a diagonal element
                  if (ri == i) O[z, y] = 2 * A[ri, ci];
                  if (ci == i) O[z, y] = 2 * A[ci, ri];
                }
                if (i != j) { // if output row is not for a diagonal element
                  //if column of output matches row of output, sum both A diags
                  if (y == z) O[z, y] = A[ri, ri] + A[ci, ci];
                  if (y != z) { // otherwise...
                    // if solution element we refer to is related to output row...
                    if (ci == ri) { // if solution element is a variance
                      // if variance of solution corresponds to row of our output
                      if (ci == i) O[z, y] = A[j, ci];
                      // if variance of solution corresponds to col of our output
                      if (ci == j) O[z, y] = A[i, ci];
                    }
                    //if solution element is a related covariance
                    if (ci != ri && (ri == i || ri == j || ci == i || ci == j )) {
                      // for row 1,2 / 2,1 of output, if solution row ri 1 (match)
                      // and column ci 3, we need A[2,3]
                      if (ri == i) O[z, y] = A[j, ci];
                      if (ri == j) O[z, y] = A[i, ci];
                      if (ci == i) O[z, y] = A[j, ri];
                      if (ci == j) O[z, y] = A[i, ri];
                    }
                  }
                }
                if (is_nan(O[z, y])) O[z, y] = 0;
              }
            }
          }
        }
      }
    }
    z = 0; // get upper tri of Q
    for (j in 1:d) {
      for (i in 1:j) {
        z += 1;
        triQ[z] = Q[i, j];
      }
    }
    triQ = -O \ triQ; // get upper tri of asymQ
    z = 0; // put upper tri of asymQ into matrix
    for (j in 1:d) {
      for (i in 1:j) {
        z += 1;
        AQ[i, j] = triQ[z];
        if (i != j) AQ[j, i] = triQ[z];
      }
    }
    return AQ;
  }
  
  // return number of matches of y in vector x
  int num_matches(vector x, real y) {
    int n = 0;
    for (i in 1:rows(x))
      if (x[i] == y)
        n += 1;
    return n;
  }
  
  // return indices in vector x where x == y
  array[] int which_equal(vector x, real y) {
    array [num_matches(x, y)] int match_positions;
    int pos = 1;
    for (i in 1:rows(x)) {
      if (x[i] == y) {
        match_positions[pos] = i;
        pos += 1;
      }
    }
    return match_positions;
  }
}
data{
  int<lower=1> N_tips; // number of tips
  int<lower=1> N_tree; // number of trees
  int<lower=1> N_obs; // number of observations
  int<lower=2> J; // number of response traits
  int<lower=1> N_latent; // number of latent variables 
  int<lower=1> N_seg; // total number of segments in the trees
  array[N_tree, N_seg] int<lower=1> node_seq; // index of tree nodes
  array[N_tree, N_seg] int<lower=0> parent; // index of the parent node of each descendent
  array[N_tree, N_seg] real ts; // time since parent
  array[N_tree, N_seg] int<lower=0,upper=1> tip; // indicator of whether a given segment ends in a tip
  array[N_latent,N_latent] int<lower=0,upper=1> effects_mat; // which effects should be estimated?
  int<lower=2> num_effects; // number of effects being estimated
  matrix[N_obs,J] y; // observed data
  matrix[N_obs,J] miss; // are data points missing?
  array[N_obs] int<lower=1> tip_id; // index between 1 and N_tips that gives the group id
  int<lower=0,upper=1> prior_only; // should the likelihood be ignored?
  vector[J] y_mean; // sample mean for scaling
}
parameters{
  vector<upper=0>[N_latent] A_diag; // autoregressive terms of A
  vector[num_effects - N_latent] A_offdiag; // cross-lagged terms of A
  cholesky_factor_corr[N_latent] L_R; // lower-tri choleksy decomp corr mat, used to construct Q mat
  vector<lower=0>[N_latent] Q_sigma; // std deviation parameters of the Q mat
  vector[N_latent] b; // SDE intercepts
  array[N_tree] vector[N_latent] eta_anc; // ancestral states
  array[N_tree, N_seg - 1] vector[N_latent] z_drift; // stochastic drift, unscaled and uncorrelated
  array[N_tree] matrix[N_obs, N_latent] terminal_drift; // drift for the tips
  vector[J] alpha; // intercepts for observed traits
  vector<lower=0>[J] shape; // shape parameter for gamma distribution
  vector[J - N_latent] lambda_free; // parameters for factor matrix
}
transformed parameters{
  array[N_tree, N_seg] vector[N_latent] eta;
  matrix[N_latent,N_latent] A = diag_matrix(A_diag); // selection matrix
  matrix[N_latent,N_latent] Q = diag_matrix(Q_sigma) * (L_R * L_R') * diag_matrix(Q_sigma); // drift matrix
  matrix[N_latent,N_latent] Q_inf; // asymptotic covariance matrix
  array[N_tree, N_seg] matrix[N_latent,N_latent] VCV_tips; // variance-covariance matrix for drift at the tips
  matrix[J, N_latent] Lambda = rep_matrix(0.0, J, N_latent); // latent factor matrix 
  // fill factor matrix
  Lambda[1, 1] = 1.0;
  Lambda[3, 1] = lambda_free[1];
  Lambda[4, 1] = lambda_free[2];
  Lambda[2, 2] = 1.0;
  // fill off diagonal of A matrix
  {
    int ticker = 1;
    for (i in 1:N_latent) {
      for (j in 1:N_latent) {
        if (i != j) {
          if (effects_mat[i,j] == 1) {
            A[i,j] = A_offdiag[ticker];
            ticker += 1;
          } else if (effects_mat[i,j] == 0) {
            A[i,j] = 0;
          }
        }
      }
    }
  }
  // calculate asymptotic covariance
  Q_inf = ksolve(A, Q);
  // loop over phylogenetic trees
  for (t in 1:N_tree) {
    // setting ancestral states and placeholders
    eta[t, node_seq[t, 1]] = eta_anc[t];
    VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(-99, N_latent));
    for (i in 2:N_seg) {
      matrix[N_latent,N_latent] A_delta; // amount of deterministic change (selection)
      matrix[N_latent,N_latent] VCV; // variance-covariance matrix of stochastic change (drift)
      vector[N_latent] drift_seg; // accumulated drift over the segment
      A_delta = matrix_exp(A * ts[t, i]);
      VCV = Q_inf - quad_form_sym(Q_inf, A_delta');
      drift_seg = cholesky_decompose(VCV) * z_drift[t, i-1];
      // if not a tip, add the drift parameter
      if (tip[t, i] == 0) {
        eta[t, node_seq[t, i]] = to_vector(
          A_delta * eta[t, parent[t, i]] + ((A \ add_diag(A_delta, -1)) * b) + drift_seg
        );
        VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(-99, N_latent));
      }
      // if is a tip, omit, we'll deal with it in the model block;
      else {
        eta[t, node_seq[t, i]] = to_vector(
          A_delta * eta[t, parent[t, i]] + ((A \ add_diag(A_delta, -1)) * b)
        );
        VCV_tips[t, node_seq[t, i]] = VCV;
      }
    }
  }
}
model{
  // priors
  b ~ std_normal();
  for (t in 1:N_tree) {
    eta_anc[t][1] ~ std_normal();
    eta_anc[t][2] ~ normal(-0.2, 0.15); // informative prior on ancestral state for brain allometry, latent scale
    for (i in 1:(N_seg - 1)) z_drift[t, i] ~ std_normal();
  }
  A_offdiag ~ std_normal();
  A_diag ~ std_normal();
  L_R ~ lkj_corr_cholesky(4);
  Q_sigma ~ std_normal();
  alpha ~ std_normal();
  shape ~ gamma(0.01, 0.01);
  lambda_free ~ std_normal();
  // model loop over trees and observations
  if (!prior_only) {
    for (i in 1:N_obs) {
      vector[N_tree] lp;
      for (t in 1:N_tree) {
        vector[N_latent] eta_tips;
        real beta; // allometric slope

        lp[t] = multi_normal_cholesky_lpdf(terminal_drift[t][i,] | rep_vector(0.0, N_latent), cholesky_decompose(VCV_tips[t, tip_id[i]]));

        for (n in 1:N_latent) {
          eta_tips[n] = eta[t, tip_id[i]][n] + terminal_drift[t][i,n];
        }
        
        // body size
        lp[t] += gamma_lpdf(y[i,1]/y_mean[1] | shape[1], shape[1]/exp(alpha[1] + Lambda[1,1] * eta_tips[1]));
        
        // brain size
        // allometric eq for brain size = alpha*(body_weight^b)
        // we'll work on log scale so log(brain) = log(alpha) + beta * log(body_weight)
        // don't scale to avoid exponentation below 1
        beta = log(1 + exp(Lambda[2,2] * eta_tips[2]));
        lp[t] += gamma_lpdf(y[i,2] | shape[2], shape[2]/exp(alpha[2] + log(y[i,1])*beta));
        
        // longevity
        if (miss[i,3] == 0) lp[t] += gamma_lpdf(y[i,3]/y_mean[3] | shape[3], shape[3]/exp(alpha[3] + Lambda[3,1] * eta_tips[1]));
        
        // age at fem maturity
        if (miss[i,4] == 0) lp[t] += gamma_lpdf(y[i,4]/y_mean[4] | shape[4], shape[4]/exp(alpha[4] + Lambda[4,1] * eta_tips[1]));
      }
      target += log_sum_exp(lp);
    }
  }
}
generated quantities{
  array[N_tree,N_obs,J] real yrep; // predictive checks
  matrix[N_latent,N_latent] cor_R; // correlated drift
  cor_R = multiply_lower_tri_self_transpose(L_R);
  {
    for (i in 1:N_obs) {
      for (t in 1:N_tree) {
        vector[N_latent] terminal_drift_rep;
        vector[N_latent] eta_tips_rep;
        real beta_rep;
        
        for (n in 1:N_latent) terminal_drift_rep[n] = normal_rng(0, 1);
        terminal_drift_rep = cholesky_decompose(VCV_tips[t, tip_id[i]]) * terminal_drift_rep;
        eta_tips_rep = eta[t, tip_id[i]] + terminal_drift_rep;
        
        // body size
        yrep[t,i,1] = gamma_rng(shape[1], shape[1]/exp(alpha[1] + Lambda[1,1] * eta_tips_rep[1])) * y_mean[1];
        
        // brain size
        beta_rep = log(1 + exp(Lambda[2,2] * eta_tips_rep[2]));
        yrep[t,i,2] = gamma_rng(shape[2], shape[2]/exp(alpha[2] + log(yrep[t,i,1])*beta_rep));
        
        // longevity
        yrep[t,i,3] = gamma_rng(shape[3], shape[3]/exp(alpha[3] + Lambda[3,1] * eta_tips_rep[1])) * y_mean[3];
        
        // maturity
        yrep[t,i,4] = gamma_rng(shape[4], shape[4]/exp(alpha[4] + Lambda[4,1] * eta_tips_rep[1])) * y_mean[4];
        }
      }
    }
}
