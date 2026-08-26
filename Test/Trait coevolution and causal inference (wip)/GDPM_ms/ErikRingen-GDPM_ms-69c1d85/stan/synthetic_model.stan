// Generated with coevolve 0.0.0.9017
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
  int<lower=1> N_seg; // total number of segments in the trees
  array[N_tree, N_seg] int<lower=1> node_seq; // index of tree nodes
  array[N_tree, N_seg] int<lower=0> parent; // index of the parent node of each descendent
  array[N_tree, N_seg] real ts; // time since parent
  array[N_tree, N_seg] int<lower=0,upper=1> tip; // indicator of whether a given segment ends in a tip
  array[J,J] int<lower=0,upper=1> effects_mat; // which effects should be estimated?
  int<lower=2> num_effects; // number of effects being estimated
  matrix[N_obs,J] y; // observed data
  matrix[N_obs,J] miss; // are data points missing?
  array[N_obs] int<lower=1> tip_id; // index between 1 and N_tips that gives the group id
  int<lower=0,upper=1> prior_only; // should the likelihood be ignored?
}
transformed data{
  vector[to_int(N_obs - sum(col(miss, 1)))] obs1; // observed data for variable 1
  vector[to_int(N_obs - sum(col(miss, 2)))] obs2; // observed data for variable 2
  vector[to_int(N_obs - sum(col(miss, 3)))] obs3; // observed data for variable 3
  obs1 = col(y, 1)[which_equal(col(miss, 1), 0)];
  obs2 = col(y, 2)[which_equal(col(miss, 2), 0)];
  obs3 = col(y, 3)[which_equal(col(miss, 3), 0)];
}
parameters{
  array[N_tree, N_seg - 1] vector[J] z_drift; // stochastic drift, unscaled and uncorrelated
  array[N_tree] matrix[N_obs, J] terminal_drift; // drift for the tips
}
transformed parameters{
  array[N_tree, N_seg] vector[J] eta;
  array[N_tree] vector[J] eta_anc; // ancestral states
  vector[J] b = rep_vector(0.0, J);
  matrix[J,J] A = diag_matrix(rep_vector(-0.5, J)); // selection matrix
  matrix[J,J] Q = diag_matrix(rep_vector(2.0, J)); // drift matrix
  matrix[J,J] Q_inf; // asymptotic covariance matrix
  array[N_tree, N_seg] matrix[J,J] VCV_tips; // variance-covariance matrix for drift at the tips
 
  A[2,1] = 3;
  A[1,3] = -2;
  A[2,3] = -2;
  A[3,1] = 1.5;
 
 for (t in 1:N_tree) eta_anc[t] = rep_vector(0.0, J);
 
  // calculate asymptotic covariance
  Q_inf = ksolve(A, Q);
  // loop over phylogenetic trees
  for (t in 1:N_tree) {
    // setting ancestral states and placeholders
    eta[t, node_seq[t, 1]] = eta_anc[t];
    VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(-99, J));
    for (i in 2:N_seg) {
      matrix[J,J] A_delta; // amount of deterministic change (selection)
      matrix[J,J] VCV; // variance-covariance matrix of stochastic change (drift)
      vector[J] drift_seg; // accumulated drift over the segment
      A_delta = matrix_exp(A * ts[t, i]);
      VCV = Q_inf - quad_form_sym(Q_inf, A_delta');
      drift_seg = cholesky_decompose(VCV) * z_drift[t, i-1];
      // if not a tip, add the drift parameter
      if (tip[t, i] == 0) {
        eta[t, node_seq[t, i]] = to_vector(
          A_delta * eta[t, parent[t, i]] + ((A \ add_diag(A_delta, -1)) * b) + drift_seg
        );
        VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(-99, J));
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
  for (t in 1:N_tree) {
    for (i in 1:(N_seg - 1)) z_drift[t, i] ~ std_normal();
  }
  if (!prior_only) {
    for (i in 1:N_obs) {
      vector[N_tree] lp = rep_vector(0.0, N_tree);
      for (t in 1:N_tree) {
        vector[J] residuals;
        if (miss[i,1] == 0) {
          residuals[1] = y[i,1] - eta[t,tip_id[i]][1];
          terminal_drift[t][i,1] ~ std_normal();
        } else {
          residuals[1] = terminal_drift[t][i,1];
        }
        if (miss[i,2] == 0) {
          residuals[2] = y[i,2] - eta[t,tip_id[i]][2];
          terminal_drift[t][i,2] ~ std_normal();
        } else {
          residuals[2] = terminal_drift[t][i,2];
        }
        if (miss[i,3] == 0) {
          residuals[3] = y[i,3] - eta[t,tip_id[i]][3];
          terminal_drift[t][i,3] ~ std_normal();
        } else {
          residuals[3] = terminal_drift[t][i,3];
        }
        lp[t] = multi_normal_cholesky_lpdf(residuals | rep_vector(0.0, J), cholesky_decompose(VCV_tips[t, tip_id[i]]));
      }
      target += log_sum_exp(lp);
    }
  }
}
generated quantities{
  array[N_tree,N_obs,J] real yrep; // predictive checks
  {
    for (i in 1:N_obs) {
      for (t in 1:N_tree) {
        vector[J] mu_cond;
        vector[J] sigma_cond;
        vector[J] residuals;
        vector[J] terminal_drift_rep;
        for (j in 1:J) terminal_drift_rep[j] = normal_rng(0, 1);
        terminal_drift_rep = cholesky_decompose(VCV_tips[t, tip_id[i]]) * terminal_drift_rep;
        residuals[1] = y[i][1] - eta[t,tip_id[i]][1];
        residuals[2] = y[i][2] - eta[t,tip_id[i]][2];
        residuals[3] = y[i][3] - eta[t,tip_id[i]][3];
        matrix[J,J] cov_inv = inverse_spd(VCV_tips[t, tip_id[i]]);
        mu_cond = residuals - (cov_inv * residuals) ./ diagonal(cov_inv);
        sigma_cond = sqrt(1 / diagonal(cov_inv));
        yrep[t,i,1] = eta[t,tip_id[i]][1] + terminal_drift_rep[1];
        yrep[t,i,2] = eta[t,tip_id[i]][2] + terminal_drift_rep[2];
        yrep[t,i,3] = eta[t,tip_id[i]][3] + terminal_drift_rep[3];
      }
    }
  }
}
