// Generated with coevolve 1.1.0

functions {

  // Charles Driver's solver for the asymptotic Q matrix
  matrix ksolve (matrix A, matrix Q) {
    int d = rows(A);
    int d2 = (d * d - d) %/% 2;
    matrix [d + d2, d + d2] O;
    vector [d + d2] triQ;
    matrix[d,d] AQ;
    int z = 0;         // z is row of output
    for (j in 1:d) {   // for column reference of solution vector
      for (i in 1:j) { // and row reference...
        if (j >= i) {  // if i and j denote a covariance parameter
          int y = 0;   // start new output row
          z += 1;      // shift current output row down
          for (ci in 1:d) {   // for columns and
            for (ri in 1:d) { // rows of solution
              if (ci >= ri) { // when in upper tri (inc diag)
                y += 1;       // move to next column of output
                if (i == j) { // if output row is a diag element
                  if (ri == i) O[z, y] = 2 * A[ri, ci];
                  if (ci == i) O[z, y] = 2 * A[ci, ri];
                }
                if (i != j) { // if output row is not a diag element
                  //if column matches row, sum both A diags
                  if (y == z) O[z, y] = A[ri, ri] + A[ci, ci];
                  if (y != z) { // otherwise...
                    // if solution element is related to output row...
                    if (ci == ri) { // if solution element is variance
                      // if variance of solution corresponds to row
                      if (ci == i) O[z, y] = A[j, ci];
                      // if variance of solution corresponds to col
                      if (ci == j) O[z, y] = A[i, ci];
                    }
                    //if solution element is a related covariance
                    if (ci != ri && (ri == i || ri == j || ci == i || ci == j )) {
                      // for row 1,2 / 2,1 of output,
                      // if solution row ri 1 (match)
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
  int<lower=1> J; // number of response traits
  int<lower=1> N_seg; // total number of segments in the trees
  array[N_tree, N_seg] int<lower=1> node_seq; // index of tree nodes
  array[N_tree, N_seg] int<lower=0> parent; // index of parent nodes
  array[N_tree, N_seg] real ts; // time since parent
  array[N_tree, N_seg] int<lower=0,upper=1> tip; // segment ends in tip
  array[J,J] int<lower=0,upper=1> effects_mat; // effects matrix
  int<lower=1> num_effects; // number of effects being estimated
  matrix[N_obs,J] y; // observed data
  matrix[N_obs,J] miss; // are data points missing?
  array[N_obs] int<lower=1> tip_id; // group index between 1 and N_tips
  int<lower=1> N_unique_lengths; // number of unique branch lengths
  array[N_unique_lengths] real unique_lengths; // unique branch lengths for caching
  array[N_tree, N_seg] int<lower=0> length_index; // mapping from segments to unique lengths
  array[N_tree, N_tips] int<lower=0> tip_to_seg; // mapping from tips to segments
  int<lower=0,upper=1> prior_only; // should likelihood be ignored?

}

transformed data {

  vector[to_int(N_obs - sum(col(miss, 1)))] obs1; // observed data for variable 1
  vector[to_int(N_obs - sum(col(miss, 2)))] obs2; // observed data for variable 2
  vector[to_int(N_obs - sum(col(miss, 3)))] obs3; // observed data for variable 3
  obs1 = col(y, 1)[which_equal(col(miss, 1), 0)];
  obs2 = col(y, 2)[which_equal(col(miss, 2), 0)];
  obs3 = col(y, 3)[which_equal(col(miss, 3), 0)];

}

parameters{

  vector<upper=0>[J] A_diag; // autoregressive terms of A
  vector[num_effects - J] A_offdiag; // cross-lagged terms of A
  vector<lower=0>[J] Q_sigma; // std deviation parameters of the Q mat
  vector[J] b; // SDE intercepts
  array[N_tree] vector[J] eta_anc; // ancestral states
  array[N_tree, N_seg - 1] vector[J] z_drift; // stochastic drift
  array[N_tree] matrix[N_tips, J] terminal_drift; // drift for the tips

}

transformed parameters{

  array[N_tree, N_seg] vector[J] eta;
  matrix[J,J] A = diag_matrix(A_diag); // selection matrix
  matrix[J,J] Q = diag_matrix(Q_sigma^2); // drift matrix
  matrix[J,J] Q_inf; // asymptotic covariance matrix
  array[N_tree, N_seg] matrix[J,J] VCV_tips; // vcov matrix for drift
  array[N_tree, N_seg] matrix[J,J] L_VCV_tips; // Cholesky factor of VCV_tips
  // fill off diagonal of A matrix
  {
    int ticker = 1;
    for (i in 1:J) {
      for (j in 1:J) {
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
  {
    array[N_unique_lengths] matrix[J,J] A_delta_cache;
    array[N_unique_lengths] matrix[J,J] VCV_cache;
    array[N_unique_lengths] matrix[J,J] L_VCV_cache;
    array[N_unique_lengths] matrix[J,J] A_solve_cache;
    for (u in 1:N_unique_lengths) {
      A_delta_cache[u] = matrix_exp(A * unique_lengths[u]);
      VCV_cache[u] = Q_inf - quad_form_sym(Q_inf, A_delta_cache[u]');
      L_VCV_cache[u] = cholesky_decompose(VCV_cache[u]);
      A_solve_cache[u] = A \ add_diag(A_delta_cache[u], -1);
      for (i in 1:J) {
        for (j in 1:i) {
          real val = 0.5 * (A_solve_cache[u][i, j] + A_solve_cache[u][j, i]);
          A_solve_cache[u][i, j] = val;
          A_solve_cache[u][j, i] = val;
        }
      }
    }
    for (t in 1:N_tree) {
      // setting ancestral states and placeholders
      eta[t, node_seq[t, 1]] = eta_anc[t];
      VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(-99, J));
      L_VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(1.0, J));
      for (i in 2:N_seg) {
        matrix[J,J] A_delta;
        matrix[J,J] VCV;
        vector[J] drift_seg;
        matrix[J,J] L_VCV;
        matrix[J,J] A_solve;
        if (length_index[t, i] > 0) {
          A_delta = A_delta_cache[length_index[t, i]];
          VCV = VCV_cache[length_index[t, i]];
          L_VCV = L_VCV_cache[length_index[t, i]];
          A_solve = A_solve_cache[length_index[t, i]];
        } else {
          A_delta = matrix_exp(A * ts[t, i]);
          VCV = Q_inf - quad_form_sym(Q_inf, A_delta');
          L_VCV = cholesky_decompose(VCV);
          A_solve = A \ add_diag(A_delta, -1);
        }
        drift_seg = L_VCV * z_drift[t, i-1];
        // if not a tip, add the drift parameter
        if (tip[t, i] == 0) {
          eta[t, node_seq[t, i]] = to_vector(
            A_delta * eta[t, parent[t, i]] + (A_solve * b) + drift_seg
          );
          VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(-99, J));
          L_VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(1.0, J));
        }
        // if is a tip, omit, we'll deal with it in the model block;
        else {
          eta[t, node_seq[t, i]] = to_vector(
            A_delta * eta[t, parent[t, i]] + (A_solve * b)
          );
          VCV_tips[t, node_seq[t, i]] = VCV;
          L_VCV_tips[t, node_seq[t, i]] = L_VCV;
        }
      }
    }
  }

}

model{

  // priors
  b ~ std_normal();
  for (t in 1:N_tree) {
    eta_anc[t] ~ std_normal();
    for (i in 1:(N_seg - 1)) z_drift[t, i] ~ std_normal();
  }
  A_offdiag ~ normal(0, 2);
  A_diag ~ std_normal();
  Q_sigma ~ normal(0, 2);

  // likelihood
  if (!prior_only) {
    for (i in 1:N_obs) {
      vector[N_tree] lp = rep_vector(0.0, N_tree);
      for (t in 1:N_tree) {
        // initialise tdrift
        vector[J] tdrift;
        // set tdrift
        if (miss[i,1] == 0) {
          tdrift[1] = y[i,1] - (eta[t,tip_id[i]][1]);
          terminal_drift[t][tip_id[i],1] ~ std_normal();
        } else {
          tdrift[1] = terminal_drift[t][tip_id[i],1];
        }
        if (miss[i,2] == 0) {
          tdrift[2] = y[i,2] - (eta[t,tip_id[i]][2]);
          terminal_drift[t][tip_id[i],2] ~ std_normal();
        } else {
          tdrift[2] = terminal_drift[t][tip_id[i],2];
        }
        if (miss[i,3] == 0) {
          tdrift[3] = y[i,3] - (eta[t,tip_id[i]][3]);
          terminal_drift[t][tip_id[i],3] ~ std_normal();
        } else {
          tdrift[3] = terminal_drift[t][tip_id[i],3];
        }
        lp[t] = multi_normal_cholesky_lpdf(tdrift | rep_vector(0.0, J), L_VCV_tips[t, tip_id[i]]);
      }
      target += log_sum_exp(lp);
    }
  }

}

generated quantities{

  array[N_tree,N_obs,J] real yrep; // predictive checks
  {
    array[N_tree,N_tips] vector[J] terminal_drift_rep;
    for (i in 1:N_tips) {
      for (t in 1:N_tree) {
        for (j in 1:J) terminal_drift_rep[t,i][j] = normal_rng(0, 1);
        terminal_drift_rep[t,i] = cholesky_decompose(VCV_tips[t, i]) * terminal_drift_rep[t,i];
      }
    }
    for (i in 1:N_obs) {
      for (t in 1:N_tree) {
        yrep[t,i,1] = eta[t,tip_id[i]][1] + terminal_drift_rep[t,tip_id[i]][1];
        yrep[t,i,2] = eta[t,tip_id[i]][2] + terminal_drift_rep[t,tip_id[i]][2];
        yrep[t,i,3] = eta[t,tip_id[i]][3] + terminal_drift_rep[t,tip_id[i]][3];
      }
    }
  }

}
