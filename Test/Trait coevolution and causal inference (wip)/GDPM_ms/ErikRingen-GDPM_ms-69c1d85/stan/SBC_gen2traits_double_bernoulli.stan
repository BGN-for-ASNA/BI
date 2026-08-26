// Generative model for SBC with 2 traits: both Bernoulli-logit
// Variant of SBC_gen2traits.stan for double Bernoulli traits
functions {
  // Charles Driver's solver for the asymptotic Q matrix
  matrix ksolve (matrix A, matrix Q) {
    int d = rows(A);
    int d2 = (d * d - d) %/% 2;
    matrix [d + d2, d + d2] O;
    vector [d + d2] triQ;
    matrix[d,d] AQ;
    int z = 0;       // z is row of output
    for (j in 1:d) {    // for column reference of solution vector
      for (i in 1:j) { // and row reference...
        if (j >= i) {  // if i and j denote a covariance parameter
          int y = 0;    // start new output row
          z += 1;      // shift current output row down
          for (ci in 1:d) {   // for columns and
            for (ri in 1:d) { // rows of solution
              if (ci >= ri) { // when in upper tri (inc diag)
                y += 1;        // move to next column of output
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
  int<lower=2> J; // number of response traits
  int<lower=1> N_seg; // total number of segments in the trees
  array[N_tree, N_seg] int<lower=1> node_seq; // index of tree nodes
  array[N_tree, N_seg] int<lower=0> parent; // index of parent nodes
  array[N_tree, N_seg] real ts; // time since parent
  array[N_tree, N_seg] int<lower=0,upper=1> tip; // segment ends in tip
  array[J,J] int<lower=0,upper=1> effects_mat; // effects matrix
  int<lower=2> num_effects; // number of effects being estimated
  matrix[N_obs,J] y; // observed data (not used in prior predictive)
  matrix[N_obs,J] miss; // are data points missing? (not used in prior predictive)
  array[N_obs] int<lower=1> tip_id; // group index between 1 and N_tips
  int<lower=1> N_unique_lengths; // number of unique branch lengths
  array[N_unique_lengths] real unique_lengths; // unique branch lengths for caching
  array[N_tree, N_seg] int<lower=0> length_index; // mapping from segments to unique lengths
  array[N_tree, N_tips] int<lower=0> tip_to_seg; // mapping from tips to segments
  int<lower=0,upper=1> prior_only; // should likelihood be ignored? (Always true now)
}
transformed data{
  vector[to_int(N_obs - sum(col(miss, 1)))] obs1;
  vector[to_int(N_obs - sum(col(miss, 2)))] obs2;
  obs1 = col(y, 1)[which_equal(col(miss, 1), 0)];
  obs2 = col(y, 2)[which_equal(col(miss, 2), 0)];
}
model{
  // The model block is effectively empty since there are no parameters
  // to be sampled and no likelihood to be computed.
}
generated quantities{
  // Declare all variables that were previously in 'parameters' and 'transformed parameters'
  vector[J] b;
  vector[J] A_diag;
  vector[num_effects - J] A_offdiag;
  vector[J] Q_sigma;

  // SDE state variables and intermediate calculations
  matrix[J,J] A;
  matrix[J,J] Q;
  matrix[J,J] Q_inf;
  array[N_tree] vector[J] eta_anc;
  array[N_tree, N_seg - 1] vector[J] z_drift;
  array[N_tree, N_seg] matrix[J,J] VCV_tips;
  array[N_tree, N_seg] matrix[J,J] L_VCV_tips;
  array[N_tree, N_seg] vector[J] eta;

  // Prior sampling of all latent variables using _rng functions
  for (j in 1:J) b[j] = normal_rng(0.0, 1);
  for (j in 1:J) A_diag[j] = -1*abs(normal_rng(-1.0, 0.5));  // Centered on -0.5 (abs centered on 0.5)
  for (j in 1:(num_effects - J)) A_offdiag[j] = normal_rng(0, 2.5);  // Increased from 2 to 2.5
  for (j in 1:J) Q_sigma[j] = normal_rng(2, 1); // std_normal_rng is for positive, but we will square it later

  // Ensure Q_sigma elements are positive (since they are std deviations)
  for (j in 1:J) Q_sigma[j] = abs(Q_sigma[j]);

  for (t in 1:N_tree) {
    for (j in 1:J) eta_anc[t][j] = std_normal_rng();
    for (i in 1:(N_seg - 1)) {
      for (j in 1:J) z_drift[t, i, j] = std_normal_rng();
    }
  }

  // Recalculate 'transformed parameters' logic

  // 1. Calculate A and Q matrices
  A = diag_matrix(A_diag);
  Q = diag_matrix(Q_sigma^2);

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

  // 2. Calculate asymptotic covariance
  Q_inf = ksolve(A, Q);

  // 3. Phylogenetic process calculation (including caching)
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
      VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(-99, J)); // placeholder
      L_VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(1.0, J)); // placeholder

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

        drift_seg = L_VCV * z_drift[t, i-1]; // Use z_drift (sampled from prior)

        // if not a tip, add the drift parameter
        if (tip[t, i] == 0) {
          eta[t, node_seq[t, i]] = to_vector(
            A_delta * eta[t, parent[t, i]] + (A_solve * b) + drift_seg
          );
          VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(-99, J)); // placeholder
          L_VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(1.0, J)); // placeholder
        }
        // if is a tip, store VCV for terminal drift calculation
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

  // 4. Prior Predictive Check (yrep) calculation
  // BOTH TRAITS ARE BERNOULLI-LOGIT
  array[N_tree,N_obs,J] real yrep; // predictive checks
  {
    array[N_tree,N_tips] vector[J] terminal_drift_rep;
    for (i in 1:N_tips) {
      for (t in 1:N_tree) {
        // Sample the standard normal latent drift for the tips
        vector[J] z_terminal_drift;
        for (j in 1:J) z_terminal_drift[j] = std_normal_rng();
        // Calculate terminal drift using Cholesky factor (L_VCV_tips)
        terminal_drift_rep[t,i] = L_VCV_tips[t, i] * z_terminal_drift;
      }
    }

    // Combine systemic state (eta) and terminal drift to get final trait values (yrep)
    for (i in 1:N_obs) {
      for (t in 1:N_tree) {
        // Trait 1 (Bernoulli-logit)
        yrep[t,i,1] = bernoulli_logit_rng(eta[t,tip_id[i]][1] + terminal_drift_rep[t,tip_id[i]][1]);
        // Trait 2 (Bernoulli-logit) - BOTH ARE BERNOULLI
        yrep[t,i,2] = bernoulli_logit_rng(eta[t,tip_id[i]][2] + terminal_drift_rep[t,tip_id[i]][2]);
      }
    }
  }
}

