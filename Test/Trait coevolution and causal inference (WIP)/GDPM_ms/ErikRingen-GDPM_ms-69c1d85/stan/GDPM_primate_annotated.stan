// ============================================================================
// GENERALIZED DYNAMIC PHYLOGENETIC MODEL (GDPM) - ANNOTATED VERSION
// ============================================================================
// This Stan model implements a GDPM for analyzing coevolution of continuous
// traits on phylogenetic trees. This annotated version provides detailed
// documentation connecting the Stan implementation to the mathematical model
// described in the manuscript.
//
// REFERENCE: This implementation follows Driver (2018) for efficient SDE
// solution via the asymptotic covariance approach.
//
// ============================================================================
// MODEL OVERVIEW
// ============================================================================
// The GDPM has two main components:
//
// 1. LATENT EVOLUTIONARY MODEL (Manuscript Eq. 1):
//    A system of stochastic differential equations (SDEs) describing how
//    K latent traits evolve over time along the phylogeny:
//
//      dη(t) = (A·η(t) + b)dt + G·dW(t)
//      where dW(t) ~ √dt · N(0, I)
//
// 2. OBSERVATIONAL MODEL (Manuscript Eq. 2):
//    A measurement model linking latent traits to J observed variables:
//
//      y[n,j] ~ f(μ[n,j], φ[j])
//      g(μ[n,j]) = Λ[j,·] · η[n] + α[j]
//
//    where f() is a probability distribution, g() is a link function,
//    and Λ is a factor loading matrix.
//
// ============================================================================
// SYMBOL MAPPING: Manuscript ↔ Stan Code
// ============================================================================
//
//  Manuscript    Stan Variable        Dimensions        Description
//  ----------    -------------        ----------        -----------
//  η(t)          eta                  [N_tree,N_seg,K]  Latent trait values at nodes
//  A             A                    [K,K]             Selection/cross-effect matrix
//  b             b                    [K]               Continuous-time intercepts
//  Q             Q                    [K,K]             Drift covariance matrix
//  G             cholesky_decompose(Q) [K,K]            Cholesky factor of Q (G·G'=Q)
//  Q_Δ           VCV                  [K,K]             Drift covariance for interval Δt
//  Q_∞           Q_inf                [K,K]             Asymptotic covariance
//  A_Δ           A_delta              [K,K]             Matrix exponential exp(A·Δt)
//  b_Δ           (A\(A_Δ-I))·b        [K]               Intercept contribution over Δt
//  Λ             Lambda               [J,K]             Factor loading matrix
//  α             alpha                [J]               Observational intercepts
//  φ             shape                [J]               Distributional parameters
//  K             N_latent             scalar            Number of latent traits
//  J             J                    scalar            Number of observed traits
//  N             N_tips               scalar            Number of tip species
//
// ============================================================================
// FILE STRUCTURE
// ============================================================================
// - functions:             Custom functions for efficient SDE solution
// - data:                  Input data including phylogeny and observations
// - parameters:            Model parameters to be estimated via MCMC
// - transformed parameters: Derived quantities including trait evolution
// - model:                 Prior distributions and likelihood
// - generated quantities:  Posterior predictive checks
//
// ============================================================================

functions {
  // ==========================================================================
  // FUNCTION: ksolve - Efficient Lyapunov Equation Solver
  // ==========================================================================
  // Computes the asymptotic covariance matrix Q_∞ by solving the continuous-
  // time Lyapunov equation. This approach follows Driver (2018).
  //
  // MATHEMATICAL BACKGROUND:
  // -------------------------
  // For the multivariate OU process dη = (A·η + b)dt + G·dW, the stationary
  // (asymptotic) covariance Q_∞ satisfies the Lyapunov equation. The manuscript
  // presents this in vectorized form (Eq. in Section 3.1):
  //
  //   vec(Q_∞) = -(A ⊗ I + I ⊗ A)^{-1} · vec(Q)
  //
  // where ⊗ denotes the Kronecker product and vec() vectorizes a matrix.
  //
  // COMPUTATIONAL EFFICIENCY:
  // -------------------------
  // Direct solution via Kronecker products requires O(K^6) operations for a

  // K×K system. This function exploits symmetry of covariance matrices to work
  // only with the upper triangle, reducing complexity significantly.
  //
  // KEY INSIGHT: Q_∞ only needs to be computed ONCE per MCMC iteration
  // (assuming A and Q don't vary across tree segments). The per-segment
  // drift covariance Q_Δ is then computed efficiently as:
  //
  //   Q_Δ = Q_∞ - A_Δ · Q_∞ · A_Δ'
  //
  // where A_Δ = exp(A·Δt) is the matrix exponential for time interval Δt.
  //
  // INPUTS:
  //   A: Selection matrix (K×K), must have negative real eigenvalues for
  //      stability (diagonal elements must be negative)
  //   Q: Drift covariance matrix (K×K), positive semi-definite
  //
  // OUTPUT:
  //   Q_∞: Asymptotic covariance matrix (K×K), positive definite
  //
  // NUMERICAL STABILITY NOTE:
  // If A has eigenvalues close to zero, Q_∞ can become very large or
  // ill-conditioned. The constraint A_diag < 0 helps ensure stability.
  // ==========================================================================
  matrix ksolve (matrix A, matrix Q) {
    int d = rows(A);
    int d2 = (d * d - d) %/% 2;  // Number of unique off-diagonal elements
    matrix [d + d2, d + d2] O;    // Coefficient matrix for reduced linear system
    vector [d + d2] triQ;         // Vectorized upper triangle of Q
    matrix[d,d] AQ;               // Output: asymptotic covariance Q_∞
    int z = 0;                     // Row index in coefficient matrix O
    
    // Build the coefficient matrix O for the reduced linear system
    // This encodes the Lyapunov equation but works only with the K(K+1)/2
    // unique elements of the symmetric matrix Q_∞
    for (j in 1:d) {
      for (i in 1:j) {
        if (j >= i) {
          int y = 0;
          z += 1;
          for (ci in 1:d) {
            for (ri in 1:d) {
              if (ci >= ri) {
                y += 1;
                
                // Diagonal elements of Q_∞ (variances)
                if (i == j) {
                  if (ri == i) O[z, y] = 2 * A[ri, ci];
                  if (ci == i) O[z, y] = 2 * A[ci, ri];
                }
                
                // Off-diagonal elements of Q_∞ (covariances)
                if (i != j) {
                  if (y == z) O[z, y] = A[ri, ri] + A[ci, ci];
                  if (y != z) {
                    if (ci == ri) {
                      if (ci == i) O[z, y] = A[j, ci];
                      if (ci == j) O[z, y] = A[i, ci];
                    }
                    if (ci != ri && (ri == i || ri == j || ci == i || ci == j )) {
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
    
    // Extract upper triangle of Q into vector form
    z = 0;
    for (j in 1:d) {
      for (i in 1:j) {
        z += 1;
        triQ[z] = Q[i, j];
      }
    }
    
    // Solve the linear system: O · vec(Q_∞) = -vec(Q)
    // The backslash operator (\) solves the system efficiently
    triQ = -O \ triQ;
    
    // Reconstruct symmetric Q_∞ matrix from vectorized upper triangle
    z = 0;
    for (j in 1:d) {
      for (i in 1:j) {
        z += 1;
        AQ[i, j] = triQ[z];
        if (i != j) AQ[j, i] = triQ[z];  // Fill lower triangle by symmetry
      }
    }
    return AQ;
  }
  
  // Utility functions for indexing operations
  // NOTE: These are included for potential use in model extensions or
  // post-processing but are not called in this particular model.
  
  // Count occurrences of value y in vector x
  int num_matches(vector x, real y) {
    int n = 0;
    for (i in 1:rows(x))
      if (x[i] == y)
        n += 1;
    return n;
  }
  
  // Return indices where vector x equals value y
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

// ============================================================================
// DATA BLOCK: Input Data Structure
// ============================================================================
// All data must be prepared in R before fitting. The phylogenetic tree is
// preprocessed into a segment-based representation for efficient traversal.
// ============================================================================
data{
  // ========================================================================
  // DIMENSIONS
  // ========================================================================
  int<lower=1> N_tips;      // Number of tip species in the phylogeny
  int<lower=1> N_tree;      // Number of trees (>1 for phylogenetic uncertainty)
  int<lower=1> N_obs;       // Number of observations (= N_tips if no repeated measures)
  int<lower=2> J;           // Number of observed response traits
  int<lower=1> N_latent;    // Number of latent variables (K in manuscript notation)
  int<lower=1> N_seg;       // Total number of segments (branches) per tree
  
  // ========================================================================
  // PHYLOGENETIC TREE STRUCTURE
  // ========================================================================
  // The tree is represented as a sequence of directed segments (edges), where
  // each segment connects a parent node to a child node. This representation
  // enables efficient forward-time traversal from root to tips.
  //
  // PREPROCESSING (done in R):
  // The tree is converted to these arrays using a pre-order (root-first,
  // depth-first) traversal. For each segment i:
  //   - node_seq[t,i]: Index of the child node for segment i in tree t
  //   - parent[t,i]:   Index of the parent node (0 for root)
  //   - ts[t,i]:       Branch length (time duration) of segment i
  //   - tip[t,i]:      1 if segment ends at a tip species, 0 otherwise
  //
  // EXAMPLE: Simple 3-tip tree
  //
  //              root (node 1)
  //              /          \
  //         node 2           \
  //         /    \            \
  //      tip 1   tip 2       tip 3
  //
  // Would be encoded as (for one tree):
  //   Segment 1: node_seq=2, parent=1, ts=t₁, tip=0  (root → node2)
  //   Segment 2: node_seq=3, parent=2, ts=t₂, tip=1  (node2 → tip1)
  //   Segment 3: node_seq=4, parent=2, ts=t₃, tip=1  (node2 → tip2)
  //   Segment 4: node_seq=5, parent=1, ts=t₄, tip=1  (root → tip3)
  //
  // The first "segment" (i=1) initializes the root with ancestral state.
  // ========================================================================
  array[N_tree, N_seg] int<lower=1> node_seq;  // Child node index for each segment
  array[N_tree, N_seg] int<lower=0> parent;   // Parent node index (0 = root)
  array[N_tree, N_seg] real ts;                // Branch length (time since parent)
  array[N_tree, N_seg] int<lower=0,upper=1> tip; // 1 if segment ends at tip, 0 otherwise
  
  // ========================================================================
  // MODEL STRUCTURE: Directed Acyclic Graph (DAG)
  // ========================================================================
  // effects_mat specifies which cross-effects in A should be estimated.
  // This encodes the causal DAG structure:
  //   effects_mat[i,j] = 1: Estimate A[i,j] (effect of trait j on trait i)
  //   effects_mat[i,j] = 0: Fix A[i,j] = 0 (no causal effect)
  //
  // Diagonal elements (i=j) are always estimated (autoregressive terms).
  array[N_latent,N_latent] int<lower=0,upper=1> effects_mat;
  int<lower=2> num_effects;  // Total effects to estimate (diagonal + specified off-diagonal)
  
  // ========================================================================
  // OBSERVED DATA
  // ========================================================================
  matrix[N_obs,J] y;         // Observed trait values [observations × traits]
  matrix[N_obs,J] miss;      // Missing data indicator: 1 = missing, 0 = observed
  
  // tip_id maps observations to tip nodes in the tree
  // IMPORTANT: This indexes into the segment structure, where tip nodes
  // correspond to segments where tip[t,i] = 1. The R preprocessing ensures
  // tip_id[i] points to the correct segment index for observation i.
  array[N_obs] int<lower=1> tip_id;
  
  // ========================================================================
  // MODEL OPTIONS
  // ========================================================================
  int<lower=0,upper=1> prior_only; // 1 = sample from prior only (ignore likelihood)
  vector[J] y_mean;                // Sample means used for scaling (numerical stability)
}

// ============================================================================
// PARAMETERS BLOCK: Quantities Estimated via MCMC
// ============================================================================
// Parameters are organized into three groups:
// 1. SDE parameters: Control the evolutionary process (A, Q, b)
// 2. Phylogenetic parameters: Trait values along the tree (η, drift)
// 3. Observational parameters: Link latent to observed traits (Λ, α, φ)
// ============================================================================
parameters{
  // ========================================================================
  // 1. STOCHASTIC DIFFERENTIAL EQUATION PARAMETERS
  // ========================================================================
  // These define the latent evolutionary process: dη = (A·η + b)dt + G·dW
  
  // SELECTION MATRIX A
  // ------------------
  // A[i,j] represents the effect of latent trait j on the rate of change of
  // trait i. Diagonal elements A[i,i] must be negative for stationarity
  // (mean-reversion). In univariate OU, this corresponds to -α (selection
  // strength). Off-diagonal elements can be positive or negative.
  //
  // Interpretation: A[i,j] > 0 means trait j positively affects trait i;
  //                 A[i,j] < 0 means trait j negatively affects trait i.
  vector<upper=0>[N_latent] A_diag;           // Diagonal (autoregressive) terms
  vector[num_effects - N_latent] A_offdiag;  // Off-diagonal (cross-effect) terms
  
  // DRIFT COVARIANCE MATRIX Q
  // -------------------------
  // Q controls stochastic evolution (Brownian motion). In the SDE, G·dW where
  // Q = G·G'. The manuscript notation uses G as Cholesky factor of Q.
  //
  // Here we use a VARIANCE-CORRELATION DECOMPOSITION for better interpretability:
  //   Q = diag(σ) · R · diag(σ)
  // where σ are standard deviations and R is a correlation matrix.
  // This differs notationally from the manuscript but is mathematically equivalent.
  cholesky_factor_corr[N_latent] L_R;  // Cholesky factor of correlation matrix R
  vector<lower=0>[N_latent] Q_sigma;   // Standard deviations (√diag(Q))
  
  // CONTINUOUS-TIME INTERCEPTS b
  // ----------------------------
  // Along with A, these determine the equilibrium (stationary) trait values.
  // At equilibrium: A·η_eq + b = 0, so η_eq = -A⁻¹·b
  // Setting b = 0 means traits revert toward zero.
  vector[N_latent] b;
  
  // ========================================================================
  // 2. PHYLOGENETIC TRAIT EVOLUTION PARAMETERS
  // ========================================================================
  
  // ANCESTRAL STATE
  // ---------------
  // η at the root of the tree (t=0). One vector per tree to allow for
  // tree-specific ancestral reconstructions when marginalizing over trees.
  array[N_tree] vector[N_latent] eta_anc;
  
  // STOCHASTIC DRIFT ALONG BRANCHES
  // -------------------------------
  // NON-CENTERED PARAMETERIZATION (NCP): z_drift contains standard normal
  // values that are scaled by cholesky_decompose(Q_Δ) to produce drift with
  // the correct covariance. NCP often improves MCMC sampling efficiency,
  // especially when the data are informative about latent states.
  //
  // There are N_seg-1 drift terms (one per non-root segment).
  array[N_tree, N_seg - 1] vector[N_latent] z_drift;
  
  // TERMINAL DRIFT AT TIPS
  // ----------------------
  // Additional drift between the tip node value and each observation.
  // This serves multiple purposes:
  //   1. Captures residual variation not explained by the phylogeny
  //   2. Allows for measurement error / within-species variation
  //   3. Enables repeated measures per species (when N_obs > N_tips)
  //
  // Dimensions: [N_tree][N_obs, N_latent] - one matrix per tree
  array[N_tree] matrix[N_obs, N_latent] terminal_drift;
  
  // ========================================================================
  // 3. OBSERVATIONAL MODEL PARAMETERS
  // ========================================================================
  // These link latent traits η to observed data y via: g(μ) = Λ·η + α
  
  vector[J] alpha;              // Intercepts for each observed trait
  vector<lower=0>[J] shape;     // Shape parameters for Gamma distributions
  vector[J - N_latent] lambda_free; // Free elements of factor matrix Λ
}

// ============================================================================
// TRANSFORMED PARAMETERS BLOCK: Derived Quantities
// ============================================================================
// This block:
// 1. Constructs full A, Q, and Λ matrices from their parameterizations
// 2. Computes the asymptotic covariance Q_∞ (ONCE per iteration)
// 3. Propagates trait evolution along the entire phylogeny
//
// This is where the mathematical SDE solution is translated into computation.
// ============================================================================
transformed parameters{
  // Latent trait values at each node: eta[tree, segment] = K-vector
  array[N_tree, N_seg] vector[N_latent] eta;
  
  // ========================================================================
  // CONSTRUCT SELECTION MATRIX A
  // ========================================================================
  matrix[N_latent,N_latent] A = diag_matrix(A_diag);
  
  // ========================================================================
  // CONSTRUCT DRIFT COVARIANCE MATRIX Q
  // ========================================================================
  // Using variance-correlation decomposition: Q = diag(σ)·R·diag(σ)
  // where R = L_R·L_R' (correlation matrix from Cholesky factor)
  matrix[N_latent,N_latent] Q = diag_matrix(Q_sigma) * (L_R * L_R') * diag_matrix(Q_sigma);
  
  // ========================================================================
  // ASYMPTOTIC COVARIANCE Q_∞
  // ========================================================================
  // Computed ONCE per iteration via ksolve(), then used to derive Q_Δ for
  // each branch. This is the key computational efficiency of this approach.
  matrix[N_latent,N_latent] Q_inf;
  
  // ========================================================================
  // DRIFT COVARIANCE AT TIPS
  // ========================================================================
  // VCV_tips stores Q_Δ for terminal branches. For internal nodes, we use
  // a placeholder value (-99 diagonal matrix) that is never accessed.
  // This avoids conditional logic but requires careful indexing.
  array[N_tree, N_seg] matrix[N_latent,N_latent] VCV_tips;
  
  // ========================================================================
  // FACTOR LOADING MATRIX Λ (APPLICATION-SPECIFIC)
  // ========================================================================
  // Λ maps K latent traits to J observed traits: g(μ[j]) = Λ[j,·]·η + α[j]
  //
  // FOR THIS PRIMATE MODEL (J=4 observed, K=2 latent):
  //   Latent 1: "Body size" dimension
  //   Latent 2: "Brain allometry" dimension
  //
  //   Observed traits and their loadings:
  //   - Trait 1 (body mass):    Λ[1,1]=1, Λ[1,2]=0   → directly measures latent 1
  //   - Trait 2 (brain mass):   Λ[2,1]=0, Λ[2,2]=1   → directly measures latent 2
  //   - Trait 3 (longevity):    Λ[3,1]=λ₁, Λ[3,2]=0  → loads on latent 1
  //   - Trait 4 (maturity age): Λ[4,1]=λ₂, Λ[4,2]=0  → loads on latent 1
  //
  // Fixed loadings of 1.0 set the scale; free loadings (lambda_free) are estimated.
  matrix[J, N_latent] Lambda = rep_matrix(0.0, J, N_latent);
  Lambda[1, 1] = 1.0;              // Body mass → latent 1 (scale-setting)
  Lambda[3, 1] = lambda_free[1];  // Longevity → latent 1 (estimated)
  Lambda[4, 1] = lambda_free[2];  // Maturity → latent 1 (estimated)
  Lambda[2, 2] = 1.0;              // Brain mass → latent 2 (scale-setting)
  
  // Fill off-diagonal elements of A based on DAG structure (effects_mat)
  {
    int ticker = 1;
    for (i in 1:N_latent) {
      for (j in 1:N_latent) {
        if (i != j) {
          if (effects_mat[i,j] == 1) {
            A[i,j] = A_offdiag[ticker];
            ticker += 1;
          } else if (effects_mat[i,j] == 0) {
            A[i,j] = 0;  // Structural zero from DAG
          }
        }
      }
    }
  }
  
  // ========================================================================
  // SOLVE FOR ASYMPTOTIC COVARIANCE Q_∞
  // ========================================================================
  Q_inf = ksolve(A, Q);
  
  // ========================================================================
  // TREE TRAVERSAL: PROPAGATE TRAIT EVOLUTION
  // ========================================================================
  // For each tree, we traverse from root to tips, computing latent trait
  // values at each node using the SDE solution.
  //
  // SDE SOLUTION (Manuscript Section 3.1):
  // For time interval Δt from parent (time t₀) to child (time t):
  //
  //   η(t) = A_Δ·η(t₀) + b_Δ + ε
  //
  // where:
  //   A_Δ = exp(A·Δt)                    [matrix exponential]
  //   b_Δ = A⁻¹·(A_Δ - I)·b              [intercept contribution]
  //   ε ~ N(0, Q_Δ)                       [stochastic drift]
  //   Q_Δ = Q_∞ - A_Δ·Q_∞·A_Δ'           [drift covariance]
  //
  // INTERPRETATION:
  //   - A_Δ·η(t₀): Deterministic evolution from parent state
  //   - b_Δ: Pull toward equilibrium over the interval
  //   - ε: Random drift accumulated over the interval
  // ========================================================================
  for (t in 1:N_tree) {
    // Initialize root node with ancestral state
    eta[t, node_seq[t, 1]] = eta_anc[t];
    VCV_tips[t, node_seq[t, 1]] = diag_matrix(rep_vector(-99, N_latent)); // Placeholder
    
    // Traverse remaining segments (branches)
    for (i in 2:N_seg) {
      // ------------------------------------------------------------------
      // STEP 1: Matrix exponential A_Δ = exp(A·Δt)
      // ------------------------------------------------------------------
      // This captures cumulative deterministic selection over time Δt.
      // For small Δt: A_Δ ≈ I + A·Δt (first-order approximation)
      // For larger Δt: full matrix exponential captures nonlinear dynamics
      matrix[N_latent,N_latent] A_delta = matrix_exp(A * ts[t, i]);
      
      // ------------------------------------------------------------------
      // STEP 2: Drift covariance Q_Δ = Q_∞ - A_Δ·Q_∞·A_Δ'
      // ------------------------------------------------------------------
      // This is the variance of stochastic drift accumulated over Δt.
      // quad_form_sym(Q_inf, A_delta') efficiently computes A_Δ·Q_∞·A_Δ'
      matrix[N_latent,N_latent] VCV = Q_inf - quad_form_sym(Q_inf, A_delta');
      
      // ------------------------------------------------------------------
      // STEP 3: Sample stochastic drift (non-centered parameterization)
      // ------------------------------------------------------------------
      // Transform standard normal z_drift to have covariance Q_Δ:
      //   drift = chol(Q_Δ) · z_drift
      //
      // This is the NCP: z_drift ~ N(0,I), so drift ~ N(0, Q_Δ)
      vector[N_latent] drift_seg = cholesky_decompose(VCV) * z_drift[t, i-1];
      
      // ------------------------------------------------------------------
      // STEP 4: Compute trait value at child node
      // ------------------------------------------------------------------
      // Full solution: η_child = A_Δ·η_parent + b_Δ + drift
      //
      // The intercept contribution b_Δ = A⁻¹·(A_Δ - I)·b is computed as:
      //   A \ (A_Δ - I) · b
      // where \ is matrix left-division (solving A·x = (A_Δ-I)·b for x)
      // and add_diag(A_delta, -1) computes A_Δ - I
      
      if (tip[t, i] == 0) {
        // INTERNAL NODE: Include drift in trait value
        eta[t, node_seq[t, i]] = to_vector(
          A_delta * eta[t, parent[t, i]] + ((A \ add_diag(A_delta, -1)) * b) + drift_seg
        );
        VCV_tips[t, node_seq[t, i]] = diag_matrix(rep_vector(-99, N_latent)); // Placeholder
      } else {
        // TIP NODE: Compute expected value; terminal drift handled in model block
        // This separation allows the observational model to add species-specific
        // variation via terminal_drift, enabling repeated measures.
        eta[t, node_seq[t, i]] = to_vector(
          A_delta * eta[t, parent[t, i]] + ((A \ add_diag(A_delta, -1)) * b)
        );
        VCV_tips[t, node_seq[t, i]] = VCV;  // Store for terminal drift likelihood
      }
    }
  }
}

// ============================================================================
// MODEL BLOCK: Priors and Likelihood
// ============================================================================
model{
  // ========================================================================
  // PRIOR DISTRIBUTIONS
  // ========================================================================
  // Priors are weakly informative, centered at zero where appropriate.
  
  b ~ std_normal();  // Intercepts: N(0,1)
  
  for (t in 1:N_tree) {
    eta_anc[t][1] ~ std_normal();         // Ancestral body size: N(0,1), in this case corresponds to a prior that the ancestral state is around the mean of body size at the tips (exp(0) = 1, mean = 1 after scaling the data by the sample mean)
    eta_anc[t][2] ~ normal(-0.2, 0.15);   // Ancestral brain allometry: informative
    // NCP drift terms: standard normal (scaled in transformed parameters)
    for (i in 1:(N_seg - 1)) z_drift[t, i] ~ std_normal();
  }
  
  A_offdiag ~ std_normal();        // Cross-effects: N(0,1), centered at no effect
  A_diag ~ std_normal();           // Autoregression: N(0,1) truncated at 0 (upper=0)
  L_R ~ lkj_corr_cholesky(4);     // Correlation prior: LKJ(η=4), favors weaker correlations
  Q_sigma ~ std_normal();          // Drift SD: half-normal (lower=0 implied)
  alpha ~ std_normal();            // Observational intercepts: N(0,1)
  shape ~ gamma(0.01, 0.01);       // Gamma shape: weakly informative
  lambda_free ~ std_normal();      // Factor loadings: N(0,1)
  
  // ========================================================================
  // LIKELIHOOD: OBSERVATIONAL MODEL
  // ========================================================================
  // This implements: y[n,j] ~ f(μ[n,j], φ[j]) with g(μ) = Λ·η + α
  //
  // For each observation:
  //   1. Evaluate likelihood of terminal_drift given Q_Δ at that tip
  //   2. Compute latent trait value: η_tip = η_node + terminal_drift
  //   3. Transform to expected value via link function and Λ
  //   4. Evaluate likelihood of observed data given μ and shape
  //
  // TREE MARGINALIZATION:
  // When N_tree > 1, we marginalize over phylogenetic uncertainty by summing
  // likelihoods across trees: target += log_sum_exp(lp)
  // This computes log(Σ_t P(y|tree_t)), which with uniform prior over trees
  // gives the marginal likelihood P(y) = (1/N_tree) Σ_t P(y|tree_t).
  // ========================================================================
  if (!prior_only) {
    for (i in 1:N_obs) {
      vector[N_tree] lp;  // Log-likelihood for each tree
      
      for (t in 1:N_tree) {
        vector[N_latent] eta_tips;
        real beta;  // Allometric exponent for brain size
        
        // ----------------------------------------------------------------
        // TERMINAL DRIFT LIKELIHOOD
        // ----------------------------------------------------------------
        // terminal_drift[t][i,] ~ N(0, Q_Δ_tip)
        lp[t] = multi_normal_cholesky_lpdf(terminal_drift[t][i,] | 
                                           rep_vector(0.0, N_latent), 
                                           cholesky_decompose(VCV_tips[t, tip_id[i]]));
        
        // Compute latent trait at observation: η_tip = η_node + terminal_drift
        for (n in 1:N_latent) {
          eta_tips[n] = eta[t, tip_id[i]][n] + terminal_drift[t][i,n];
        }
        
        // ----------------------------------------------------------------
        // TRAIT 1: Body Mass (Gamma with log link)
        // ----------------------------------------------------------------
        // y[1] ~ Gamma(shape, rate) where rate = shape/μ, μ = exp(α + Λ·η)
        // Scaling by y_mean improves numerical stability for large values.
        lp[t] += gamma_lpdf(y[i,1]/y_mean[1] | 
                            shape[1], 
                            shape[1]/exp(alpha[1] + Lambda[1,1] * eta_tips[1]));
        
        // ----------------------------------------------------------------
        // TRAIT 2: Brain Mass (APPLICATION-SPECIFIC ALLOMETRIC MODEL)
        // ----------------------------------------------------------------
        // Standard allometric equation: brain = α · body^β
        // On log scale: log(brain) = log(α) + β·log(body)
        //
        // We model the allometric exponent β as evolving:
        //   β = softplus(η[2]) = log(1 + exp(η[2]))
        //
        // This ensures β > 0 (positive allometry expected for brain/body)
        // and allows β to vary across species via latent trait 2.
        //
        // The likelihood is: brain ~ Gamma(shape, shape/μ)
        //   where μ = exp(α[2]) · body^β = exp(α[2] + β·log(body))
        beta = log(1 + exp(Lambda[2,2] * eta_tips[2]));
        lp[t] += gamma_lpdf(y[i,2] | 
                            shape[2], 
                            shape[2]/exp(alpha[2] + log(y[i,1])*beta));
        
        // ----------------------------------------------------------------
        // TRAIT 3: Longevity (Gamma with log link, loads on latent 1)
        // ----------------------------------------------------------------
        if (miss[i,3] == 0) {
          lp[t] += gamma_lpdf(y[i,3]/y_mean[3] | 
                              shape[3], 
                              shape[3]/exp(alpha[3] + Lambda[3,1] * eta_tips[1]));
        }
        
        // ----------------------------------------------------------------
        // TRAIT 4: Age at Female Maturity (Gamma with log link, loads on latent 1)
        // ----------------------------------------------------------------
        if (miss[i,4] == 0) {
          lp[t] += gamma_lpdf(y[i,4]/y_mean[4] | 
                              shape[4], 
                              shape[4]/exp(alpha[4] + Lambda[4,1] * eta_tips[1]));
        }
      }
      
      // Marginalize over trees: log(Σ_t exp(lp[t]))
      target += log_sum_exp(lp);
    }
  }
}

// ============================================================================
// GENERATED QUANTITIES BLOCK: Posterior Predictive Checks
// ============================================================================
// Generates replicated data yrep from the posterior predictive distribution.
// If the model fits well, yrep should be statistically similar to observed y.
// ============================================================================
generated quantities{
  array[N_tree,N_obs,J] real yrep;  // Replicated observations
  matrix[N_latent,N_latent] cor_R;  // Drift correlation matrix (derived)
  
  // Reconstruct correlation matrix from Cholesky factor: R = L_R · L_R'
  cor_R = multiply_lower_tri_self_transpose(L_R);
  
  {
    for (i in 1:N_obs) {
      for (t in 1:N_tree) {
        vector[N_latent] terminal_drift_rep;
        vector[N_latent] eta_tips_rep;
        real beta_rep;
        
        // Sample new terminal drift from its distribution
        for (n in 1:N_latent) terminal_drift_rep[n] = normal_rng(0, 1);
        terminal_drift_rep = cholesky_decompose(VCV_tips[t, tip_id[i]]) * terminal_drift_rep;
        
        // Compute replicated latent trait value
        eta_tips_rep = eta[t, tip_id[i]] + terminal_drift_rep;
        
        // Generate replicated observations
        // Body mass
        yrep[t,i,1] = gamma_rng(shape[1], shape[1]/exp(alpha[1] + Lambda[1,1] * eta_tips_rep[1])) * y_mean[1];
        
        // Brain mass (allometric)
        beta_rep = log(1 + exp(Lambda[2,2] * eta_tips_rep[2]));
        yrep[t,i,2] = gamma_rng(shape[2], shape[2]/exp(alpha[2] + log(yrep[t,i,1])*beta_rep));
        
        // Longevity
        yrep[t,i,3] = gamma_rng(shape[3], shape[3]/exp(alpha[3] + Lambda[3,1] * eta_tips_rep[1])) * y_mean[3];
        
        // Maturity age
        yrep[t,i,4] = gamma_rng(shape[4], shape[4]/exp(alpha[4] + Lambda[4,1] * eta_tips_rep[1])) * y_mean[4];
      }
    }
  }
}
