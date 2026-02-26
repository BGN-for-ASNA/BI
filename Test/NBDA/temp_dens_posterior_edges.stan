//stan
data {
    int<lower=0> K;                // Number of trials
    int<lower=0> Q;                // Number of individuals in each trial
    int<lower=1> P;                // Number of unique individuals
    array[K] int<lower=0> N;       // Number of individuals that learned during observation period
    array[K] int<lower=0> N_c;     // Number of right-censored individuals
    array[K, Q] int<lower=-1> ind_id; // IDs of individuals
    array[K] int<lower=1> T;       // Maximum time periods
    int<lower=1> T_max;            // Max timesteps reached
    array[K,P] int t;     // Time of acquisition for each individual
    array[K, T_max] real<lower=0> D; // Scaled durations
    int<lower=1> N_networks;
    array[K] matrix[T_max, P] Z;   // Knowledge state * cue matrix
    array[K] matrix[T_max, P] Zn;   // Knowledge state
    int<lower=0> N_veff;
    int N_dyad;  // number of dyads
    matrix[N_networks, N_dyad] logit_edge_mu;  // logit edge values
    array[N_networks] matrix[N_dyad, N_dyad] logit_edge_cov;  // covariance matrix
    array[N_dyad] int<lower=1> focal_ID;
    array[N_dyad] int<lower=1> other_ID;
}
parameters {
    matrix[N_networks, N_dyad] edge_logit;
    real log_lambda_0_mean;  // Log baseline learning rate
    real log_s_prime_mean;
}
transformed parameters {
    real s_prime;
    real<lower=0> lambda_0;
    s_prime = exp(log_s_prime_mean);
    lambda_0 = exp(log_lambda_0_mean);
    array[N_networks] matrix[P, P] A;
    for (network in 1:N_networks) {
        A[network] = rep_matrix(0, P, P);
    }
    for (network in 1:N_networks){
        for (edge_idx in 1:N_dyad) {
            real w = inv_logit(edge_logit[network, edge_idx]);
            A[network, focal_ID[edge_idx], other_ID[edge_idx]] = w;
            A[network, other_ID[edge_idx], focal_ID[edge_idx]] = w;
        }
    }
}
model {
    log_lambda_0_mean ~ normal(-4, 2);
    log_s_prime_mean ~ normal(-4, 2);
    for (n in 1:N_networks) {
        edge_logit[n] ~ multi_normal(logit_edge_mu[n], logit_edge_cov[n]);
    }
    for (trial in 1:K) {
        for (n in 1:N[trial]) {
            int id = ind_id[trial, n];
            int learn_time = t[trial, id];
            if (learn_time > 0) {
                for (time_step in 1:learn_time) {
                    real ind_term = 1.0;
                    real net_effect = 0;
                    for (network in 1:N_networks) {
                        net_effect += s_prime * dot_product(A[network][id, ],Z[trial][time_step, ]);
                    }
                    real soc_term = net_effect;
                    real lambda =  (lambda_0 * ind_term + soc_term) * D[trial, time_step] ;
                    target += -lambda;
                    if (time_step == learn_time) {
                        target += log( (lambda_0 * ind_term + soc_term));
                    }
                }
            }
        }
        if (N_c[trial] > 0) {
            for (c in 1:N_c[trial]) {
                int id = ind_id[trial, N[trial] + c];
                for (time_step in 1:T[trial]) {
                    real ind_term = 1.0;
                    real net_effect = 0;
                    for (network in 1:N_networks) {
                        net_effect += s_prime * dot_product(A[network][id, ],Z[trial][time_step, ]);
                    }
                    real soc_term = net_effect;
                    real lambda =  (lambda_0 * ind_term + soc_term) * D[trial, time_step] ;
                    target += -lambda;
                }
            }
        }
    }
}
