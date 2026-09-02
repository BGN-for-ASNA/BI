data {
    int N_networktypes;                                               
    int N_id;           // Number of individuals                                                                                                 
    int N_dyads;        // Number of unique dyads
    int N_obs;          // Number of observations (edges/rows in the edgelist)
    int N_responses;        

    array[3] int N_params;                                          
    
    // --- Long format (edgelist) inputs ---
    array[N_obs] int sender;      // Index of sender (1 to N_id)
    array[N_obs] int receiver;    // Index of receiver (1 to N_id)
    array[N_obs] int dyad_id;     // Index of dyad (1 to N_dyads)
    array[N_obs] int dyad_dir;    // Direction identifier (1 or 2)

    array[N_obs] int outcomes;  
    vector[N_obs] outcomes_real; 
    array[N_obs] int exposure; 

    // --- Predictors ---
    matrix[N_id, N_params[1]] focal_set;
    matrix[N_id, N_params[2]] target_set;
    matrix[N_obs, N_params[3]] dyad_set; // Notice this is now [N_obs, K] to match edgelist

    matrix[23, 2] priors;
    
    int export_network;
    int outcome_mode;
    int link_mode;                       
}

transformed data {
    // Strip the intercept/padding column just like the original code
    matrix[N_id, N_params[1]-1] focal_individual_predictors; 
    matrix[N_id, N_params[2]-1] target_individual_predictors; 
    matrix[N_obs, N_params[3]-1] flat_dyad_preds; 

    if (N_params[1] > 1) {
        for(i in 2:N_params[1]) focal_individual_predictors[ , i-1] = focal_set[,i];  
    }
    if (N_params[2] > 1) {
        for(i in 2:N_params[2]) target_individual_predictors[ , i-1] = target_set[,i];  
    }
    if (N_params[3] > 1) {
        for(i in 2:N_params[3]) flat_dyad_preds[ , i-1] = dyad_set[,i];  
    }
}

parameters {
    matrix[1,1] B;

    // Sender-receiver effects (Declared as Matrix to avoid loops)
    vector<lower=0>[2] sr_sigma;  
    cholesky_factor_corr[2] sr_L;
    matrix[2, N_id] z_sr;

    // Dyadic effects (Declared as Matrix)
    real<lower=0> dr_sigma;     
    cholesky_factor_corr[2] dr_L;
    matrix[2, N_dyads] z_dr;

    //# Effects of covariate
    vector[N_params[1]-1] focal_effects;
    vector[N_params[2]-1] target_effects;
    vector[N_params[3]-1] dyad_effects;

    //# Error in Gaussian model
    real<lower=0> error_sigma;       
}

transformed parameters {
    matrix[2*N_responses, 2*N_responses] G_corr; 
    matrix[2*N_responses, 2*N_responses] D_corr; 

    G_corr = tcrossprod(sr_L); 
    D_corr = tcrossprod(dr_L);  
}

model {
    //# Priors on effects of covariates
    focal_effects ~ normal(priors[12,1], priors[12,2]);
    target_effects ~ normal(priors[13,1], priors[13,2]);
    dyad_effects ~ normal(priors[14,1], priors[14,2]);

    error_sigma ~ normal(priors[23,1], priors[23,2]);
    B[1,1] ~ normal(logit(priors[10,1]/sqrt(N_id)), priors[10,2]);

    //# Sender-receiver priors for social relations model (Vectorized)
    to_vector(z_sr) ~ std_normal();
    sr_sigma ~ normal(priors[15,1], priors[15,2]);    
    sr_L ~ lkj_corr_cholesky(priors[17,1]);

    //# Dyadic priors for social relations model (Vectorized)
    to_vector(z_dr) ~ std_normal();
    dr_sigma ~ normal(priors[16,1], priors[16,2]);
    dr_L ~ lkj_corr_cholesky(priors[18,1]);

    //------------------------------------------------------------------------//
    // MATRICES ALGEBRA (Matches NumPyro tensor operations)
    //------------------------------------------------------------------------//
    // Precompute scale and correlations
    matrix[N_id, 2] sr_eff = (diag_pre_multiply(sr_sigma, sr_L) * z_sr)';
    matrix[N_dyads, 2] dr_eff = (diag_pre_multiply(rep_vector(dr_sigma, 2), dr_L) * z_dr)';

    // Precalculate covariate effects
    vector[N_id] f_eff = rep_vector(0.0, N_id);
    if (N_params[1] > 1) f_eff = focal_individual_predictors * focal_effects;
    
    vector[N_id] t_eff = rep_vector(0.0, N_id);
    if (N_params[2] > 1) t_eff = target_individual_predictors * target_effects;
    
    vector[N_obs] d_eff = rep_vector(0.0, N_obs);
    if (N_params[3] > 1) d_eff = flat_dyad_preds * dyad_effects;

    // Compile entire linear predictor in long-format (No nested loops!)
    // Stan allows passing integer arrays to matrices/vectors to extract a vector
    vector[N_obs] linear_model = rep_vector(B[1,1], N_obs) 
                                 + f_eff[sender] 
                                 + t_eff[receiver] 
                                 + sr_eff[sender, 1] 
                                 + sr_eff[receiver, 2] 
                                 + d_eff;

    // Add direction-aware Dyadic Random Effects
    for (k in 1:N_obs) {
        linear_model[k] += dr_eff[dyad_id[k], dyad_dir[k]];
    }

    //------------------------------------------------------------------------//
    // LIKELIHOOD EVALUATION (Fully Vectorized)
    //------------------------------------------------------------------------//
    if (outcome_mode == 1) {
        if (link_mode == 1) outcomes ~ bernoulli_logit(linear_model);
        if (link_mode == 2) outcomes ~ bernoulli(Phi(linear_model));
    } else if (outcome_mode == 2) {
        if (link_mode == 1) outcomes ~ binomial_logit(exposure, linear_model);
        if (link_mode == 2) outcomes ~ binomial(exposure, Phi(linear_model));
    } else if (outcome_mode == 3) {
        outcomes ~ poisson_log(linear_model);
    } else if (outcome_mode == 4) {
        outcomes_real ~ normal(linear_model, error_sigma);
    }
}

generated quantities {
    // Output predictions as an edgelist matching NumPyro's 1D deterministic array
    vector[export_network == 1 ? N_obs : 0] p;

    if (export_network == 1) {
        matrix[2, N_id] sr_base = diag_pre_multiply(sr_sigma, sr_L) * z_sr;
        matrix[2, N_dyads] dr_base = diag_pre_multiply(rep_vector(dr_sigma, 2), dr_L) * z_dr;

        for (k in 1:N_obs) {
            real f_val = 0.0;
            real t_val = 0.0;
            real d_val = 0.0;
            
            if (N_params[1] > 1) f_val = dot_product(focal_effects, to_vector(focal_individual_predictors[sender[k]]));
            if (N_params[2] > 1) t_val = dot_product(target_effects, to_vector(target_individual_predictors[receiver[k]]));
            if (N_params[3] > 1) d_val = dot_product(dyad_effects, to_vector(flat_dyad_preds[k]));

            real lp = B[1,1] + f_val + t_val 
                      + sr_base[1, sender[k]] + sr_base[2, receiver[k]] 
                      + d_val + dr_base[dyad_dir[k], dyad_id[k]];

            if (outcome_mode == 1 || outcome_mode == 2) {
                p[k] = (link_mode == 1) ? inv_logit(lp) : Phi(lp);
            } else if (outcome_mode == 3) {
                p[k] = exp(lp);
            } else if (outcome_mode == 4) {
                p[k] = lp;
            }
        }
    }
}