data {
    int K;              #// num options(gnomes)
    int N;              #// num observations
    int J;              #// num individuals
    int tech[N];        #// technique chosen
    real y[N,K];        #// observed personal yields of techs 1-K
    real ps[N,K];       #// observed payoff social variables of techs 1-K
    real s[N,K];        #// observed number of ttimes observing behaviors
    real ks[N,K];       #// observed if kin
    real press[N,K];    #// observed if alpha male or female
    real cohos[N,K];    #// similarity cohort bias        
    real yobs[N,K];     #// yob prestige bias
    int bout[N];        #// bout or fruit
    int id[N];          #// player id
    int N_effects;      #// number of learning parameters to estimate
    real age[N];        #//mono age
}
parameters {
    real<lower=0> lambda;                   #// mutlinomial error parameter
    vector[N_effects] mu;                   #// average effects
    matrix[N_effects,J] zed;                #// individual z-scores
    cholesky_factor_corr[N_effects] L_Rho;  #// correlation matrix
    vector<lower=0>[N_effects] sigma;       #// standard deviations
    vector[2] b_age;                        #// slope for effect of age

}
transformed parameters{
    matrix[J,N_effects] a_id;
    #// put scales and correlations back in
    a_id = (diag_pre_multiply(sigma,L_Rho) * zed)';
}
model {
    vector[K] AC;        #// attraction scores
    real logPrA;         #// individual learning temp
    real PrS;            #// social learning temp
    vector[K] s_temp;        
    vector[K] lin_mod;
    real phi;            #// stickiness parameter
    real gamma;          #// social weight
    real Bpay;           #// payoff exponent
    real fconf;          #// conform exponent
    real Bkin;           #// kin exponent
    real Bpres;          #// alpha rank exponent
    real Bcoho;          #// cohort/age-similartiy bias exponent
    real Byob;           #// age-bias exponent

    #//priors
    lambda ~ exponential(1);
    mu ~ normal(0,1);
    sigma ~ exponential(3);
    to_vector(zed) ~ normal(0,1);
    L_Rho ~ lkj_corr_cholesky(3);
    b_age ~ normal(0,1);


    for ( i in 1:N ) {
        #//update attraction scores
        for ( j in 1:K ) {
            if ( bout[i] > 1 ) {
                AC[j] = (1-phi)*AC[j] + phi*y[i-1,j];
            } else {
                AC[j] = 0;
            }
        }#//j

        if ( bout[i]==1 ) {
            #// calculate new individual's parameter values
            phi = inv_logit( mu[1] + a_id[id[i],1]  + b_age[1]*age[i] );  //attraction to new experience
            gamma = inv_logit( mu[2] + a_id[id[i],2] + b_age[2]*age[i] ); //weight given to social info
            fconf = exp( mu[3] + a_id[id[i],3] );                         // strength of frequency dependence
            Bpay = ( mu[4] + a_id[id[i],4] );                             //contribution of payoff bias
            Bkin = ( mu[5] + a_id[id[i],5] );                             //contribution of matrlineal kin bias
            Bpres = ( mu[6] + a_id[id[i],6] );                            //contribution of rank bias
            Bcoho = ( mu[7] + a_id[id[i],7] );                            //contribution of age-cohort bias
            Byob = ( mu[8] + a_id[id[i],8] );                             //contribution of age bias 
        }

        logPrA = lambda*AC[tech[i]] - log_sum_exp( lambda*AC );           // calculate individual learning prob

        #//conformity aspect below
        if ( bout[i] > 1 ) {
            if (sum( s[i] ) > 0 ) {                                       //only use social info when it is observed

                #// compute non-frequency cues as log-linear model
                for ( j in 2:K ) {
                    lin_mod[j] = exp( Bpay*ps[i,j] + Bkin*ks[i,j] + Bpres*press[i,j] + Bcoho*cohos[i,j] + Byob*yobs[i,j] );
                }
                lin_mod[1] = 1; #// aliased outcome

                #// compute frequency cue
                for ( j in 1:K ) s_temp[j] = pow(s[i,j],fconf);
                for ( j in 1:K ) lin_mod[j] = lin_mod[j] * s_temp[j];

                PrS = lin_mod[tech[i]]/sum(lin_mod);
                
                target += ( log( (1-gamma)*exp(logPrA) + gamma*PrS ) );

            } else {
                target += ( logPrA );
            }
        } else {
            target += ( logPrA );
         }
     }#//i  

}#//end of model

generated quantities{
    real dev;            #//deviance to calc DIC
    vector[N] log_lik;   #//log-likelihood to calc WAIC with compare fcn 

    vector[K] AC;        #// attraction scores
    real logPrA;         #// individual learning temp
    real PrS;            #// social learning temp
    vector[K] s_temp;        
    vector[K] lin_mod;
    real phi;            #// stickiness parameter
    real gamma;          #// social weight
    real Bpay;           #// payoff exponent
    real fconf;          #// conform exponent
    real Bkin;           #// kin exponent
    real Bpres;          #// alpha rank exponent
    real Bcoho;          #// cohort bias exponent
    real Byob;           #// age-bias exponent
    matrix[N_effects,N_effects] Rho;
    vector[N_effects] Sigma;

    Sigma = sigma;
    Rho = L_Rho * L_Rho';

    dev=0;
    for ( i in 1:N ) {
        #//update attractions
        for ( j in 1:K ) {
            if ( bout[i] > 1 ) {
                AC[j] = (1-phi)*AC[j] + phi*y[i-1,j];
            } else {
                AC[j] = 0;
            }
        }#//j

        if ( bout[i]==1 ) {
            #// calculate new individual's parameter values
            phi = inv_logit( mu[1] + a_id[id[i],1]  + b_age[1]*age[i] );
            gamma = inv_logit( mu[2] + a_id[id[i],2] + b_age[2]*age[i] );
            fconf = exp( mu[3] + a_id[id[i],3] );
            Bpay = ( mu[4] + a_id[id[i],4] );
            Bkin = ( mu[5] + a_id[id[i],5] );
            Bpres = ( mu[6] + a_id[id[i],6] );
            Bcoho = ( mu[7] + a_id[id[i],7] );
            Byob = ( mu[8] + a_id[id[i],8] );
        }

        logPrA = lambda*AC[tech[i]] - log_sum_exp( lambda*AC );

        #//conformity aspect below
        if ( bout[i] > 1 ) {
            if (sum( s[i] ) > 0 ) {

                #// compute non-frequency cues as log-linear model
                for ( j in 2:K ) {
                    lin_mod[j] = exp( Bpay*ps[i,j] + Bkin*ks[i,j] + Bpres*press[i,j] + Bcoho*cohos[i,j] + Byob*yobs[i,j] );
                }
                lin_mod[1] = 1; #// aliased outcome

                #// compute frequency cue
                for ( j in 1:K ) s_temp[j] = pow(s[i,j],fconf);
                for ( j in 1:K ) lin_mod[j] = lin_mod[j] * s_temp[j];
                PrS = lin_mod[tech[i]]/sum(lin_mod);
                
                dev = dev + -2*( log( (1-gamma)*exp(logPrA) + gamma*PrS ) );
                log_lik[i] = ( log( (1-gamma)*exp(logPrA) + gamma*PrS ) ) ;  

            } else {
                 dev = dev + -2*( logPrA );
                 log_lik[i] = (logPrA);
            }
        } else {
                 dev = dev + -2*( logPrA );
                 log_lik[i] = (logPrA);           }
     }#//i  
}#//end of model