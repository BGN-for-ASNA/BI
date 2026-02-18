# %%
# Dependencies
library(reticulate)
library(BayesianInference)
m <- importBI("cpu")
jnp <- import("jax.numpy")

pd <- import("pandas")
load("Personality Jackdaw/Data/Data.RData")

x <- data$N_att
r_to_jax <- function(x) {
  # Rebuild R array
  if (x$dim == 1) {
    x <- as.integer(x$x)
    return(x)
  } else {
    arr_r <- array(x$x, dim = x$dim)

    # R -> JAX with dtype
    jnp$array(arr_r, dtype = x$dtype)
  }
}

data <- lapply(data, r_to_jax)

# %%
# Model Definition -----------------------
## 1. T2R ---------------------
### 1.1. Priors for regression coefficients ('b_XXX' = random slopes, 'gamma_XXX' = fix slopes) ---------------------
a_t2r <- m$dist$normal(0, 1, sample = TRUE) # random intercept
b_urban <- m$dist$normal(0, 1, sample = TRUE)
b_enviro <- m$dist$normal(0, 1, sample = TRUE)
b_pers1 <- m$dist$normal(0, 1, sample = TRUE)
b_male <- m$dist$normal(0, 1, sample = TRUE)
gamma_brood <- m$dist$normal(0, 1, sample = TRUE)
gamma_age <- m$dist$normal(0, 1, sample = TRUE)
gamma_wing <- m$dist$normal(0, 1, sample = TRUE)
gamma_tarsus <- m$dist$normal(0, 1, sample = TRUE)
gamma_skull <- m$dist$normal(0, 1, sample = TRUE)

## 2 Aggression  ---------------------
### 2.1. Priors for regression coefficients ('b_XXX' = random slopes, 'gamma_XXX' = fix slopes)  ---------------------
a_bites <- m$dist$normal(0, 1, sample = TRUE)
b_agg_enviro <- m$dist$normal(0, 1, sample = TRUE)
b_agg_age <- m$dist$normal(0, 1, sample = TRUE)
b_agg_wing <- m$dist$normal(0, 1, sample = TRUE)
b_agg_tarsus <- m$dist$normal(0, 1, sample = TRUE)
b_agg_skull <- m$dist$normal(0, 1, sample = TRUE)

## 3. Random Effects (11)--------------------------------
### 0:RiskInt, 1:Urban, 2:Env, 3:Pers1, 4:Male, 5:RiskInt, 6:AggEnv, 7:AggAge, 8:AggWing, 9:AggTarsus, 10:AggSkull
sigma_u <- m$dist$exponential(1.0, sample = TRUE, shape = tuple(11L))
L_omega <- m$dist$lkj_cholesky(11L, concentration = 2.0, sample = TRUE)
L_Sigma <- jnp$matmul(jnp$diag(sigma_u), L_omega)
u_id <- m$dist$multivariate_normal(
  loc = jnp$stack(list(
    a_t2r,
    b_urban, b_enviro, b_pers1, b_male,
    a_bites,
    b_agg_enviro, b_agg_age, b_agg_wing, b_agg_tarsus, b_agg_skull
  )),
  scale_tril = L_Sigma, sample = TRUE, shape = tuple(data$N_id)
)


## 4. Likelihoods --------------------------------
### 4.1. T2R --------------------------------
mu_t2r <- (
  u_id[data$att_id, 0] + # random intercept
    u_id[data$att_id, 1] * data$urban_z + # random slope
    u_id[data$att_id, 2] * data$enviro_idx + # random slope
    u_id[data$att_id, 3] * data$pers1_idx + # random slope
    u_id[data$att_id, 4] * data$male_arrival + # random slope
    gamma_brood * data$brood_z +
    gamma_age * data$age_z +
    gamma_wing * data$wing_z +
    gamma_tarsus * data$tarsus_z +
    gamma_skull * data$skull_z
)

sigma_t2r <- m$dist$exponential(1.0, sample = TRUE)

m$dist$log_normal(mu_t2r, sigma_t2r, sample = TRUE, obs = data$t2r)

### 4.1. Aggression --------------------------------
logits_bites <-
  u_id[data$agg_id, 5] + # random intercept
  u_id[data$agg_id, 6] * data$t2r + # Agg env slope (none nested random factor, but as categorical fixed effect in the model)
  u_id[data$agg_id, 7] * data$agg_age_z + # Agg Age Slope
  u_id[data$agg_id, 8] * data$agg_wing_z + # Agg Wing Slope
  u_id[data$agg_id, 9] * data$agg_tarsus_z + # Agg Tarsus Slope
  u_id[data$agg_id, 10] * data$agg_skull_z # Agg Skull Slope


m$dist$binomial(total_count = 6L, logits = logits_bites, sample = TRUE, obs = data$bites) # for each test, birds got 6 trials

# %%
# Model function --------------------------------

model <- function(data) {
  # 1. Priors for regression coefficients ('b_XXX' = random slopes, 'gamma_XXX' = fix slopes) ---------------------
  a_t2r <- m$dist$normal(0, 1, name = "a_t2r") # random intercept
  b_urban <- m$dist$normal(0, 1, name = "b_urban")
  b_enviro <- m$dist$normal(0, 1, name = "b_enviro")
  b_pers1 <- m$dist$normal(0, 1, name = "b_pers1")
  b_male <- m$dist$normal(0, 1, name = "b_male")
  gamma_brood <- m$dist$normal(0, 1, name = "gamma_brood")
  gamma_age <- m$dist$normal(0, 1, name = "gamma_age")
  gamma_wing <- m$dist$normal(0, 1, name = "gamma_wing")
  gamma_tarsus <- m$dist$normal(0, 1, name = "gamma_tarsus")
  gamma_skull <- m$dist$normal(0, 1, name = "gamma_skull")

  # 2. Priors for regression coefficients ('b_XXX' = random slopes, 'gamma_XXX' = fix slopes)  ---------------------
  a_bites <- m$dist$normal(0, 1, name = "a_bites")
  b_agg_enviro <- m$dist$normal(0, 1, name = "b_agg_enviro")
  b_agg_age <- m$dist$normal(0, 1, name = "b_agg_age")
  b_agg_wing <- m$dist$normal(0, 1, name = "b_agg_wing")
  b_agg_tarsus <- m$dist$normal(0, 1, name = "b_agg_tarsus")
  b_agg_skull <- m$dist$normal(0, 1, name = "b_agg_skull")

  # 3. Random Effects (11)--------------------------------
  ### 0:RiskInt, 1:Urban, 2:Env, 3:Pers1, 4:Male, 5:RiskInt, 6:AggEnv, 7:AggAge, 8:AggWing, 9:AggTarsus, 10:AggSkull
  sigma_u <- m$dist$exponential(1.0, shape = tuple(11L), name = "sigma_u")
  L_omega <- m$dist$lkj_cholesky(11L, concentration = 2.0, sample = TRUE, name = "L_omega")
  L_Sigma <- jnp$matmul(jnp$diag(sigma_u), L_omega)
  u_id <- m$dist$multivariate_normal(
    loc = jnp$stack(list(
      a_t2r,
      b_urban, b_enviro, b_pers1, b_male,
      a_bites,
      b_agg_enviro, b_agg_age, b_agg_wing, b_agg_tarsus, b_agg_skull
    )),
    scale_tril = L_Sigma, shape = tuple(data$N_id), name = "u_id"
  )

  ## 4. Likelihoods --------------------------------
  ### 4.1. T2R --------------------------------
  mu_t2r <- (
    u_id[data$att_id, 0] + # random intercept
      u_id[data$att_id, 1] * data$urban_z + # random slope
      u_id[data$att_id, 2] * data$enviro_idx + # random slope
      u_id[data$att_id, 3] * data$pers1_idx + # random slope
      u_id[data$att_id, 4] * data$male_arrival + # random slope
      gamma_brood * data$brood_z +
      gamma_age * data$age_z +
      gamma_wing * data$wing_z +
      gamma_tarsus * data$tarsus_z +
      gamma_skull * data$skull_z
  )

  sigma_t2r <- m$dist$exponential(1.0, name = "sigma_t2r")

  m$dist$log_normal(mu_t2r, sigma_t2r, obs = data$t2r, name = "T2R")

  ### 4.1. Aggression --------------------------------
  logits_bites <-
    u_id[data$agg_id, 5] + # random intercept
    u_id[data$agg_id, 6] * data$t2r[data$agg_id] + # Agg env slope (none nested random factor, but as categorical fixed effect in the model)
    u_id[data$agg_id, 7] * data$agg_age_z + # Agg Age Slope
    u_id[data$agg_id, 8] * data$agg_wing_z + # Agg Wing Slope
    u_id[data$agg_id, 9] * data$agg_tarsus_z + # Agg Tarsus Slope
    u_id[data$agg_id, 10] * data$agg_skull_z # Agg Skull Slope

  m$dist$binomial(total_count = 6L, logits = logits_bites, obs = data$bites, name = "bites") # for each test, birds got 6 trials
}

m$data_on_model <- list(data = data)
m$fit(model)







# Estimating NA ----------------------------------

# def model(cafe, id, id_to_cafe, wait, afternoon, N_cafes, N_ids):
#
#  # Global fixed effects
#  a = m.dist.normal(5, 2, name="a")
#  b = m.dist.normal(-1, 0.5, name="b")

## Cafe-level covariance
# sigma_cafe = m.dist.exponential(1, shape=(2,), name="sigma_cafe")
# Rho_cafe = m.dist.lkj(2, 2, name="Rho_cafe")
# cov_cafe = jnp.outer(sigma_cafe, sigma_cafe) * Rho_cafe

# a_b_cafe = m.dist.multivariate_normal(
#  jnp.stack([a, b]),
#  cov_cafe,
#  shape=(N_cafes,),
#  name="a_b_cafe"
# )

# a_cafe = a_b_cafe[:, 0]
# b_cafe = a_b_cafe[:, 1]

## ID-level (nested) covariance
# sigma_id = m.dist.exponential(1, shape=(2,), name="sigma_id")
# Rho_id = m.dist.lkj(2, 2, name="Rho_id")
# cov_id = jnp.outer(sigma_id, sigma_id) * Rho_id

# a_b_id = m.dist.multivariate_normal(
#  jnp.stack([a_cafe[id_to_cafe], b_cafe[id_to_cafe]]),
#  cov_id,
#  shape=(N_ids,),
#  name="a_b_id"
# )

# a_id = a_b_id[:, 0]
# b_id = a_b_id[:, 1]

## Linear predictor
# mu = a_id[id] + b_id[id] * afternoon

# sigma = m.dist.exponential(1, name="sigma")
# m.dist.normal(mu, sigma, obs=wait)


# 4. Imputation for Missing Data --------------------------------
### AGE (Standardized)
# mu_age <- m$dist$normal(0, 1, sample = TRUE, name = "mu_age")
# sigma_age <- m$dist$exponential(1, sample = TRUE, name = "sigma_age")
## Use pre-calculated indices for static shape
# n_age_miss <- data$age_miss_idx$shape[0]
# age_missing <- m$dist$normal(mu_age, sigma_age, sample = TRUE, shape = tuple(n_age_miss), name = #"age_missing")
# age_complete <- data$age_z$at[data$age_miss_idx]$set(age_missing)
# m$dist$normal(mu_age, sigma_age, sample = TRUE, obs = age_complete, name = "age_obs")
# age_sq_complete <- age_complete**2
#
### WING (Standardized)
# mu_wing <- m$dist$normal(0, 1, sample = TRUE, name = "mu_wing")
# sigma_wing <- m$dist$exponential(1, sample = TRUE, name = "sigma_wing")
# n_wing_miss <- data$wing_miss_idx$shape[0]
# wing_missing <- m$dist$normal(mu_wing, sigma_wing, sample = TRUE, shape = tuple(n_wing_miss), name = #"wing_missing")
# wing_complete <- data$wing_z$at[data$wing_miss_idx]$set(wing_missing)
# m$dist$normal(mu_wing, sigma_wing, sample = TRUE, obs = wing_complete, name = "wing_obs")
# wing_sq_complete <- wing_complete**2
#
### TARSUS (Standardized)
# mu_tarsus <- m$dist$normal(0, 1, sample = TRUE, name = "mu_tarsus")
# sigma_tarsus <- m$dist$exponential(1, sample = TRUE, name = "sigma_tarsus")
# n_tarsus_miss <- data$tarsus_miss_idx$shape[0]
# tarsus_missing <- m$dist$normal(mu_tarsus, sigma_tarsus, sample = TRUE, shape = tuple(n_tarsus_miss), #name = "tarsus_missing")
# tarsus_complete <- data$tarsus_z$at[data$tarsus_miss_idx]$set(tarsus_missing)
# m$dist$normal(mu_tarsus, sigma_tarsus, sample = TRUE, obs = tarsus_complete, name = "tarsus_obs")
# tarsus_sq_complete <- tarsus_complete**2
#
### SKULL (Standardized)
# mu_skull <- m$dist$normal(0, 1, sample = TRUE, name = "mu_skull")
# sigma_skull <- m$dist$exponential(1, sample = TRUE, name = "sigma_skull")
# n_skull_miss <- data$skull_miss_idx$shape[0]
# skull_missing <- m$dist$normal(mu_skull, sigma_skull, sample = TRUE, shape = tuple(n_skull_miss), name = #"skull_missing")
# skull_complete <- data$skull_z$at[data$skull_miss_idx]$set(skull_missing)
# m$dist$normal(mu_skull, sigma_skull, sample = TRUE, obs = skull_complete, name = "skull_obs")
# skull_sq_complete <- skull_complete**2
#
### MALE_ARRIVAL (Approximated as Normal for Imputation)
# mu_male <- m$dist$normal(0, 1, sample = TRUE, name = "mu_male")
# sigma_male <- m$dist$exponential(1, sample = TRUE, name = "sigma_male")
# n_male_miss <- data$male_miss_idx$shape[0]
# male_missing <- m$dist$normal(mu_male, sigma_male, sample = TRUE, shape = tuple(n_male_miss), name = #"male_missing")
# male_complete <- data$male_arrival$at[data$male_miss_idx]$set(male_missing)
# m$dist$normal(mu_male, sigma_male, sample = TRUE, obs = male_complete, name = "male_obs")

print("Model Definition in R Complete.")
