##############################
#### Tested on linux
##############################
#%%
import sys
import os
sys.path.insert(0, "/home/sebastian_sosa/BF/BayesForge/Network")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import (
    ListVector,
    FloatVector,
    FloatMatrix,
    IntVector,
    IntMatrix,
    StrVector,
)
from rpy2.rinterface import NULLType
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax import vmap
from functools import partial
from model_effects import Neteffect
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from cmdstanpy import CmdStanModel

m = bf("cpu", cores=10)  # 10 divides N=50; locks JAX at 10 virtual devices for the whole script


# %% Run R simulation and NumPyro fit
with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
    ro.r(
        """
library(posterior)
library(igraph)
library(STRAND)
library(ggplot2)

set.seed(1)

# Make data
N_id = 50

# Covariates
Kinship = standardize_strand(rlkjcorr( 1 , N_id , eta=1.5 ))
Dominant = matrix(rbern(N_id*N_id), 0.2, nrow=N_id, ncol=N_id)
Mass = rbern(N_id, 0.4)
Age = rnorm(N_id, 0, 1)
Love = rbern(N_id, 0.4)
Fire = rbern(N_id, 0.6)

# Organize into list
dyadic_preds = array(NA,c(N_id,N_id,3))
dyadic_preds[,,1] = Kinship
dyadic_preds[,,2] = Dominant
dyadic_preds[,,3] = Kinship*Dominant

# Set effect sizes
sr_mu = c(0,0)  
sr_sigma = c(2.2, 1.7) 
sr_rho = -0.7
dr_mu = 0 
dr_sigma = 3.5
dr_rho= 0.8
sr_effects_1 = c(2, 1.5)
sr_effects_2 = c(-1.4, 1.0)
sr_effects_3 = c(1.4, -1.3)
sr_effects_4 = c(-0.9, 0)
dr_effects_1 = c(1.5, -1.9, 2.4)

# Block structure
group_probs_block_size = c(0.25, c(0.25, 0.25)*(1-0.25))

B_1 = matrix(-8,nrow=1,ncol=1)
B_2 = matrix(rnorm(9,0,3),nrow=3,ncol=3)

diag(B_2) = diag(B_2) + 3.5

B = list(B_1, B_2)

groups_1 = rep("Any",N_id) 
groups_2 = sample( c("Red","White","Blue") , size=N_id , replace=TRUE , prob=group_probs_block_size )

groups_f = data.frame(Intercept=factor(groups_1), Merica=factor(groups_2))
individual = data.frame(Mass = Mass, Age = Age, Love = Love, Fire = Fire)

G = simulate_sbm_plus_srm_network(N_id = N_id, 
                                  B = B, 
                                  V = 2,
                                  groups=data.frame(Intercept=as.numeric(factor(groups_1)), Merica=as.numeric(factor(groups_2))),                  
                                  sr_mu = sr_mu,  
                                  sr_sigma = sr_sigma, 
                                  sr_rho = sr_rho,
                                  dr_mu = dr_mu,  
                                  dr_sigma = dr_sigma, 
                                  dr_rho = dr_rho,
                                  error_sigma = 1.0,
                                  outcome_mode="bernoulli", 
                                  link_mode="logit",                 
                                  individual_predictors = individual,
                                  dyadic_predictors = dyadic_preds,
                                  individual_effects = cbind(sr_effects_1, sr_effects_2, sr_effects_3, sr_effects_4),
                                  dyadic_effects = dr_effects_1
)

name_vec = paste("Individual", 1:N_id)
rownames(G$network) = colnames(G$network) = name_vec
rownames(G$samps) = colnames(G$samps) = name_vec
rownames(Kinship) = colnames(Kinship) = name_vec
rownames(Dominant) = colnames(Dominant) = name_vec
rownames(groups_f) = name_vec
rownames(individual) = name_vec

model_dat = make_strand_data(outcome=list(Association = G$network),  
                             block_covariates=groups_f, 
                             individual_covariates=individual, 
                             dyadic_covariates=list(Kinship=Kinship, Dominant=Dominant),  
                             outcome_mode="bernoulli", 
                             link_mode="logit",
                             exposure=list(Association = G$samps)
)

fit_numpyro = fit_block_plus_social_relations_model(
  data=model_dat,
  block_regression = ~ Merica,
  focal_regression = ~ Mass + Age + Love + Fire,
  target_regression = ~ Mass + Age + Love + Fire,
  dyad_regression = ~ Kinship*Dominant,
  mode="numpyro",
  mcmc_parameters = list(seed = 1, chains = 1, iter_warmup = 1000, iter_sampling = 1000)
)

res_stan = summarize_strand_results(fit_numpyro)
# Merge SRM samples and Block samples
strand_posteriors = c(res_stan$samples$srm_model_samples, res_stan$samples$block_parameters)
save(fit_numpyro, model_dat, file = 'STRAND.Rdata')

# Regenerate exact arrays for Python loading
d    = fit_numpyro$data
N_id = d$N_id
locs = make_dyadic_edgelist(N_id)

long_focal_set  = focal_set_to_long(as.matrix(d$focal_set))
long_target_set = target_set_to_long(as.matrix(d$target_set))
long_dyad_set   = dyadic_set_to_long(d$dyad_set)

# For blocks
block_dat = block_set_to_dyadic_block_set(d$block_set, priors=d$priors)
long_block_set = dyadic_set_to_long(block_dat$Y)
block_mu       = block_dat$Mu
block_sigma    = block_dat$Sigma

# For outcomes and exposure
out_mat = if(is.list(d$outcomes)) d$outcomes[[1]] else d$outcomes
exp_mat = if(is.list(d$exposure)) d$exposure[[1]] else d$exposure
if(length(dim(out_mat)) == 3) out_mat = out_mat[,,1]
if(length(dim(exp_mat)) == 3) exp_mat = exp_mat[,,1]

long_outcome = array(NA, c(nrow(locs), 2, 1))
long_outcome[,1,1] = out_mat[as.matrix(locs[,c("Var2","Var1")])]
long_outcome[,2,1] = out_mat[as.matrix(locs[,c("Var1","Var2")])]

long_exposure = array(NA, c(nrow(locs), 2, 1))
long_exposure[,1,1] = exp_mat[as.matrix(locs[,c("Var2","Var1")])]
long_exposure[,2,1] = exp_mat[as.matrix(locs[,c("Var1","Var2")])]

# --- Wide-format arrays ---
wide_outcome  = out_mat
wide_exposure = exp_mat
wide_focal_mat  = as.matrix(d$focal_set)[, -1, drop=FALSE]
wide_target_mat = as.matrix(d$target_set)[, -1, drop=FALSE]
raw_dyad_list = d$dyad_set[-1]
if (length(raw_dyad_list) > 0) {
  wide_dyad_mat = simplify2array(lapply(raw_dyad_list, as.matrix))
} else {
  wide_dyad_mat = array(0, dim=c(N_id, N_id, 0))
}
Any    = as.integer(as.factor(groups_1)) - 1L
Merica = as.integer(as.factor(groups_2)) - 1L

save(long_focal_set, long_target_set, long_dyad_set, long_block_set, 
     block_mu, block_sigma, long_outcome, long_exposure, locs, 
     wide_outcome, wide_exposure, wide_focal_mat, wide_target_mat, wide_dyad_mat,
     Any, Merica,
     file = 'strand_arrays.RData')
"""
    )


# %% Python side: Load and prepare data
ro.r["load"]("strand_arrays.RData")
with localconverter(ro.default_converter + numpy2ri.converter):
    long_focal_np = np.array(ro.globalenv["long_focal_set"])
    long_target_np = np.array(ro.globalenv["long_target_set"])
    long_dyad_np = np.array(ro.globalenv["long_dyad_set"])
    long_block_np = np.array(ro.globalenv["long_block_set"])
    block_mu_np = np.array(ro.globalenv["block_mu"])
    block_sigma_np = np.array(ro.globalenv["block_sigma"])
    long_outcome_np = np.array(ro.globalenv["long_outcome"])
    long_exposure_np = np.array(ro.globalenv["long_exposure"])
    locs_raw = np.array(ro.globalenv["locs"])
    wide_focal_np = np.array(ro.globalenv["wide_focal_mat"])
    wide_target_np = np.array(ro.globalenv["wide_target_mat"])
    wide_dyad_np = np.array(ro.globalenv["wide_dyad_mat"])
    Any_np = np.array(ro.globalenv["Any"]).astype(int).ravel()
    Merica_np = np.array(ro.globalenv["Merica"]).astype(int).ravel()

# Determine expected number of dyads to handle rpy2's unpredictable array transpositions
N_id_r = int(np.array(ro.globalenv["N_id"])[0])
N_dyads_expected = N_id_r * (N_id_r - 1) // 2

if long_focal_np.shape[0] != N_dyads_expected:
    long_focal_np = long_focal_np.transpose(2, 1, 0)
if long_block_np.shape[0] != N_dyads_expected:
    long_block_np = long_block_np.transpose(2, 1, 0)
if long_outcome_np.shape[0] != N_dyads_expected:
    long_outcome_np = long_outcome_np.transpose(2, 1, 0)
if long_exposure_np.shape[0] != N_dyads_expected:
    long_exposure_np = long_exposure_np.transpose(2, 1, 0)


if locs_raw.dtype.names is not None:
    locs_2d = np.stack([locs_raw[name] for name in ["Var2", "Var1"]], axis=1)
else:
    locs_2d = locs_raw.T if locs_raw.shape[0] == 2 else locs_raw
long_ids_int = jnp.array(locs_2d) - 1

long_focal_set = jnp.array(long_focal_np[:, :, 1:])
long_target_set = jnp.array(long_target_np[:, :, 1:])
long_dyad_set = jnp.array(long_dyad_np[:, :, 1:])
long_block_set = jnp.array(long_block_np)
network = jnp.array(long_outcome_np[:, :, 0])
exposure = jnp.array(long_exposure_np[:, :, 0])
N_nodes = int(jnp.max(long_ids_int) + 1)
N_dyads = int(long_ids_int.shape[0])

wide_network_edgl = jnp.array(long_outcome_np[:, :, 0])
wide_dyad_edgl = long_dyad_set
wide_sender_preds = jnp.array(wide_focal_np)
wide_receiver_preds = jnp.array(wide_target_np)

Any = jnp.array(Any_np, dtype=jnp.int32)
Merica = jnp.array(Merica_np, dtype=jnp.int32)

_, N_per_grp_Any = jnp.unique(Any, return_counts=True)
_, N_per_grp_Merica = jnp.unique(Merica, return_counts=True)
N_grp_Any = int(N_per_grp_Any.shape[0])
N_grp_Merica = int(N_per_grp_Merica.shape[0])

# Matrix-form data for NeteffectMatrix models
Y_mat_jnp = NeteffectMatrix.edgelist_to_matrix_outcome(wide_network_edgl, N_nodes)
dyadic_preds_mat_jnp = NeteffectMatrix.edgelist_to_matrix_predictors(wide_dyad_edgl, N_nodes)
mask_mat = (1 - jnp.eye(N_nodes)).astype(jnp.float32)
CORES_SHARD = 10  # N=50 divisible by 10; JAX already initialized with 10 devices above

# --- Block Priors: use strand_block_prior() to auto-slice STRAND's flat vectors ---
# This replaces all manual reshaping and ensures identical priors across N_id values.
# (Removed manual injection as block_model now programmatically aligns with STRAND)


# %% Model definitions
def model_srm_long_block(
    network,
    long_dyad_set,
    long_focal_set,
    long_target_set,
    long_block_set,
    long_ids_int,
    exposure,
    sample=False,
):
    m = bf()
    N_var_focal, N_var_target, N_var_dyad, N_var_block = (
        long_focal_set.shape[2],
        long_target_set.shape[2],
        long_dyad_set.shape[2],
        long_block_set.shape[2],
    )
    block_effects = m.dist.normal(
        loc=block_mu_np,
        scale=block_sigma_np,
        shape=(N_var_block,),
        sample=sample,
        name="block_effects",
    )
    focal_effects = m.dist.normal(
        loc=jnp.zeros(N_var_focal),
        scale=2.5,
        shape=(N_var_focal,),
        sample=sample,
        name="focal_effects",
    )
    target_effects = m.dist.normal(
        loc=jnp.zeros(N_var_target),
        scale=2.5,
        shape=(N_var_target,),
        sample=sample,
        name="target_effects",
    )
    dyad_effects = m.dist.normal(
        loc=jnp.zeros(N_var_dyad),
        scale=2.5,
        shape=(N_var_dyad,),
        sample=sample,
        name="dyad_effects",
    )
    sr_L = m.dist.lkj_cholesky(2, 2.5, name="sr_L", sample=sample)
    sr_sigma = m.dist.truncated_normal(
        loc=0.0, scale=2.5, low=0.0, shape=(2,), sample=sample, name="sr_sigma"
    )
    sr_raw = m.dist.normal(
        loc=jnp.zeros((2, N_nodes)),
        scale=1.0,
        shape=(2, N_nodes),
        sample=sample,
        name="sr_raw",
    )
    sr = jnp.transpose(jnp.matmul(jnp.diag(sr_sigma) @ sr_L, sr_raw))
    dr_L = m.dist.lkj_cholesky(2, 2.5, name="dr_L", sample=sample)
    dr_sigma = m.dist.truncated_normal(
        loc=0.0, scale=2.5, low=0.0, shape=(), sample=sample, name="dr_sigma"
    )
    dr_raw = m.dist.normal(
        loc=jnp.zeros((2, N_dyads)),
        scale=1.0,
        shape=(2, N_dyads),
        sample=sample,
        name="dr_raw",
    )
    dr = jnp.transpose(
        jnp.matmul(jnp.expand_dims(jnp.repeat(dr_sigma, 2), 1) * dr_L, dr_raw)
    )
    mu = (
        jnp.tensordot(long_block_set, block_effects, axes=1)
        + jnp.tensordot(long_focal_set, focal_effects, axes=1)
        + jnp.tensordot(long_target_set, target_effects, axes=1)
        + jnp.tensordot(long_dyad_set, dyad_effects, axes=1)
    )
    S_i, R_j, S_j, R_i = (
        sr[long_ids_int[:, 0], 0],
        sr[long_ids_int[:, 1], 1],
        sr[long_ids_int[:, 1], 0],
        sr[long_ids_int[:, 0], 1],
    )
    gr_long = jnp.stack([S_i + R_j, S_j + R_i], axis=1)
    m.dist.binomial(
        jnp.ones_like(network), logits=mu + dr + gr_long, obs=network, name="network"
    )


def model_srm_wide(
    network_edgl, dyadic_predictors, sender_predictors, receiver_predictors, Any, Merica
):
    """Wide-format SRM using the programmatically aligned block_model API."""
    m2_inner = bf()
    # Now calls the updated internal logic which matches STRAND exactly
    B_any = Neteffect.block_model(Any, N_grp_Any, N_per_grp_Any, name="intercept")
    B_Merica = Neteffect.block_model(
        Merica, N_grp_Merica, N_per_grp_Merica, name="Merica"
    )
    sr = m2_inner.net.sender_receiver(sender_predictors, receiver_predictors)
    dr = m2_inner.net.dyadic_effect(dyadic_predictors)
    m2_inner.dist.bernoulli(
        logits=B_any + B_Merica + sr + dr, obs=network_edgl, name="network_edgl"
    )


def model_srm_matrix(
    network_edgl, dyadic_predictors_mat, sender_predictors, receiver_predictors, Any, Merica
):
    """Matrix-form SRM (no sharding): outer-sum SR + (N,N,K) dyadic tensordot.

    Functionally identical to model_srm_wide but all effects live in (N,N) space.
    Likelihood is on the edgelist (mat_to_edgl) so the obs shape matches BF wide.
    """
    B_any = NeteffectMatrix.block_model(Any, N_grp_Any, N_per_grp_Any, name="intercept")
    B_Merica = NeteffectMatrix.block_model(
        Merica, N_grp_Merica, N_per_grp_Merica, name="Merica"
    )
    SR_mat = m3.net2.sender_receiver(sender_predictors, receiver_predictors)
    D_mat = NeteffectMatrix.dyadic_effect(dyadic_predictors_mat)
    logits_edgl = m3.net.mat_to_edgl(B_any + B_Merica + SR_mat + D_mat)
    m3.dist.bernoulli(logits=logits_edgl, obs=network_edgl, name="network_edgl")


def model_srm_matrix_sharded(Y_mat, dyadic_predictors_mat, sender_predictors):
    """Matrix-form SRM (sharded): axis-0 sharding of Y_mat and dyadic preds.

    Only Y_mat, dyadic_predictors_mat, sender_predictors are in data_on_model
    so _auto_shard_data shards exactly those three arrays along axis 0.
    receiver_predictors, Any, Merica are captured from closure (replicated on all
    devices) because they are needed in their full-N form for the column dimension
    of the outer sum and block lookup.

    Likelihood is in (N,N) space; diagonal is masked to a constant contribution
    (logit=0, obs=0 → log(0.5)) that cancels from the gradient.
    """
    B_any = NeteffectMatrix.block_model(Any, N_grp_Any, N_per_grp_Any, name="intercept")
    B_Merica = NeteffectMatrix.block_model(
        Merica, N_grp_Merica, N_per_grp_Merica, name="Merica"
    )
    SR_mat = m4.net2.sender_receiver(sender_predictors, wide_receiver_preds)
    D_mat = NeteffectMatrix.dyadic_effect(dyadic_predictors_mat)
    logits = (B_any + B_Merica + SR_mat + D_mat) * mask_mat
    m4.dist.bernoulli(logits=logits, obs=Y_mat, name="Y_mat")


# %% Fit models
m.data_on_model = dict(
    long_block_set=long_block_set,
    long_dyad_set=long_dyad_set,
    long_focal_set=long_focal_set,
    long_target_set=long_target_set,
    long_ids_int=long_ids_int,
    network=network,
    exposure=exposure,
)
m.fit(model_srm_long_block, num_samples=1000, num_warmup=1000, num_chains=1)
m2 = bf("cpu")
m2.data_on_model = dict(
    network_edgl=wide_network_edgl,
    dyadic_predictors=wide_dyad_edgl,
    sender_predictors=wide_sender_preds,
    receiver_predictors=wide_receiver_preds,
    Any=Any,
    Merica=Merica,
)
m2.fit(model_srm_wide, num_samples=1000, num_warmup=1000, num_chains=1)

# BF Matrix SRM — non-sharded (edgelist likelihood, matrix effects)
m3 = bf("cpu")
m3.data_on_model = dict(
    network_edgl=wide_network_edgl,
    dyadic_predictors_mat=dyadic_preds_mat_jnp,
    sender_predictors=wide_sender_preds,
    receiver_predictors=wide_receiver_preds,
    Any=Any,
    Merica=Merica,
)
m3.fit(model_srm_matrix, num_samples=1000, num_warmup=1000, num_chains=1)

# BF Matrix SRM — sharded (matrix likelihood, axis-0 sharding on CORES_SHARD devices)
# Only Y_mat / dyadic_predictors_mat / sender_predictors go in data_on_model so that
# _auto_shard_data shards exactly those three. receiver_predictors, Any, Merica are
# captured from closure (replicated) because they span the full N column dimension.
m4 = bf("cpu", cores=CORES_SHARD)
m4.data_on_model = dict(
    Y_mat=Y_mat_jnp,
    dyadic_predictors_mat=dyadic_preds_mat_jnp,
    sender_predictors=wide_sender_preds,
)
m4.fit(model_srm_matrix_sharded, num_samples=1000, num_warmup=1000, num_chains=4, shard=True)


# %% STAN2 — vectorized Stan SRM
def _build_stan2_data():
    """Build the Stan2 data dict from arrays already loaded in SIM.py."""
    sender_np   = np.array(wide_sender_preds)    # (N, 4)
    receiver_np = np.array(wide_receiver_preds)  # (N, 4)
    dyad_np     = np.array(wide_dyad_edgl)       # (N_dyads, 2, K)

    N  = N_nodes
    ND = N_dyads
    NO = N_dyads * 2
    K  = dyad_np.shape[2]

    # Edgelist indices (0-based → 1-based for Stan)
    urows, ucols = np.triu_indices(N, k=1)
    sender_arr   = np.concatenate([urows, ucols]) + 1
    receiver_arr = np.concatenate([ucols, urows]) + 1
    dyad_id_arr  = np.concatenate([np.arange(1, ND + 1), np.arange(1, ND + 1)])
    dyad_dir_arr = np.concatenate([np.ones(ND, dtype=int), np.full(ND, 2, dtype=int)])

    outcomes_arr = np.concatenate([
        np.array(wide_network_edgl[:, 0]),
        np.array(wide_network_edgl[:, 1]),
    ]).astype(int)

    # Predictor matrices with leading intercept column
    focal_set_stan  = np.column_stack([np.ones(N), sender_np])    # (N, 5)
    target_set_stan = np.column_stack([np.ones(N), receiver_np])  # (N, 5)
    flat_dyad       = np.concatenate([dyad_np[:, 0, :], dyad_np[:, 1, :]], axis=0)  # (NO, K)
    dyad_set_stan   = np.column_stack([np.ones(NO), flat_dyad])   # (NO, K+1)

    block_set_stan = np.column_stack([Any_np + 1, Merica_np + 1])

    priors = np.zeros((23, 2))
    for i, row in enumerate([
        [-3.00, 1.5], [3.00, 1.5], [-1.50, 1.0], [1.00, 0.0], [1.00, 0.0], [1.00, 0.0],
        [0.00, 2.5],  [0.00, 2.5], [0.00, 2.5],  [0.10, 2.5], [0.01, 2.5], [0.00, 2.5],
        [0.00, 2.5],  [0.00, 2.5], [0.00, 2.5],  [0.00, 2.5], [2.50, 0.0], [2.50, 0.0],
        [1.50, 0.0],  [3.00, 1.0], [2.00, 0.0],  [3.00, 12.0],[0.00, 2.5],
    ]):
        priors[i] = row

    return {
        "N_networktypes": 1, "N_id": N, "N_dyads": ND, "N_obs": NO, "N_responses": 1,
        "N_params": [sender_np.shape[1] + 1, receiver_np.shape[1] + 1, K + 1],
        "sender": sender_arr.tolist(), "receiver": receiver_arr.tolist(),
        "dyad_id": dyad_id_arr.tolist(), "dyad_dir": dyad_dir_arr.tolist(),
        "outcomes": outcomes_arr.tolist(),
        "outcomes_real": outcomes_arr.astype(float).tolist(),
        "exposure": np.ones(NO, dtype=int).tolist(),
        "N_group_vars": 2, "max_N_groups": int(N_grp_Merica),
        "N_groups_per_var": [int(N_grp_Any), int(N_grp_Merica)],
        "block_set": block_set_stan.tolist(),
        "focal_set": focal_set_stan.tolist(), "target_set": target_set_stan.tolist(),
        "dyad_set": dyad_set_stan.tolist(),
        "priors": priors.tolist(), "export_network": 0, "outcome_mode": 1, "link_mode": 1,
    }


def _extract_stan2_posteriors(fit, N_grp_Any, N_grp_Merica):
    """Map cmdstanpy draws to the same key names used by BF wide / _find_BF."""
    import logging; logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    def sv(name):
        # stan_variable returns (S, ...) where S = chains * iter_sampling
        arr = fit.stan_variable(name)
        return np.array(arr)

    # STAN2.stan block_effects: flat vector [b_any(1), b_merica(9)] column-major
    block_draws = sv("block_effects")
    idx = 0
    b_any_draws = block_draws[:, idx: idx + N_grp_Any ** 2].reshape(-1, N_grp_Any, N_grp_Any)
    idx += N_grp_Any ** 2
    b_merica_flat  = block_draws[:, idx: idx + N_grp_Merica ** 2]
    b_merica_draws = b_merica_flat.reshape(-1, N_grp_Merica, N_grp_Merica).transpose(0, 2, 1)

    sr_L = sv("sr_L")
    dr_L = sv("dr_L")
    if sr_L.ndim == 2: sr_L = sr_L.reshape(-1, 2, 2)
    if dr_L.ndim == 2: dr_L = dr_L.reshape(-1, 2, 2)

    return {
        "b_intercept":      b_any_draws,
        "b_Merica":         b_merica_draws,
        "sender_effects":   sv("focal_effects"),
        "receiver_effects": sv("target_effects"),
        "dyad_effects":     sv("dyad_effects"),
        "sr_sigma":         sv("sr_sigma"),
        "sr_L":             sr_L,
        "dr_sigma":         sv("dr_sigma"),
        "dr_L":             dr_L,
        "sr_raw":           sv("z_sr"),
        "dr_raw":           sv("z_dr"),
    }


_stan2_path = "/home/sebastian_sosa/BF/Test/Network/SRM/benchmark_suite/STAN2.stan"
import logging; logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
print("Compiling STAN2 ...")
_sm_stan2 = CmdStanModel(stan_file=_stan2_path)
print("Running STAN2 ...")
_fit_stan2 = _sm_stan2.sample(
    data=_build_stan2_data(),
    iter_warmup=1000, iter_sampling=1000, chains=1, show_progress=True,
)
stan2_posteriors = _extract_stan2_posteriors(_fit_stan2, N_grp_Any, N_grp_Merica)
print("STAN2 done.")


# %% Robust Comparison
def r_to_py(obj):
    if isinstance(obj, ListVector):
        if isinstance(obj.names, (NULLType, type(None))):
            return [r_to_py(el) for el in obj]
        names = [str(n) for n in obj.names]
        return {n: r_to_py(obj.rx2(n)) for n in names}
    if hasattr(obj, "dim") and not isinstance(obj.dim, (NULLType, type(None))):
        return np.array(obj).reshape(list(obj.dim), order="F")
    if isinstance(obj, (FloatVector, IntVector, StrVector)):
        return np.array(obj)
    return obj


def compare_results(
    BF_long_posteriors,
    strand_posteriors,
    BF_edgelist_posteriors=None,
    STAN2_posteriors=None,
    BF_matrix_posteriors=None,
    BF_matrix_shard_posteriors=None,
    outpath="forest_plot.png",
):
    def get_stats(data):
        m = np.mean(data)
        hpd = np.percentile(data, [2.5, 97.5])
        return m, [m - hpd[0], hpd[1] - m]

    def _flatten_block(arr):
        arr = np.array(arr)
        if arr.ndim == 3:
            S, M, N = arr.shape
            return arr.transpose(0, 2, 1).reshape(S, M * N)
        return arr.reshape(arr.shape[0], -1)

    def _b_keys(post_dict):
        if post_dict is None:
            return []
        return sorted([k for k in post_dict if k.startswith("b_")], key=str.lower)

    def _find_BF(post_dict, keys):
        if post_dict is None:
            return None
        for k in keys:
            if k in post_dict:
                return np.array(post_dict[k]).reshape(np.array(post_dict[k]).shape[0], -1)
        return None

    summaries = []

    # ── 1. Blocks ──
    arr_l_blk = (
        np.array(BF_long_posteriors["block_effects"]).reshape(
            np.array(BF_long_posteriors["block_effects"]).shape[0], -1
        )
        if "block_effects" in BF_long_posteriors
        else None
    )
    long_bi_idx = 0
    wide_b_keys  = _b_keys(BF_edgelist_posteriors)
    stan2_b_keys = _b_keys(STAN2_posteriors)
    mat_b_keys   = _b_keys(BF_matrix_posteriors)
    shard_b_keys = _b_keys(BF_matrix_shard_posteriors)

    raw_block_list = strand_posteriors.get("block_parameters", [])
    if hasattr(raw_block_list, "__len__") and not isinstance(raw_block_list, np.ndarray):
        block_list = list(raw_block_list)
    elif isinstance(raw_block_list, np.ndarray) and raw_block_list.dtype == object:
        block_list = [raw_block_list[i] for i in range(len(raw_block_list))]
    else:
        block_list = [raw_block_list] if raw_block_list is not None else []

    for b_idx, raw_s in enumerate(block_list):
        arr_s = _flatten_block(raw_s)
        n_params = arr_s.shape[1]
        wide_arr  = _flatten_block(BF_edgelist_posteriors[wide_b_keys[b_idx]]) if (BF_edgelist_posteriors  and b_idx < len(wide_b_keys))  else None
        st2_arr   = _flatten_block(STAN2_posteriors[stan2_b_keys[b_idx]]) if (STAN2_posteriors         and b_idx < len(stan2_b_keys)) else None
        mat_arr   = _flatten_block(BF_matrix_posteriors[mat_b_keys[b_idx]]) if (BF_matrix_posteriors     and b_idx < len(mat_b_keys))   else None
        shard_arr = _flatten_block(BF_matrix_shard_posteriors[shard_b_keys[b_idx]]) if (BF_matrix_shard_posteriors and b_idx < len(shard_b_keys)) else None
        for j in range(n_params):
            label = f"block{b_idx}[{j}]" if n_params > 1 else f"block{b_idx}"
            sm, se = get_stats(arr_s[:, j])
            mm, me = (get_stats(arr_l_blk[:, long_bi_idx])
                      if (arr_l_blk is not None and long_bi_idx < arr_l_blk.shape[1])
                      else (None, None))
            long_bi_idx += 1
            wm,  we  = (get_stats(wide_arr[:, j])  if (wide_arr  is not None and j < wide_arr.shape[1])  else (None, None))
            s2m, s2e = (get_stats(st2_arr[:, j])   if (st2_arr   is not None and j < st2_arr.shape[1])   else (None, None))
            xm,  xe  = (get_stats(mat_arr[:, j])   if (mat_arr   is not None and j < mat_arr.shape[1])   else (None, None))
            shm, she = (get_stats(shard_arr[:, j]) if (shard_arr is not None and j < shard_arr.shape[1]) else (None, None))
            summaries.append({
                "param": label,
                "strand_m": sm,  "strand_err": se,
                "long_m": mm,    "long_err": me,
                "wide_m": wm,    "wide_err": we,
                "stan2_m": s2m,  "stan2_err": s2e,
                "mat_m": xm,     "mat_err": xe,
                "shard_m": shm,  "shard_err": she,
            })

    # ── 2. SRM ──
    mapping = [
        (("focal_effects", "sender_effects"), "focal_coeffs"),
        (("target_effects", "receiver_effects"), "target_coeffs"),
        (("dyad_effects", "dyadic_effects"), "dyadic_coeffs"),
        (("sr_sigma",), "focal_target_sd"),
        (("sr_L",), "focal_target_L"),
        (("dr_sigma",), "dyadic_sd"),
        (("dr_L",), "dyadic_L"),
    ]
    for BF_keys, s_key in mapping:
        if s_key not in strand_posteriors:
            continue
        arr_s  = np.array(strand_posteriors[s_key]).reshape(np.array(strand_posteriors[s_key]).shape[0], -1)
        arr_l  = _find_BF(BF_long_posteriors, BF_keys)
        arr_w  = _find_BF(BF_edgelist_posteriors, BF_keys)
        arr_s2 = _find_BF(STAN2_posteriors, BF_keys)
        arr_x  = _find_BF(BF_matrix_posteriors, BF_keys)
        arr_sh = _find_BF(BF_matrix_shard_posteriors, BF_keys)
        for j in range(arr_s.shape[1]):
            sm,  se  = get_stats(arr_s[:, j])
            mm,  me  = (get_stats(arr_l[:, j])  if (arr_l  is not None and j < arr_l.shape[1])  else (None, None))
            wm,  we  = (get_stats(arr_w[:, j])  if (arr_w  is not None and j < arr_w.shape[1])  else (None, None))
            s2m, s2e = (get_stats(arr_s2[:, j]) if (arr_s2 is not None and j < arr_s2.shape[1]) else (None, None))
            xm,  xe  = (get_stats(arr_x[:, j])  if (arr_x  is not None and j < arr_x.shape[1])  else (None, None))
            shm, she = (get_stats(arr_sh[:, j]) if (arr_sh is not None and j < arr_sh.shape[1]) else (None, None))
            label = f"{BF_keys[0]}[{j}]" if arr_s.shape[1] > 1 else BF_keys[0]
            summaries.append({
                "param": label,
                "strand_m": sm,  "strand_err": se,
                "long_m": mm,    "long_err": me,
                "wide_m": wm,    "wide_err": we,
                "stan2_m": s2m,  "stan2_err": s2e,
                "mat_m": xm,     "mat_err": xe,
                "shard_m": shm,  "shard_err": she,
            })

    params = [d["param"] for d in summaries]
    n = len(params)
    y = np.arange(n)
    colors = {
        "strand":   "#074FA2",
        "long":     "#1f77b4",
        "edgelist": "#B58900",
        "stan2":    "#228B86",
        "mat":      "#912D58",
        "shard":    "#A33B07",
        "unique":   "#488A6C",
    }
    # 6 series centred on y, step 0.15
    offsets = {"strand": 0.375, "long": 0.225, "edgelist": 0.075,
               "stan2": -0.075, "mat": -0.225, "shard": -0.375}

    def _safe_xerr(summaries, key_err):
        errs = []
        for d in summaries:
            e = d[key_err]
            errs.append(e if e is not None else [0, 0])
        return np.array(errs).T

    plt.figure(figsize=(10, n * 0.45 + 2))
    plt.axvline(0, color="k", ls="--", alpha=0.3)
    plt.errorbar(
        [d["strand_m"] for d in summaries], y + offsets["strand"],
        xerr=np.array([d["strand_err"] for d in summaries]).T,
        fmt="o", label="STRAND", color=colors["strand"], capsize=3,
    )
    plt.errorbar(
        [d["long_m"] if d["long_m"] is not None else np.nan for d in summaries],
        y + offsets["long"],
        xerr=_safe_xerr(summaries, "long_err"),
        fmt="s", label="BF long", color=colors["long"], capsize=3,
    )
    if BF_edgelist_posteriors:
        plt.errorbar(
            [d["wide_m"] if d["wide_m"] is not None else np.nan for d in summaries],
            y + offsets["edgelist"],
            xerr=_safe_xerr(summaries, "wide_err"),
            fmt="^", label="BF edgelist", color=colors["edgelist"], capsize=3,
        )
    if STAN2_posteriors:
        plt.errorbar(
            [d["stan2_m"] if d["stan2_m"] is not None else np.nan for d in summaries],
            y + offsets["stan2"],
            xerr=_safe_xerr(summaries, "stan2_err"),
            fmt="v", label="STAN2", color=colors["stan2"], capsize=3,
        )
    if BF_matrix_posteriors:
        plt.errorbar(
            [d["mat_m"] if d["mat_m"] is not None else np.nan for d in summaries],
            y + offsets["mat"],
            xerr=_safe_xerr(summaries, "mat_err"),
            fmt="D", label="BF matrix", color=colors["mat"], capsize=3,
        )
    if BF_matrix_shard_posteriors:
        plt.errorbar(
            [d["shard_m"] if d["shard_m"] is not None else np.nan for d in summaries],
            y + offsets["shard"],
            xerr=_safe_xerr(summaries, "shard_err"),
            fmt="P", label="BF matrix (shard)", color=colors["shard"], capsize=3,
        )
    plt.yticks(y, params, fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.title("SRM Posterior Comparison", fontsize=25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    plt.close()

    # ── 4. Scatters ──
    def _panel(ax, x, y, xl, yl, col, m):
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return
        ax.scatter(x, y, color=col, alpha=0.6, s=25, marker=m)
        all_v = np.concatenate([x, y])
        lo, hi = all_v.min(), all_v.max()
        pad = (hi - lo) * 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.3)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(xl, fontsize=15)
        ax.set_ylabel(yl, fontsize=15)
        ax.set_title(f"r={r:.3f}", fontsize=15)

    # Nodal random effects
    if "focal_target_random_effects" in strand_posteriors:
        st_sr = np.array(strand_posteriors["focal_target_random_effects"])
        print(f"[DEBUG] STRAND nodal RE shape: {st_sr.shape}")
        if st_sr.ndim >= 2:
            if st_sr.ndim == 3:
                st_sr = st_sr.mean(axis=0)  # (N, 2)
            li_sr = (np.array(BF_long_posteriors["sr_raw"]).mean(axis=0).T
                     if "sr_raw" in BF_long_posteriors else None)
            wi_sr = (np.array(BF_edgelist_posteriors["sr_raw"]).mean(axis=0).T
                     if BF_edgelist_posteriors and "sr_raw" in BF_edgelist_posteriors else None)
            s2_sr = (np.array(STAN2_posteriors["sr_raw"]).mean(axis=0).T
                     if STAN2_posteriors and "sr_raw" in STAN2_posteriors else None)
            ma_sr = (np.array(BF_matrix_posteriors["sr_raw"]).mean(axis=0).T
                     if BF_matrix_posteriors and "sr_raw" in BF_matrix_posteriors else None)
            sh_sr = (np.array(BF_matrix_shard_posteriors["sr_raw"]).mean(axis=0).T
                     if BF_matrix_shard_posteriors and "sr_raw" in BF_matrix_shard_posteriors else None)
            n_cols = (3 + (1 if s2_sr is not None else 0)
                        + (1 if ma_sr is not None else 0)
                        + (1 if sh_sr is not None else 0))
            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)
            for i, role in enumerate(["Sender", "Receiver"]):
                _panel(axes[i, 0], st_sr[:, i], li_sr[:, i] if li_sr is not None else None,
                       "STRAND", f"BF long {role}", colors["long"], "s")
                _panel(axes[i, 1], st_sr[:, i], wi_sr[:, i] if wi_sr is not None else None,
                       "STRAND", f"BF edgelist {role}", colors["edgelist"], "^")
                _panel(axes[i, 2], li_sr[:, i] if li_sr is not None else None,
                       wi_sr[:, i] if wi_sr is not None else None,
                       "BF long", f"BF edgelist {role}", colors["unique"], "o")
                col = 3
                if s2_sr is not None:
                    _panel(axes[i, col], st_sr[:, i], s2_sr[:, i],
                           "STRAND", f"STAN2 {role}", colors["stan2"], "v")
                    col += 1
                if ma_sr is not None:
                    _panel(axes[i, col], st_sr[:, i], ma_sr[:, i],
                           "STRAND", f"BF matrix {role}", colors["mat"], "D")
                    col += 1
                if sh_sr is not None:
                    _panel(axes[i, col], st_sr[:, i], sh_sr[:, i],
                           "STRAND", f"BF shard {role}", colors["shard"], "P")
            plt.tight_layout()
            plt.savefig(outpath.replace(".png", "_nodal_re.png"), dpi=150)
            plt.show()
            plt.close()

    # Dyadic random effects
    if "dyadic_random_effects" in strand_posteriors:
        st_dr = np.array(strand_posteriors["dyadic_random_effects"])
        print(f"[DEBUG] STRAND dyadic RE shape: {st_dr.shape}")
        if st_dr.ndim >= 2:
            if st_dr.ndim == 3:
                st_dr = st_dr.mean(axis=0)  # (N_dyads, 2)
            li_dr = (np.array(BF_long_posteriors["dr_raw"]).mean(axis=0).T
                     if "dr_raw" in BF_long_posteriors else None)
            wi_dr = (np.array(BF_edgelist_posteriors["dr_raw"]).mean(axis=0).T
                     if BF_edgelist_posteriors and "dr_raw" in BF_edgelist_posteriors else None)
            s2_dr = (np.array(STAN2_posteriors["dr_raw"]).mean(axis=0).T
                     if STAN2_posteriors and "dr_raw" in STAN2_posteriors else None)
            ma_dr = (np.array(BF_matrix_posteriors["dr_raw"]).mean(axis=0).T
                     if BF_matrix_posteriors and "dr_raw" in BF_matrix_posteriors else None)
            sh_dr = (np.array(BF_matrix_shard_posteriors["dr_raw"]).mean(axis=0).T
                     if BF_matrix_shard_posteriors and "dr_raw" in BF_matrix_shard_posteriors else None)
            n_cols = (3 + (1 if s2_dr is not None else 0)
                        + (1 if ma_dr is not None else 0)
                        + (1 if sh_dr is not None else 0))
            fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8), squeeze=False)
            for i, dlabel in enumerate(["i->j", "j->i"]):
                _panel(axes[i, 0], st_dr[:, i], li_dr[:, i] if li_dr is not None else None,
                       "STRAND", f"BF long {dlabel}", colors["long"], "s")
                _panel(axes[i, 1], st_dr[:, i], wi_dr[:, i] if wi_dr is not None else None,
                       "STRAND", f"BF edgelist {dlabel}", colors["edgelist"], "^")
                _panel(axes[i, 2], li_dr[:, i] if li_dr is not None else None,
                       wi_dr[:, i] if wi_dr is not None else None,
                       "BF long", f"BF edgelist {dlabel}", colors["unique"], "o")
                col = 3
                if s2_dr is not None:
                    _panel(axes[i, col], st_dr[:, i], s2_dr[:, i],
                           "STRAND", f"STAN2 {dlabel}", colors["stan2"], "v")
                    col += 1
                if ma_dr is not None:
                    _panel(axes[i, col], st_dr[:, i], ma_dr[:, i],
                           "STRAND", f"BF matrix {dlabel}", colors["mat"], "D")
                    col += 1
                if sh_dr is not None:
                    _panel(axes[i, col], st_dr[:, i], sh_dr[:, i],
                           "STRAND", f"BF shard {dlabel}", colors["shard"], "P")
            plt.tight_layout()
            plt.savefig(outpath.replace(".png", "_dyadic_re.png"), dpi=150)
            plt.show()
            plt.close()
        else:
            print(f"[WARNING] Skipping dyadic plot: unexpected shape {st_dr.shape}")

def compare_results_srm(
    BF_long_posteriors,
    strand_posteriors,
    BF_edgelist_posteriors=None,
    STAN2_posteriors=None,
    BF_matrix_posteriors=None,
    BF_matrix_shard_posteriors=None,
    outpath="forest_plot.png",
):
    return compare_results(
        BF_long_posteriors, strand_posteriors,
        BF_edgelist_posteriors, STAN2_posteriors,
        BF_matrix_posteriors, BF_matrix_shard_posteriors,
        outpath,
    )


# %% Run
strand_post = r_to_py(ro.globalenv["strand_posteriors"])
compare_results(
    m.posteriors, strand_post,
    m2.posteriors, stan2_posteriors,
    m3.posteriors, m4.posteriors,
)
# %%
