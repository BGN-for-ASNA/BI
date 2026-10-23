##############################
#### Multiplex Network Model Test
#### BF vs STRAND numpyro backend
##############################
#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import subprocess
import sys
import numpy as np
import jax.numpy as jnp
import numpyro
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULLType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from BayesForge import BayesForge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDATA_PATH = os.path.join(SCRIPT_DIR, 'multiplex_arrays.RData')

# %% Step 1: Run STRAND simulation + numpyro fit as separate Rscript process
# This avoids rpy2/reticulate conflict (STRAND numpyro calls Python via reticulate)
r_script = os.path.join(SCRIPT_DIR, 'strand_sim.R')
print("Running strand_sim.R ...")
result = subprocess.run(
    ['Rscript', '--vanilla', r_script],
    cwd=SCRIPT_DIR,
    capture_output=False,
    text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"strand_sim.R failed with code {result.returncode}")
print("strand_sim.R done.")

# %% Step 2: Load saved arrays via rpy2
ro.r['load'](RDATA_PATH)

with localconverter(ro.default_converter + numpy2ri.converter):
    long_focal_np   = np.array(ro.globalenv['long_focal_set'])
    long_target_np  = np.array(ro.globalenv['long_target_set'])
    long_dyad_np    = np.array(ro.globalenv['long_dyad_set'])
    long_block_np   = np.array(ro.globalenv['long_block_set'])
    block_mu_np     = np.array(ro.globalenv['block_mu'])
    block_sigma_np  = np.array(ro.globalenv['block_sigma'])
    long_outcome_np = np.array(ro.globalenv['long_outcome'])
    locs_raw        = np.array(ro.globalenv['locs'])
    bandage_penalty = float(np.array(ro.globalenv['bandage_penalty']).ravel()[0])
    N_id_r          = int(np.array(ro.globalenv['N_id_d']).ravel()[0])
    N_layers_r      = int(np.array(ro.globalenv['N_lay']).ravel()[0])

def _safe_int_arr(r_name):
    try:
        v = ro.globalenv[r_name]
        if isinstance(v, NULLType):
            return None
        arr = np.array(v)
        if arr.size == 0 or (arr.ndim == 0 and int(arr) == 0):
            return None
        return arr.astype(int) - 1  # R 1-indexed -> 0-indexed
    except Exception:
        return None

dr_bind_out1 = _safe_int_arr('dr_bind_out1')
dr_bind_out2 = _safe_int_arr('dr_bind_out2')
dr_bind_in1  = _safe_int_arr('dr_bind_in1')
dr_bind_in2  = _safe_int_arr('dr_bind_in2')

# Compute standard binding pattern if STRAND didn't expose them
if dr_bind_out1 is None:
    NL = N_layers_r
    o1, o2, i1, i2 = [], [], [], []
    for l1 in range(NL):
        for l2 in range(l1 + 1, NL):
            o1.append(l1);    o2.append(NL + l2)
            i1.append(l2);    i2.append(NL + l1)
    dr_bind_out1 = np.array(o1, dtype=int)
    dr_bind_out2 = np.array(o2, dtype=int)
    dr_bind_in1  = np.array(i1, dtype=int)
    dr_bind_in2  = np.array(i2, dtype=int)

# Fix rpy2 transpositions: long_*_set target = (N_dyads, 2, N_var)
N_dyads_expected = N_id_r * (N_id_r - 1) // 2

for name, arr in [('focal', long_focal_np), ('target', long_target_np),
                  ('dyad', long_dyad_np), ('block', long_block_np)]:
    if arr.shape[0] != N_dyads_expected:
        arr = arr.transpose(2, 1, 0)
    globals()[f'long_{name}_np'] = arr

# long_outcome target = (N_layers, N_dyads, 2)
if long_outcome_np.shape[1] != N_dyads_expected:
    long_outcome_np = long_outcome_np.transpose(0, 2, 1)
# strand_sim.R stores slot0=om[Var1,Var2] but STRAND convention is slot0=om[Var2,Var1] → swap
long_outcome_np = long_outcome_np[:, :, [1, 0]]

if locs_raw.dtype.names is not None:
    locs_2d = np.stack([locs_raw['Var2'], locs_raw['Var1']], axis=1)
else:
    locs_2d = locs_raw.T if locs_raw.shape[0] == 2 else locs_raw
long_ids_int = jnp.array(locs_2d, dtype=jnp.int32) - 1

long_focal_set  = jnp.array(long_focal_np)   # full design matrix (incl intercept col)
long_target_set = jnp.array(long_target_np)
long_dyad_set   = jnp.array(long_dyad_np)
long_block_set  = jnp.array(long_block_np)
long_outcome    = jnp.array(long_outcome_np)

N_dyads      = int(long_ids_int.shape[0])
N_id         = int(jnp.max(long_ids_int) + 1)
N_layers     = N_layers_r
N_var_focal  = int(long_focal_set.shape[2])
N_var_target = int(long_target_set.shape[2])
N_var_dyad   = int(long_dyad_set.shape[2])
N_var_block  = int(long_block_set.shape[2])

block_mu_jnp    = jnp.array(block_mu_np)
block_sigma_jnp = jnp.array(block_sigma_np)

# STRAND zeros out intercept col (index 0) in focal/target/dyad effects
A_F = jnp.array([0.] + [1.] * (N_var_focal - 1))
A_T = jnp.array([0.] + [1.] * (N_var_target - 1))
A_D = jnp.array([0.] + [1.] * (N_var_dyad - 1))

bind_out1 = jnp.array(dr_bind_out1)
bind_out2 = jnp.array(dr_bind_out2)
bind_in1  = jnp.array(dr_bind_in1)
bind_in2  = jnp.array(dr_bind_in2)

print(f"N_id={N_id}, N_layers={N_layers}, N_dyads={N_dyads}")
print(f"N_var_focal={N_var_focal}, N_var_target={N_var_target}, "
      f"N_var_dyad={N_var_dyad}, N_var_block={N_var_block}")
print(f"outcome shape: {long_outcome.shape}")
print(f"bandage bindings: {bind_out1.shape[0]} pairs, penalty={bandage_penalty}")

# %% Step 3: Load STRAND numpyro posterior samples from .npy files
# strand_sim.R saves each param as strand_post_<name>.npy via numpy directly
# (bypasses reticulate serialization issues when saving Python dicts to RData)
import glob
strand_post = {}
npy_files = glob.glob(os.path.join(SCRIPT_DIR, 'strand_post_*.npy'))
for f in npy_files:
    nm = os.path.basename(f)[len('strand_post_'):-len('.npy')]
    strand_post[nm] = np.load(f)
print("STRAND numpyro params:", list(strand_post.keys()))

# %% Step 4: BF multiplex model (mirrors numpyro_multiplex exactly)
def model_multiplex_BF(outcome, long_focal_set, long_target_set, long_dyad_set,
                       long_block_set, long_ids_int,
                       block_mu, block_sigma,
                       A_F, A_T, A_D,
                       bind_out1, bind_out2, bind_in1, bind_in2,
                       bandage_penalty,
                       N_id, N_dyads, N_layers,
                       N_var_focal, N_var_target, N_var_dyad, N_var_block,
                       sample=False):
    m = BF()

    # Observation noise: (N_layers, 1, 1) matches numpyro_multiplex exactly
    # prior_23: loc=0, scale=2.5 (STRAND default)
    error_sigma = m.dist.truncated_normal(
        loc=jnp.zeros((N_layers, 1, 1)),
        scale=jnp.full((N_layers, 1, 1), 2.5),
        low=0.,
        shape=(N_layers, 1, 1),
        name='error_sigma', sample=sample)

    # Fixed effects: (N_layers, N_var_*)
    # prior_12/13/14: loc=0, scale=2.5 (STRAND default)
    focal_effects = m.dist.normal(
        loc=jnp.zeros((N_layers, N_var_focal)),
        scale=jnp.full((N_layers, N_var_focal), 2.5),
        shape=(N_layers, N_var_focal),
        name='focal_effects', sample=sample)

    target_effects = m.dist.normal(
        loc=jnp.zeros((N_layers, N_var_target)),
        scale=jnp.full((N_layers, N_var_target), 2.5),
        shape=(N_layers, N_var_target),
        name='target_effects', sample=sample)

    dyad_effects = m.dist.normal(
        loc=jnp.zeros((N_layers, N_var_dyad)),
        scale=jnp.full((N_layers, N_var_dyad), 2.5),
        shape=(N_layers, N_var_dyad),
        name='dyad_effects', sample=sample)

    block_effects = m.dist.normal(
        loc=jnp.broadcast_to(block_mu, (N_layers, N_var_block)),
        scale=jnp.broadcast_to(block_sigma, (N_layers, N_var_block)),
        shape=(N_layers, N_var_block),
        name='block_effects', sample=sample)

    # Linear predictor: (N_layers, N_dyads, 2)
    # A_F/A_T/A_D zero out intercept col (index 0) matching STRAND's masking
    mu = (
        jnp.einsum('dxf,lf->ldx', long_focal_set,  focal_effects * A_F) +
        jnp.einsum('dxf,lf->ldx', long_target_set, target_effects * A_T) +
        jnp.einsum('dxf,lf->ldx', long_dyad_set,   dyad_effects * A_D) +
        jnp.einsum('dxf,lf->ldx', long_block_set,  block_effects)
    )

    # Dyadic random effects: 2*N_layers Cholesky
    dr_raw = m.dist.normal(
        loc=jnp.zeros((N_layers * 2, N_dyads)),
        scale=1.,
        shape=(N_layers * 2, N_dyads),
        name='dr_raw', sample=sample)

    # prior_18: LKJ concentration=2.5 (STRAND default)
    dr_L = m.dist.lkj_cholesky(N_layers * 2, 2.5, name='dr_L', sample=sample)

    # prior_16: loc=0, scale=2.5
    dr_sigma = m.dist.truncated_normal(
        loc=jnp.zeros(N_layers),
        scale=jnp.full((N_layers,), 2.5),
        low=0.,
        shape=(N_layers,),
        name='dr_sigma', sample=sample)

    dr_sigma_temp   = jnp.expand_dims(jnp.repeat(dr_sigma, 2), 1)  # (2*N_layers, 1)
    dr_sigma_scaled = dr_sigma_temp * dr_L
    dr              = jnp.matmul(dr_sigma_scaled, dr_raw)           # (2*N_layers, N_dyads)
    dr_long         = jnp.stack(jnp.split(dr, 2), axis=2)          # (N_layers, N_dyads, 2)

    # Sender-receiver random effects: 2*N_layers Cholesky
    sr_raw = m.dist.normal(
        loc=jnp.zeros((N_layers * 2, N_id)),
        scale=1.,
        shape=(N_layers * 2, N_id),
        name='sr_raw', sample=sample)

    # prior_17: LKJ concentration=2.5 (STRAND default)
    sr_L = m.dist.lkj_cholesky(N_layers * 2, 2.5, name='sr_L', sample=sample)

    # prior_15: loc=0, scale=2.5
    sr_sigma = m.dist.truncated_normal(
        loc=jnp.zeros(N_layers * 2),
        scale=jnp.full((N_layers * 2,), 2.5),
        low=0.,
        shape=(N_layers * 2,),
        name='sr_sigma', sample=sample)

    sr_sigma_temp   = jnp.expand_dims(sr_sigma, 1)
    sr_sigma_scaled = sr_sigma_temp * sr_L
    gr              = jnp.matmul(sr_sigma_scaled, sr_raw)           # (2*N_layers, N_id)

    gr_sender, gr_receiver = jnp.split(gr, 2)                       # each (N_layers, N_id)

    S_i = gr_sender[:,   long_ids_int[:, 0]]    # (N_layers, N_dyads)
    S_j = gr_sender[:,   long_ids_int[:, 1]]
    R_j = gr_receiver[:, long_ids_int[:, 1]]
    R_i = gr_receiver[:, long_ids_int[:, 0]]

    gr_long = jnp.stack([S_i + R_j, S_j + R_i], axis=2)            # (N_layers, N_dyads, 2)

    linear_model = mu + dr_long + gr_long

    # Gaussian likelihood (outcome_mode=4 in numpyro_multiplex)
    m.dist.normal(loc=linear_model, scale=error_sigma,
                  obs=outcome, name='obs', sample=sample)

    # Deterministic correlation matrices
    G_corr = numpyro.deterministic('G_corr', jnp.matmul(sr_L, sr_L.T))
    D_corr = numpyro.deterministic('D_corr', jnp.matmul(dr_L, dr_L.T))

    # Bandage constraint matching numpyro_multiplex
    if bind_out1.shape[0] > 0:
        numpyro.factor(
            'dyadic_bandage_constraint',
            -0.5 * jnp.sum(
                ((D_corr[bind_in1, bind_in2] - D_corr[bind_out1, bind_out2])
                 / bandage_penalty) ** 2))


# %% Step 5: Fit BF model
m_BF = BF('cpu')
m_BF.data_on_model = dict(
    outcome=long_outcome,
    long_focal_set=long_focal_set,
    long_target_set=long_target_set,
    long_dyad_set=long_dyad_set,
    long_block_set=long_block_set,
    long_ids_int=long_ids_int,
    block_mu=block_mu_jnp,
    block_sigma=block_sigma_jnp,
    A_F=A_F,
    A_T=A_T,
    A_D=A_D,
    bind_out1=bind_out1,
    bind_out2=bind_out2,
    bind_in1=bind_in1,
    bind_in2=bind_in2,
    bandage_penalty=bandage_penalty,
    N_id=N_id,
    N_dyads=N_dyads,
    N_layers=N_layers,
    N_var_focal=N_var_focal,
    N_var_target=N_var_target,
    N_var_dyad=N_var_dyad,
    N_var_block=N_var_block,
)
m_BF.fit(model_multiplex_BF, num_samples=2000, num_warmup=2000, num_chains=1,
         target_accept_prob=0.95, max_tree_depth=12,
         init_strategy=numpyro.infer.init_to_uniform(radius=0.25))
BF_post = m_BF.posteriors


# %% Step 6: Density plot comparison utilities
def _flatten(arr):
    arr = np.array(arr)
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)
    return arr.reshape(arr.shape[0], -1)

def density_figure(strand_samples, BF_samples, param_name, out_dir):
    try:
        s = _flatten(strand_samples)
        b = _flatten(BF_samples)
    except Exception as e:
        print(f'  Skipping {param_name}: flatten error: {e}')
        return
    n = min(s.shape[1], b.shape[1])
    if n == 0:
        return
    ncols = min(7, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.0), squeeze=False)
    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        for vals, color, label in [(s[:, idx], '#e07b00', 'STRAND'),
                                   (b[:, idx], '#1565c0', 'BF')]:
            if vals.std() < 1e-9:
                continue
            lo = vals.mean() - 4 * vals.std()
            hi = vals.mean() + 4 * vals.std()
            xs = np.linspace(lo, hi, 300)
            try:
                kde = gaussian_kde(vals, bw_method='silverman')
                ax.plot(xs, kde(xs), color=color, label=label, linewidth=1.4)
                ax.axvline(vals.mean(), color=color, ls='--', alpha=0.55, lw=1)
            except Exception:
                pass
        ax.set_title(f'[{idx}]', fontsize=7)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f'{param_name}: STRAND numpyro (orange) vs BF (blue)', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f'density_{param_name}.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved {os.path.basename(path)}')


def corr_density_figure(strand_L_samples, BF_L_samples, label, out_dir):
    """Upper-triangle correlations from L @ L.T samples."""
    def corr_upper(L_samples):
        rows = []
        arr = np.array(L_samples)
        for i in range(arr.shape[0]):
            L = arr[i]
            if L.ndim == 1:
                n = int(np.sqrt(len(L)))
                L = L.reshape(n, n)
            C = L @ L.T
            ri, ci = np.triu_indices(C.shape[0], k=1)
            rows.append(C[ri, ci])
        return np.stack(rows)

    s = corr_upper(strand_L_samples)
    b = corr_upper(BF_L_samples)
    density_figure(s, b, f'{label}_corr', out_dir)


# %% Step 7: Generate all density plots
scalar_params = ['focal_effects', 'target_effects', 'dyad_effects',
                 'block_effects', 'sr_sigma', 'dr_sigma', 'error_sigma']

for pname in scalar_params:
    s = strand_post.get(pname)
    b = BF_post.get(pname)
    if s is not None and b is not None:
        print(f'  Plotting {pname}: strand dtype={np.array(s).dtype}, BF dtype={np.array(b).dtype}')
        density_figure(s, b, pname, SCRIPT_DIR)
    else:
        print(f'  Skipping {pname}: strand={s is not None}, BF={b is not None}')

for lname, cname in [('sr_L', 'sr'), ('dr_L', 'dr')]:
    s = strand_post.get(lname)
    b = BF_post.get(lname)
    if s is not None and b is not None:
        corr_density_figure(s, b, cname, SCRIPT_DIR)


# %% Step 8: Summary txt with means + differences
def build_summary_df(strand_post, BF_post, params):
    rows = []
    for pname in params:
        s = strand_post.get(pname)
        b = BF_post.get(pname)
        if s is None or b is None:
            continue
        sf = _flatten(s)
        bf = _flatten(b)
        n = min(sf.shape[1], bf.shape[1])
        for idx in range(n):
            sm = float(sf[:, idx].mean())
            bm = float(bf[:, idx].mean())
            rows.append({
                'Parameter':   f'{pname}[{idx}]',
                'STRAND_mean': round(sm, 4),
                'BF_mean':     round(bm, 4),
                'Difference':  round(bm - sm, 4),
            })
    # Add correlation matrices (mean Cholesky -> corr upper tri)
    for lname, cname in [('sr_L', 'sr_corr'), ('dr_L', 'dr_corr')]:
        s = strand_post.get(lname)
        b = BF_post.get(lname)
        if s is None or b is None:
            continue
        sL = np.array(s).mean(axis=0)
        bL = np.array(b).mean(axis=0)
        if sL.ndim == 1:
            n = int(np.sqrt(len(sL)))
            sL = sL.reshape(n, n); bL = bL.reshape(n, n)
        sC = sL @ sL.T; bC = bL @ bL.T
        ri, ci = np.triu_indices(sC.shape[0], k=1)
        for k, (r, c) in enumerate(zip(ri, ci)):
            sm = float(sC[r, c]); bm = float(bC[r, c])
            rows.append({
                'Parameter':   f'{cname}[{r},{c}]',
                'STRAND_mean': round(sm, 4),
                'BF_mean':     round(bm, 4),
                'Difference':  round(bm - sm, 4),
            })
    return pd.DataFrame(rows)


df = build_summary_df(strand_post, BF_post,
                      ['focal_effects', 'target_effects', 'dyad_effects',
                       'block_effects', 'sr_sigma', 'dr_sigma', 'error_sigma'])

txt_path = os.path.join(SCRIPT_DIR, 'parameter_comparison.txt')
with open(txt_path, 'w') as f:
    f.write('Parameter Mean Comparison: STRAND numpyro vs BF\n')
    f.write('=' * 65 + '\n\n')
    f.write(df.to_string(index=False))
    f.write('\n\n')
    f.write(f"Max  |Difference|: {df['Difference'].abs().max():.4f}\n")
    f.write(f"Mean |Difference|: {df['Difference'].abs().mean():.4f}\n")

print(f'\nSaved {txt_path}')
print(df.to_string(index=False))
#%%
