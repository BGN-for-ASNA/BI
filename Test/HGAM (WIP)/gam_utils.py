import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline, CubicSpline
import pandas as pd

def get_knots(x, k):
    """Select k knots from x (standard R-style placement)."""
    xu = np.unique(x)
    if len(xu) <= k: return xu
    return np.quantile(xu, np.linspace(0, 1, k))

def apply_sum_to_zero_constraint(X, S_list=None):
    """Apply sum-to-zero constraint. Matches mgcv's reparameterization."""
    if X.size == 0: return X, S_list, None
    N, K = X.shape
    C = np.ones((1, N)) @ X 
    Q, R = np.linalg.qr(C.T, mode='complete')
    Z = Q[:, 1:] 
    X_c = X @ Z
    S_c_list = []
    if S_list is not None:
        if isinstance(S_list, list):
            for S in S_list: S_c_list.append(Z.T @ S @ Z)
        else: S_c_list = Z.T @ S_list @ Z
    return X_c, S_c_list, Z

def basis_tp(x, k, m=2):
    """Thin Plate Regression Splines matching mgcv scaling (1D or 2D)."""
    xu, inverse_indices = np.unique(x, return_inverse=True, axis=0)
    n_u = len(xu)
    if x.ndim == 1: xu_reshaped = xu[:, None]
    else: xu_reshaped = xu
    d = xu_reshaped.shape[1]
    from scipy.spatial.distance import cdist
    r = cdist(xu_reshaped, xu_reshaped)
    if d == 1: E = (r**3) / 12.0
    elif d == 2: E = (r**2 * np.log(r + 1e-10)) / (8.0 * np.pi)
    else: E = r**2 
    T = np.column_stack([np.ones(n_u), xu_reshaped])
    Q, R = np.linalg.qr(T, mode='complete')
    Qf = Q[:, (d+1):] 
    E_proj = Qf.T @ E @ Qf
    vals, vecs = np.linalg.eigh(E_proj)
    k_smooth = k - (d + 1)
    idx = np.argsort(np.abs(vals))[::-1][:k_smooth]
    vals_k = vals[idx]
    vecs_k = vecs[:, idx]
    X_f = Qf @ vecs_k
    X_u = np.column_stack([T, X_f]) 
    S = np.zeros((k, k))
    S[(d+1):, (d+1):] = np.diag(np.abs(vals_k))
    X = X_u[inverse_indices]
    return X, S

def basis_cr(x, k, knots=None):
    if knots is None: knots = get_knots(x, k)
    h = np.diff(knots)
    k = len(knots)
    B = np.zeros((k-2, k-2))
    D = np.zeros((k-2, k))
    for i in range(k-2):
        D[i, i] = 1/h[i]
        D[i, i+1] = -1/h[i] - 1/h[i+1]
        D[i, i+2] = 1/h[i+1]
    for i in range(k-2):
        B[i, i] = (h[i] + h[i+1])/3
        if i < k-3:
            B[i, i+1] = h[i+1]/6
            B[i+1, i] = h[i+1]/6
    B_inv = np.linalg.inv(B)
    S = D.T @ B_inv @ D
    X = np.zeros((len(x), k))
    for j in range(k):
        y = np.zeros(k)
        y[j] = 1.0
        cs = CubicSpline(knots, y, bc_type='natural')
        X[:, j] = cs(x)
    return X, S

def basis_cc(x, k, knots=None):
    if knots is None: knots = get_knots(x, k)
    X, S = basis_cr(x, k, knots)
    X_cyclic = np.array(X)
    X_cyclic[:, 0] += X_cyclic[:, -1]
    X_cyclic = X_cyclic[:, :-1]
    S_cyclic = np.array(S)[:-1, :-1]
    return X_cyclic, S_cyclic

def tensor_product(Xs, Ss):
    n_obs = Xs[0].shape[0]
    X = Xs[0]
    for i in range(1, len(Xs)):
        X = (X[:, :, None] * Xs[i][:, None, :]).reshape(n_obs, -1)
    S_total = []
    for i, Si in enumerate(Ss):
        curr_S = Si
        for j in range(i): curr_S = np.kron(np.eye(Xs[j].shape[1]), curr_S)
        for j in range(i + 1, len(Xs)): curr_S = np.kron(curr_S, np.eye(Xs[j].shape[1]))
        S_total.append(curr_S)
    return X, S_total

class gam:
    @staticmethod
    def s(x, k=10, bs="tp", constraint=True):
        if bs == "tp": X, S = basis_tp(x, k)
        elif bs == "cr": X, S = basis_cr(x, k)
        elif bs == "cc": X, S = basis_cc(x, k)
        if constraint: X, S, Z = apply_sum_to_zero_constraint(X, S)
        return X, S

    @staticmethod
    def te(vars_list, k_list, bs_list, constraint=True):
        Xs, Ss_list = [], []
        for x, k, bs in zip(vars_list, k_list, bs_list):
            Xi, Si = gam.s(x, k, bs, constraint=False)
            Xs.append(Xi); Ss_list.append(Si)
        X_te, S_te_list = tensor_product(Xs, Ss_list)
        if constraint: X_te, S_te_list, Z = apply_sum_to_zero_constraint(X_te, S_te_list)
        return X_te, S_te_list

    @staticmethod
    def hgam(vars_list, k_list, bs_list, group, type="G"):
        unique_groups = np.unique(group) if group is not None else []
        n_obs = len(vars_list[0])
        n_lev = len(unique_groups)
        S_final = []
        
        if type == "G":
            X_glob, S_glob = gam.te(vars_list, k_list, bs_list, constraint=True)
            X_full = np.column_stack([np.ones((n_obs, 1)), X_glob])
            S_final.extend(S_glob)
            
        elif type == "GS":
            X_glob, S_glob = gam.te(vars_list, k_list, bs_list, constraint=True)
            X_hier_blocks, S_hier_list = [], []
            P = np.eye(n_obs) - X_glob @ np.linalg.pinv(X_glob)
            for g in unique_groups:
                mask = (group == g)
                Xi, Si = gam.te(vars_list, k_list, bs_list, constraint=False)
                X_g = np.zeros_like(Xi); X_g[mask] = Xi[mask]
                X_hier_blocks.append(P @ X_g); S_hier_list.append(Si)
            X_full = np.column_stack([np.ones((n_obs, 1)), X_glob] + X_hier_blocks)
            S_final.extend(S_glob)
            for i in range(len(S_hier_list[0])):
                S_final.append(linalg.block_diag(*[S_hier_list[j][i] for j in range(n_lev)]))
                
        elif type == "GI":
            X_ints = pd.get_dummies(group, dtype=float).values
            X_glob, S_glob = gam.te(vars_list, k_list, bs_list, constraint=True)
            X_hier_blocks, S_hier_list = [], []
            P = np.eye(n_obs) - X_glob @ np.linalg.pinv(X_glob)
            for g in unique_groups:
                mask = (group == g)
                Xi, Si = gam.te(vars_list, k_list, bs_list, constraint=True)
                X_g = np.zeros_like(Xi); X_g[mask] = Xi[mask]
                X_hier_blocks.append(P @ X_g); S_hier_list.extend(Si)
            X_full = np.column_stack([X_ints, X_glob] + X_hier_blocks)
            S_final.extend(S_glob); S_final.extend(S_hier_list)
            
        elif type == "S":
            X_hier_blocks, S_hier_list = [], []
            for g in unique_groups:
                mask = (group == g)
                Xi, Si = gam.te(vars_list, k_list, bs_list, constraint=False)
                X_g = np.zeros_like(Xi); X_g[mask] = Xi[mask]
                X_hier_blocks.append(X_g); S_hier_list.append(Si)
            X_full = np.column_stack([np.ones((n_obs, 1))] + X_hier_blocks)
            for i in range(len(S_hier_list[0])):
                S_final.append(linalg.block_diag(*[S_hier_list[j][i] for j in range(n_lev)]))
                
        elif type == "I":
            X_ints = pd.get_dummies(group, dtype=float).values
            X_hier_blocks, S_hier_list = [], []
            for g in unique_groups:
                mask = (group == g)
                Xi, Si = gam.te(vars_list, k_list, bs_list, constraint=True)
                X_g = np.zeros_like(Xi); X_g[mask] = Xi[mask]
                X_hier_blocks.append(X_g); S_hier_list.extend(Si)
            X_full = np.column_stack([X_ints] + X_hier_blocks)
            S_final.extend(S_hier_list)

        return jnp.array(np.nan_to_num(X_full)), [jnp.array(s) for s in S_final]
