"""
GGMM (Generalized Graphical Mixed Model) in BI.

Phylogenetic SEM with OU process on a tree.
Joint model for 3 mammal traits: ln_metabolism, ln_range, ln_size.
SEM paths: ln_size -> ln_metabolism (b1), ln_size -> ln_range (b2).

Uses JAX-accelerated MAP estimation comparable to phylolm's MLE.
Reference: Thorson (2026), Methods in Ecology and Evolution.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from scipy.optimize import minimize
import pandas as pd
import dendropy

DATA_DIR = "/home/sosa/work/BI/Test/GGMM (WIP)/graphical_mixed_model/data"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

PARAM_NAMES = [
    "b1", "b2",
    "xbar_metabolism", "xbar_range", "xbar_size",
    "ln_theta_metabolism", "ln_theta_range", "ln_theta_size",
    "ln_sigma2_metabolism", "ln_sigma2_range", "ln_sigma2_size",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    tree = dendropy.Tree.get(
        path=os.path.join(DATA_DIR, "VertTree_mammals.tre"),
        schema="newick",
        preserve_underscores=True,
    )
    lengths = [e.length for e in tree.edges() if e.length is not None]
    max_edge = max(lengths)
    for e in tree.edges():
        if e.length is not None:
            e.length /= max_edge

    traits = pd.read_csv(
        os.path.join(DATA_DIR, "PanTHERIA_1-0_WR05_Aug2008.txt"),
        sep="\t", low_memory=False,
    )
    tip_labels_lower = {t.label.lower() for t in tree.taxon_namespace}
    traits["traits_binom"] = traits["MSW05_Binomial"].str.replace(" ", "_").str.lower()
    traits = traits[traits["traits_binom"].isin(tip_labels_lower)].copy()
    traits = traits.reset_index(drop=True)

    def to_float(col):
        v = pd.to_numeric(traits[col], errors="coerce")
        return v.where(v != -999, other=np.nan).values

    data = pd.DataFrame({
        "ln_metabolism": np.log(to_float("18-1_BasalMetRate_mLO2hr")),
        "ln_range":      np.log(to_float("22-1_HomeRange_km2")),
        "ln_size":       np.log(to_float("5-1_AdultBodyMass_g")),
    }, index=traits["traits_binom"].values)

    data = data.dropna()
    print(f"Complete cases: {len(data)}")
    return tree, data


# ── Patristic distance matrix ─────────────────────────────────────────────────

def patristic_distances(tree, species_list):
    """Build n×n patristic distance matrix via root-to-node depths + LCA."""
    species_set = set(species_list)
    n = len(species_list)

    depth = {}
    leaf_node = {}
    for node in tree.preorder_node_iter():
        parent = node.parent_node
        edge_len = node.edge.length if node.edge.length is not None else 0.0
        depth[node] = (depth[parent] if parent is not None else 0.0) + edge_len
        label = node.taxon.label.lower() if node.taxon is not None else None
        if label in species_set:
            leaf_node[label] = node

    D = np.zeros((n, n))
    for i, si in enumerate(species_list):
        for j in range(i + 1, n):
            sj = species_list[j]
            ni, nj = leaf_node[si], leaf_node[sj]
            ancestors_i = {}
            node = ni
            while node is not None:
                ancestors_i[node] = depth[node]
                node = node.parent_node
            node = nj
            d_mrca = 0.0
            while node is not None:
                if node in ancestors_i:
                    d_mrca = ancestors_i[node]
                    break
                node = node.parent_node
            d = depth[leaf_node[si]] + depth[leaf_node[sj]] - 2.0 * d_mrca
            D[i, j] = d
            D[j, i] = d
    return D


# ── Log-likelihood (JAX, JIT-compiled) ───────────────────────────────────────

def make_nll_fn(y_obs_np, D_np):
    """
    Return JIT-compiled (neg profile log-likelihood, gradient) function.

    sigma2 is profiled out analytically (as in phylolm):
        sigma2_hat = (r' C^{-1} r) / n
    so the reduced NLL over (b1,b2,xbar,ln_theta) is:
        sum_k  0.5 * (n * log(sigma2_hat_k) + log|C_k| + n)
    """
    y = jnp.array(y_obs_np)   # (n, 3)
    D = jnp.array(D_np)       # (n, n)
    n = y.shape[0]
    log2pi = jnp.log(2.0 * jnp.pi)
    jitter = 1e-6 * jnp.eye(n)

    def profile_nll(params):
        # params: [b1, b2, xbar0, xbar1, xbar2, ln_theta0, ln_theta1, ln_theta2]
        b1, b2 = params[0], params[1]
        xbar = params[2:5]
        theta = jnp.exp(params[5:8])

        residuals = jnp.stack([
            y[:, 0] - b1 * y[:, 2] - xbar[0],
            y[:, 1] - b2 * y[:, 2] - xbar[1],
            y[:, 2] - xbar[2],
        ], axis=1)  # (n, 3)

        total_nll = 0.0
        for k in range(3):
            C = jnp.exp(-theta[k] * D) + jitter  # OU correlation matrix
            L = jnp.linalg.cholesky(C)
            r = residuals[:, k]
            Linv_r = jax.scipy.linalg.solve_triangular(L, r, lower=True)
            quad = jnp.dot(Linv_r, Linv_r)   # r' C^{-1} r
            sigma2_hat = quad / n             # profiled MLE of sigma2
            log_det_C = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            # Profile log-likelihood: -0.5*(n*log(2pi) + n*log(sigma2) + log|C| + n)
            total_nll += 0.5 * (n * log2pi + n * jnp.log(sigma2_hat) + log_det_C + n)
        return total_nll

    nll_and_grad = jit(value_and_grad(profile_nll))

    def scipy_nll_grad(params_np):
        params_jax = jnp.array(params_np)
        val, g = nll_and_grad(params_jax)
        return float(val), np.array(g, dtype=np.float64)

    return scipy_nll_grad


def compute_sigma2(y_obs_np, D_np, params_opt):
    """Compute profiled sigma2 estimates at optimum."""
    y, D = np.array(y_obs_np), np.array(D_np)
    n = len(y)
    b1, b2 = params_opt[0], params_opt[1]
    xbar = params_opt[2:5]
    theta = np.exp(params_opt[5:8])
    jitter = 1e-6 * np.eye(n)

    residuals = np.column_stack([
        y[:, 0] - b1 * y[:, 2] - xbar[0],
        y[:, 1] - b2 * y[:, 2] - xbar[1],
        y[:, 2] - xbar[2],
    ])

    sigma2 = []
    for k in range(3):
        C = np.exp(-theta[k] * D) + jitter
        Cinv_r = np.linalg.solve(C, residuals[:, k])
        sigma2.append(float(np.dot(residuals[:, k], Cinv_r) / n))
    return np.array(sigma2)


# ── MAP optimisation ──────────────────────────────────────────────────────────

def run_map(y_obs, D, tol=1e-9, maxiter=5000):
    nll_grad = make_nll_fn(y_obs, D)

    max_dist = D.max()
    y_means = y_obs.mean(axis=0)
    y_vars  = y_obs.var(axis=0)

    best_nll = np.inf
    best_x = None

    # Profile NLL: optimise over [b1,b2,xbar(3),ln_theta(3)] — 8 params
    starts = [
        {"ln_theta_init": np.log([1.0, 2.0, 0.5]),  "b_init": [0.5, 0.5]},
        {"ln_theta_init": np.log([3.0, 5.0, 0.6]),  "b_init": [0.6, 0.9]},
        {"ln_theta_init": np.log([0.5, 3.0, 0.2]),  "b_init": [0.5, 0.9]},
        {"ln_theta_init": np.log([5.0, 8.0, 1.0]),  "b_init": [0.7, 1.0]},
    ]

    # Bounds: free for b, xbar; ln_theta in [-3, 7] (theta 0.05–1097)
    bounds = (
        [(None, None)] * 2    # b1, b2
        + [(None, None)] * 3  # xbar
        + [(-3.0, 7.0)] * 3  # ln_theta
    )

    for start in starts:
        x0 = np.zeros(8)
        x0[0], x0[1] = start["b_init"]
        x0[2] = y_means[0]
        x0[3] = y_means[1]
        x0[4] = y_means[2]
        x0[5:8] = start["ln_theta_init"]

        result = minimize(
            nll_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-7},
        )
        if result.fun < best_nll:
            best_nll = result.fun
            best_x = result.x
            print(f"  NLL={result.fun:.4f}, nit={result.nit}, msg={result.message}")

    print(f"Best profile NLL = {best_nll:.4f}")
    return best_x


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ggmm():
    print("Loading data...")
    tree, data = load_data()

    species = list(data.index)
    print("Computing patristic distances...")
    D = patristic_distances(tree, species)

    y_obs = data[["ln_metabolism", "ln_range", "ln_size"]].values

    # Warm up JAX JIT
    print("Compiling JAX model...")
    _ = make_nll_fn(y_obs[:10], D[:10, :10])(np.zeros(11))

    params_opt = run_map(y_obs, D)
    theta  = np.exp(params_opt[5:8])
    # sigma2_marginal = profiled MLE of marginal variance
    # sigma2_diffusion = sigma2_marginal * 2 * theta  (matches phylolm's sigma2)
    sigma2_marg = compute_sigma2(y_obs, D, params_opt)
    sigma2_diff = sigma2_marg * 2.0 * theta

    results = {
        "b1":                float(params_opt[0]),
        "b2":                float(params_opt[1]),
        "xbar_metabolism":   float(params_opt[2]),
        "xbar_range":        float(params_opt[3]),
        "xbar_size":         float(params_opt[4]),
        "theta_metabolism":  float(theta[0]),
        "theta_range":       float(theta[1]),
        "theta_size":        float(theta[2]),
        "sigma2_metabolism": float(sigma2_diff[0]),
        "sigma2_range":      float(sigma2_diff[1]),
        "sigma2_size":       float(sigma2_diff[2]),
    }

    bi_df = pd.DataFrame([{"parameter": k, "estimate": v} for k, v in results.items()])
    bi_df.to_csv(os.path.join(OUT_DIR, "bi_estimates.csv"), index=False)
    print("\nBI estimates:")
    print(bi_df.to_string(index=False))
    return bi_df


if __name__ == "__main__":
    run_ggmm()
