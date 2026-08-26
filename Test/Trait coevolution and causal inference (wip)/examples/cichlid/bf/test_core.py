"""Check the hand-translated core against independent references.

ksolve   -> scipy.linalg.solve_continuous_lyapunov
traverse -> a plain python loop transcribed directly from the Stan source
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm as sp_expm

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/sebastian_sosa/phylo/examples/cichlid/bf")
from gdpm_core import ksolve, build_A, segment_quantities, traverse  # noqa: E402

D = json.load(open("/home/sebastian_sosa/phylo/examples/cichlid/data/standata.json"))
T = json.load(open("/home/sebastian_sosa/phylo/examples/cichlid/data/true_params.json"))
J = D["J"]
A_true = np.array(T["A"], dtype=float)
Q_true = np.array(T["Q"], dtype=float)

# --- 1. ksolve -------------------------------------------------------------
mine = np.array(ksolve(jnp.array(A_true), jnp.array(Q_true)))
ref = solve_continuous_lyapunov(A_true, -Q_true)
print("ksolve  max abs diff vs scipy :", np.abs(mine - ref).max())
assert np.allclose(mine, ref, atol=1e-9), "ksolve convention is wrong"

# --- 2. effects_mat ticker order ------------------------------------------
eff = np.array(D["effects_mat"])
rows, cols = [], []
for i in range(J):
    for j in range(J):
        if i != j and eff[i, j] == 1:
            rows.append(i)
            cols.append(j)
print("A_offdiag slots (0-based):", list(zip(rows, cols)))
off_true = np.array([A_true[i, j] for i, j in zip(rows, cols)])
print("true A_offdiag in ticker order:", off_true)
A_rebuilt = np.array(build_A(jnp.array(np.diag(A_true)), jnp.array(off_true),
                             jnp.array(rows), jnp.array(cols)))
assert np.allclose(A_rebuilt, A_true), "build_A does not reproduce A"
print("build_A round-trip           : OK")

# --- 3. traversal vs a literal transcription of the Stan loop --------------
node_seq = np.array(D["node_seq"][0]) - 1
parent = np.array(D["parent"][0]) - 1
ts = np.array(D["ts"][0], dtype=float)
tip = np.array(D["tip"][0])
N_seg = D["N_seg"]
print("length_index zeros (non-cached segments):",
      int((np.array(D["length_index"][0]) == 0).sum()), "/", N_seg)

rng = np.random.default_rng(0)
b = rng.normal(size=J)
eta_anc = rng.normal(size=J)
z_drift = rng.normal(size=(N_seg - 1, J))

Q_inf = ref
A_delta_np, L_np, As_np = [], [], []
for dt in ts:
    dt = max(dt, 0.0)  # root placeholder, see gdpm_core.segment_quantities
    Ad = sp_expm(A_true * dt)
    V = Q_inf - Ad @ Q_inf @ Ad.T
    V = 0.5 * (V + V.T)
    L = np.linalg.cholesky(V + 1e-10 * np.eye(J))
    As = np.linalg.solve(A_true, Ad - np.eye(J))
    As = 0.5 * (As + As.T)
    A_delta_np.append(Ad); L_np.append(L); As_np.append(As)
A_delta_np, L_np, As_np = map(np.array, (A_delta_np, L_np, As_np))

# literal Stan loop
eta_ref = np.zeros((N_seg, J))
eta_ref[node_seq[0]] = eta_anc
for i in range(1, N_seg):
    drift = L_np[i] @ z_drift[i - 1] if tip[i] == 0 else np.zeros(J)
    eta_ref[node_seq[i]] = A_delta_np[i] @ eta_ref[parent[i]] + As_np[i] @ b + drift

Ad_j, L_j, As_j = segment_quantities(jnp.array(A_true), jnp.array(Q_inf), jnp.array(ts))
print("segment_quantities A_delta diff:", np.abs(np.array(Ad_j) - A_delta_np).max())
print("segment_quantities L_VCV   diff:", np.abs(np.array(L_j) - L_np).max())
print("segment_quantities A_solve diff:", np.abs(np.array(As_j) - As_np).max())

eta_mine = np.array(traverse(jnp.array(eta_anc), jnp.array(z_drift), Ad_j, L_j, As_j,
                             jnp.array(b), jnp.array(node_seq), jnp.array(parent),
                             jnp.array(tip)))
print("traverse   max abs diff vs loop:", np.abs(eta_mine - eta_ref).max())
assert np.allclose(eta_mine, eta_ref, atol=1e-8), "traversal disagrees with Stan loop"
print("\nALL CORE CHECKS PASSED")

# --- 4. depth-parallel traversal must match the sequential one exactly -----
from gdpm_core import tree_levels, traverse_levels  # noqa: E402

lvl_seg, lvl_valid = tree_levels(node_seq, parent, N_seg)
print("\nlevels:", lvl_seg.shape[0], " max width:", lvl_seg.shape[1])
eta_lvl = np.array(traverse_levels(jnp.array(eta_anc), jnp.array(z_drift), Ad_j, L_j,
                                   As_j, jnp.array(b), jnp.array(node_seq),
                                   jnp.array(parent), jnp.array(tip),
                                   lvl_seg, lvl_valid))
print("traverse_levels max abs diff vs loop:", np.abs(eta_lvl - eta_ref).max())
assert np.allclose(eta_lvl, eta_ref, atol=1e-8), "depth-parallel traversal differs"
print("DEPTH-PARALLEL TRAVERSAL MATCHES")
