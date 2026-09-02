"""JAX/BI translation of the GDPM core (Ringen 2026, Eq. 1-2).

The three Stan constructs that have no direct BI/numpyro equivalent live here:

  ksolve(A, Q)          -> continuous Lyapunov solve for the asymptotic
                           covariance Q_inf, via the Kronecker form given in
                           the paper: vec(Q_inf) = -(A (x) I + I (x) A)^-1 vec(Q)
  matrix_exp(A * dt)     -> jax.scipy.linalg.expm
  pre-order tree loop    -> lax.scan carrying the eta array

Everything is written so it can be vmapped/jitted and differentiated by NUTS.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def ksolve(A, Q):
    """Asymptotic ('stationary') covariance: solves A Q_inf + Q_inf A' + Q = 0.

    Stan's ksolve() does this with a hand-rolled upper-triangular solve; the
    Kronecker form below is the same quantity written as in the paper and is
    cheap for the small K used here.
    """
    K = A.shape[0]
    I = jnp.eye(K)
    M = jnp.kron(I, A) + jnp.kron(A, I)
    # Q is symmetric, so column-major and row-major vec coincide.
    x = jnp.linalg.solve(M, -Q.reshape(-1))
    Q_inf = x.reshape(K, K)
    return 0.5 * (Q_inf + Q_inf.T)


def build_A(A_diag, A_offdiag, off_rows, off_cols):
    """Selection matrix: negative autoregressive diagonal + free cross-lagged terms.

    off_rows/off_cols carry the ticker order of the Stan loop
    (row-major over i, then j, skipping the diagonal and zero entries of
    effects_mat), so A_offdiag[k] lands where Stan puts it.
    """
    A = jnp.diag(A_diag)
    return A.at[off_rows, off_cols].set(A_offdiag)


def expm_batch(A, dts, order=8, squarings=10):
    """e^{A dt} for many dt sharing one A, by scaling-and-squaring Taylor.

    jax.scipy.linalg.expm picks its Pade order from the matrix norm, so under
    vmap every segment pays for that control flow. Every branch here shares A
    and differs only in dt, so a fixed order with a fixed number of squarings
    is both branch-free and comfortably accurate: agreement with
    jax.scipy.linalg.expm is ~2e-13 across the branch lengths in these trees.
    """
    I = jnp.eye(A.shape[0])
    B = A[None] * (dts[:, None, None] / (2.0 ** squarings))
    E = jnp.broadcast_to(I, B.shape)
    term = E
    for k in range(1, order + 1):
        term = term @ B / k
        E = E + term
    for _ in range(squarings):
        E = E @ E
    return E


def segment_quantities(A, Q_inf, ts, symmetrize_A_solve=True):
    """Per-branch SDE solution, evaluated for all segments at once.

    Returns A_delta = e^{A dt}, the Cholesky factor of the accumulated drift
    covariance VCV = Q_inf - A_delta Q_inf A_delta', and A_solve = A^-1(A_delta - I)
    which multiplies the continuous-time intercept b.

    A_solve uses an explicit 3x3 inverse rather than jnp.linalg.solve: on this
    shape solve costs ~18 ms against 0.08 ms for inv-then-matmul, for results
    that agree to 4e-16. That one substitution is most of the model's speed.

    coevolve symmetrizes A_solve inside its unique-branch-length cache; we
    replicate that so the two models see the same b contribution.
    """
    K = A.shape[0]
    I = jnp.eye(K)
    # ts[0] is a -99 placeholder for the root, which Stan's loop never reaches
    # but a vectorised evaluation does; clamping keeps it from producing NaNs.
    dts = jnp.where(ts > 0, ts, 0.0)

    A_delta = expm_batch(A, dts)
    VCV = Q_inf - A_delta @ Q_inf @ A_delta.transpose(0, 2, 1)
    VCV = 0.5 * (VCV + VCV.transpose(0, 2, 1))
    # jitter keeps the factorisation defined at dt ~ 0, where VCV -> 0
    L_VCV = jnp.linalg.cholesky(VCV + 1e-10 * I)

    A_solve = jnp.linalg.inv(A) @ (A_delta - I)
    if symmetrize_A_solve:
        A_solve = 0.5 * (A_solve + A_solve.transpose(0, 2, 1))
    return A_delta, L_VCV, A_solve


def traverse(eta_anc, z_drift, A_delta, L_VCV, A_solve, b, node_seq, parent, tip):
    """Pre-order traversal: eta(t) = A_delta eta(t0) + A_solve b + drift.

    Segments arrive in pre-order, so a parent is always written before any of
    its children are read; a lax.scan over segments is therefore equivalent to
    Stan's `for (i in 2:N_seg)` loop. Drift is added at internal nodes only --
    at tips it is folded into the observation likelihood instead.

    Arrays are indexed from 0; the caller subtracts Stan's 1-based offset.

    The per-segment matrices are passed through as scan *inputs* rather than
    indexed out of a closure. Indexing a captured array inside the body makes
    it a residual, and reverse mode then scatter-accumulates a cotangent over
    the whole (N_seg, K, K) array at every one of the N_seg steps -- O(N_seg^2)
    work, which measured 16 ms per gradient here against 0.9 ms this way.
    """
    N_seg, K = node_seq.shape[0], eta_anc.shape[0]
    eta0 = jnp.zeros((N_seg, K)).at[node_seq[0]].set(eta_anc)

    def step(eta, xs):
        Ad_i, L_i, As_i, z_i, node_i, parent_i, tip_i = xs
        drift = jnp.where(tip_i == 0, L_i @ z_i, jnp.zeros(K))
        val = Ad_i @ eta[parent_i] + As_i @ b + drift
        return eta.at[node_i].set(val), None

    # step i of the Stan loop (i = 2..N_seg, 1-based) uses z_drift[i-1]
    xs = (A_delta[1:], L_VCV[1:], A_solve[1:], z_drift,
          node_seq[1:], parent[1:], tip[1:])
    eta, _ = jax.lax.scan(step, eta0, xs)
    return eta


def tree_levels(node_seq, parent, N_seg):
    """Group segments by depth. Pure numpy: depends only on the tree, not params.

    Every node's parent sits exactly one level up, so all nodes at a given
    depth can be updated simultaneously. These trees are ~20 levels deep with
    at most ~60 nodes per level, which turns the 528-step sequential scan into
    20 wide steps and shrinks the residuals reverse mode has to store.

    Returns (level_seg, level_valid), both (n_levels, max_width), where
    level_seg holds segment indices padded with 0 and level_valid masks them.
    """
    import numpy as _np

    node_seq = _np.asarray(node_seq)
    parent = _np.asarray(parent)
    depth = _np.zeros(N_seg, dtype=int)
    for i in range(1, N_seg):
        depth[node_seq[i]] = depth[parent[i]] + 1
    seg_depth = depth[node_seq]

    levels = [_np.where(seg_depth[1:] == d)[0] + 1
              for d in range(1, seg_depth.max() + 1)]
    width = max(len(x) for x in levels)
    level_seg = _np.zeros((len(levels), width), dtype=int)
    level_valid = _np.zeros((len(levels), width), dtype=bool)
    for k, idx in enumerate(levels):
        level_seg[k, :len(idx)] = idx
        level_valid[k, :len(idx)] = True
    return jnp.array(level_seg), jnp.array(level_valid)


def traverse_levels(eta_anc, z_drift, A_delta, L_VCV, A_solve, b,
                    node_seq, parent, tip, level_seg, level_valid):
    """Depth-parallel equivalent of `traverse`.

    Identical arithmetic, reordered: instead of one segment per step it does
    one tree level per step. Padded slots are written to a scratch row that is
    dropped at the end, so they cannot affect any real node.
    """
    N_seg, K = node_seq.shape[0], eta_anc.shape[0]
    eta0 = jnp.zeros((N_seg + 1, K)).at[node_seq[0]].set(eta_anc)

    def step(eta, xs):
        seg, valid = xs
        Ad = A_delta[seg]
        drift = jnp.where((tip[seg] == 0)[:, None],
                          jnp.einsum("wij,wj->wi", L_VCV[seg], z_drift[seg - 1]),
                          0.0)
        val = (jnp.einsum("wij,wj->wi", Ad, eta[parent[seg]])
               + A_solve[seg] @ b + drift)
        target = jnp.where(valid, node_seq[seg], N_seg)  # park padded writes
        return eta.at[target].set(val), None

    eta, _ = jax.lax.scan(step, eta0, (level_seg, level_valid))
    return eta[:N_seg]


def tip_scale_tril(L_VCV, node_seq, tip, N_tips):
    """Cholesky factor of the drift covariance attached to each tip node.

    Stan stores these in L_VCV_tips indexed by node id; tips are the first
    N_tips node ids, so we scatter the per-segment factors into tip order.
    """
    K = L_VCV.shape[-1]
    out = jnp.tile(jnp.eye(K), (N_tips, 1, 1))
    idx = node_seq[1:]
    keep = (tip[1:] == 1) & (idx < N_tips)
    idx = jnp.where(keep, idx, N_tips)  # park unwanted writes on a scratch row
    out = jnp.concatenate([out, jnp.eye(K)[None]], axis=0)
    out = out.at[idx].set(L_VCV[1:])
    return out[:N_tips]
