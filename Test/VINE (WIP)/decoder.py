"""
decoder.py
==========
Differentiable UPGMA decoder via the Straight-Through Estimator (STE).

Overview
--------
Standard UPGMA requires a non-differentiable argmin to identify which pair
of nodes to merge at each step. This breaks the gradient flow needed for
variational inference. We solve this using two complementary tricks:

1. **Soft-UPGMA (forward smoothing)**:
   Replace hard argmin with a soft temperature-scaled softmax:

       W_ij = softmax(-D_ij / τ)

   As τ → 0, W_ij → one-hot(argmin), recovering the discrete algorithm.
   At higher τ, W_ij is a smooth, differentiable distribution over merge
   candidates.

2. **Straight-Through Estimator (STE)**:
   [Bengio et al., 2013, "Estimating or Propagating Gradients Through
   Stochastic Neurons for Conditional Computation"]

   The STE trick:
       z_ste = z_soft + stop_gradient(z_hard - z_soft)

   On the **forward pass**: z_ste == z_hard  (exact discrete topology)
   On the **backward pass**: ∂z_ste/∂θ == ∂z_soft/∂θ  (smooth gradients)

   This lets us build an *exactly* discrete tree for the likelihood
   computation while still propagating meaningful gradients through the
   differentiable soft approximation.

VINE Reference
--------------
The VINE algorithm ("Variational Inference with Node Embeddings") uses
this STE-based relaxation to make the full phylogenetic inference pipeline
end-to-end differentiable. The latent embeddings Z_i are optimised so that
the pairwise distances D_ij induce the correct tree topology under the
Felsenstein likelihood.

Algorithm (N-1 merge steps via jax.lax.scan)
--------------------------------------------
At each merge step k  (k = 0, 1, …, N-2):
  1. Mask the already-merged clusters so they don't participate.
  2. Compute soft merge weights W = softmax(-D_active / τ).
  3. Hard merge: (i*, j*) = argmin over active pairs.
  4. STE: combine hard one-hot with soft weights.
  5. New cluster distance (UPGMA): average of child distances.
  6. Record the parent–child tree relationships and branch lengths.

Data structures
---------------
We operate on a flat node index space of size 2N-1:
  - Nodes 0..N-1  : leaf taxa (fixed, never merged into a new cluster).
  - Nodes N..2N-2 : internal nodes created one per merge step.

The scan body carries a "state" tuple:
  (D, active_mask, node_sizes, next_internal_id, edges)

where:
  D               : (2N-1, 2N-1) current pairwise distance matrix.
  active_mask     : (2N-1,) bool — which nodes are currently available.
  node_sizes      : (2N-1,) int  — number of leaves below each node (UPGMA weight).
  next_internal_id: scalar int   — index of the next internal node to create.
  edges           : (2*(N-1), 3) float — accumulated (parent, child, branch_length).
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_INF = 1e9          # Large sentinel for masked/diagonal distances
_TAU_INIT = 5.0     # Starting temperature
_TAU_FINAL = 0.1    # Ending temperature
_EPS = 1e-8


# ---------------------------------------------------------------------------
# State carried through the scan loop
# ---------------------------------------------------------------------------

class UPGMAState(NamedTuple):
    """Carry-state for the jax.lax.scan UPGMA loop.

    All arrays are padded to size (max_nodes,) = (2N-1,) so that JAX can
    trace through static shapes.

    Fields
    ------
    D               : (max_nodes, max_nodes) — current UPGMA distance matrix.
    active          : (max_nodes,) bool      — True iff node participates.
    sizes           : (max_nodes,) int       — leaf-count below each node.
    next_id         : scalar int             — next internal node index.
    parent_buf      : (2*(N-1),) int         — accumulated parent ids.
    child_buf       : (2*(N-1),) int         — accumulated child ids.
    branch_buf      : (2*(N-1),) float       — accumulated branch lengths.
    edge_ptr        : scalar int             — write pointer into edge buffers.
    """
    D: jax.Array
    active: jax.Array
    sizes: jax.Array
    next_id: jax.Array
    parent_buf: jax.Array
    child_buf: jax.Array
    branch_buf: jax.Array
    edge_ptr: jax.Array


# ---------------------------------------------------------------------------
# Core STE helpers
# ---------------------------------------------------------------------------

def _soft_merge_weights(
    D: jax.Array,
    active: jax.Array,
    tau: float,
) -> jax.Array:
    """
    Compute soft merge weights using a masked temperature softmax.

    W_ij = softmax(-D_ij / τ)  over valid (active, off-diagonal) pairs.

    Invalid entries (inactive nodes or diagonal) are set to -∞ before
    the softmax so they contribute zero weight.

    Args:
        D      : (M, M) distance matrix (M = 2N-1).
        active : (M,)   bool mask of currently active nodes.
        tau    : Temperature τ > 0.

    Returns:
        W: (M, M) soft weight matrix, sums to 1 over all valid entries.
    """
    M = D.shape[0]

    # Pair mask: both nodes active AND i ≠ j
    pair_mask = active[:, None] & active[None, :]          # (M, M) bool
    pair_mask = pair_mask & ~jnp.eye(M, dtype=bool)        # exclude diagonal

    # Replace invalid entries with -inf before softmax
    logits = jnp.where(pair_mask, -D / tau, -_INF)        # (M, M)

    # Flatten, softmax over all pairs, reshape back
    flat_logits = logits.reshape(-1)
    flat_weights = jax.nn.softmax(flat_logits)
    return flat_weights.reshape(M, M)


def _hard_merge_argmin(
    D: jax.Array,
    active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Exact argmin over active off-diagonal entries (forward pass).

    To preserve symmetry we only consider upper-triangular pairs, then
    recover both (i*, j*) indices.

    Args:
        D      : (M, M) distance matrix.
        active : (M,)   bool mask.

    Returns:
        i_star, j_star: scalar integer indices of the closest pair.
    """
    M = D.shape[0]
    # Mask: upper triangle, both active, off-diagonal
    triu = jnp.triu(jnp.ones((M, M), dtype=bool), k=1)
    pair_mask = triu & active[:, None] & active[None, :]

    # Mask non-pairs to +∞
    D_masked = jnp.where(pair_mask, D, _INF)

    # Flat argmin → 2D index
    flat_idx = jnp.argmin(D_masked.reshape(-1))
    i_star = flat_idx // M
    j_star = flat_idx % M
    return i_star, j_star


def _ste_one_hot(
    soft_weights: jax.Array,
    i_star: jax.Array,
    j_star: jax.Array,
    M: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Apply the Straight-Through Estimator to combine hard and soft merge.

    Constructs two (M,)-dimensional "selection vectors":
      - hard_i[k] = 1 iff k == i_star   (one-hot, discrete)
      - hard_j[k] = 1 iff k == j_star   (one-hot, discrete)

    Then forms STE vectors:
      ste_i = soft_i + stop_gradient(hard_i - soft_i)
      ste_j = soft_j + stop_gradient(hard_j - soft_j)

    Forward: ste_{i,j} == hard_{i,j}  (exact topology)
    Backward: ∂ste/∂θ == ∂soft/∂θ  (gradient flows through soft weights)

    The soft_i and soft_j marginals are derived from the joint soft_weights
    by summing over the other dimension:
      soft_i[k] = Σ_l W[k,l]     (marginal probability node k is "first")
      soft_j[l] = Σ_k W[k,l]     (marginal probability node l is "second")

    Args:
        soft_weights : (M, M) soft merge weight matrix.
        i_star, j_star: scalar hard merge indices.
        M            : int, number of nodes.

    Returns:
        ste_i, ste_j: (M,) STE selection vectors.
    """
    # ------------------------------------------------------------------
    # Soft marginals
    # ------------------------------------------------------------------
    soft_i = soft_weights.sum(axis=1)   # (M,) marginal over first node
    soft_j = soft_weights.sum(axis=0)   # (M,) marginal over second node

    # ------------------------------------------------------------------
    # Hard one-hots
    # ------------------------------------------------------------------
    hard_i = jax.nn.one_hot(i_star, M)  # (M,)
    hard_j = jax.nn.one_hot(j_star, M)  # (M,)

    # ------------------------------------------------------------------
    # STE combination
    # Gradient flows through soft_i / soft_j; value equals hard_i / hard_j.
    # ------------------------------------------------------------------
    ste_i = soft_i + jax.lax.stop_gradient(hard_i - soft_i)
    ste_j = soft_j + jax.lax.stop_gradient(hard_j - soft_j)

    return ste_i, ste_j


# ---------------------------------------------------------------------------
# UPGMA distance update
# ---------------------------------------------------------------------------

def _upgma_distance_update(
    D: jax.Array,
    sizes: jax.Array,
    ste_i: jax.Array,
    ste_j: jax.Array,
    new_id: jax.Array,
    M: int,
) -> jax.Array:
    """
    Update the distance matrix after merging nodes selected by ste_i / ste_j.

    UPGMA formula for the distance from new cluster (i∪j) to any other k:

        D_{(i∪j), k} = (n_i * D_ik + n_j * D_jk) / (n_i + n_j)

    We use a differentiable implementation: instead of hard indexing, we
    compute the new cluster's distances as a weighted sum of all rows/cols,
    weighted by the STE vectors ste_i and ste_j.

    The branch length for each child c ∈ {i, j} is half the distance
    between i and j (the UPGMA half-distance):

        b_c = 0.5 * D_ij

    Args:
        D       : (M, M) current distance matrix.
        sizes   : (M,)   leaf counts.
        ste_i   : (M,)   STE selection vector for first node.
        ste_j   : (M,)   STE selection vector for second node.
        new_id  : scalar — index of the new internal node.
        M       : int    — total nodes.

    Returns:
        D_new: (M, M) updated distance matrix with row/col new_id filled in.
    """
    # ------------------------------------------------------------------
    # Effective sizes (differentiable via soft selection)
    # ------------------------------------------------------------------
    n_i = jnp.dot(ste_i, sizes.astype(jnp.float32))   # scalar
    n_j = jnp.dot(ste_j, sizes.astype(jnp.float32))   # scalar
    n_new = n_i + n_j + _EPS

    # ------------------------------------------------------------------
    # Current distances from "i" and "j" to all other nodes (soft select)
    # ------------------------------------------------------------------
    d_i = ste_i @ D   # (M,): weighted sum of rows → distance from virtual "i"
    d_j = ste_j @ D   # (M,): weighted sum of rows → distance from virtual "j"

    # New cluster distances (UPGMA average)
    d_new = (n_i * d_i + n_j * d_j) / n_new   # (M,)

    # ------------------------------------------------------------------
    # Write the new row and column into D
    # ------------------------------------------------------------------
    # Use index update to place d_new at row new_id and col new_id
    D_new = D.at[new_id, :].set(d_new)
    D_new = D_new.at[:, new_id].set(d_new)
    D_new = D_new.at[new_id, new_id].set(0.0)

    return D_new


# ---------------------------------------------------------------------------
# Single scan step
# ---------------------------------------------------------------------------

def _upgma_step(
    state: UPGMAState,
    _x,             # dummy per-step input from jax.lax.scan (xs=None → None each step)
    tau: float = 1.0,
) -> tuple[UPGMAState, None]:
    """
    One UPGMA merge step executed inside jax.lax.scan.

    This function is traced by JAX so ALL shapes must be static and ALL
    branches must be expressed via jnp.where rather than Python if/else.

    Args:
        state : UPGMAState carry.
        _x    : Dummy per-step scan input (ignored; scan passes None when xs=None).
        tau   : Temperature (closed over via lambda in soft_upgma).

    Returns:
        (new_state, None)  — new carry, no per-step output (edges stored in carry).
    """
    D       = state.D
    active  = state.active
    sizes   = state.sizes
    nid     = state.next_id    # Index of new internal node to create
    ptr     = state.edge_ptr
    M       = D.shape[0]

    # ------------------------------------------------------------------
    # 1. Compute soft merge weights (backward path, STE)
    # ------------------------------------------------------------------
    W = _soft_merge_weights(D, active, tau)

    # ------------------------------------------------------------------
    # 2. Hard argmin (forward path, STE)
    # ------------------------------------------------------------------
    i_star, j_star = _hard_merge_argmin(D, active)

    # ------------------------------------------------------------------
    # 3. STE selection vectors
    # ------------------------------------------------------------------
    ste_i, ste_j = _ste_one_hot(W, i_star, j_star, M)

    # ------------------------------------------------------------------
    # 4. Branch lengths:  b = 0.5 * D_{i*, j*}
    # ------------------------------------------------------------------
    d_ij = D[i_star, j_star]    # scalar (exact, forward pass)
    branch_i = 0.5 * d_ij
    branch_j = 0.5 * d_ij

    # ------------------------------------------------------------------
    # 5. Update the distance matrix (differentiable)
    # ------------------------------------------------------------------
    D_new = _upgma_distance_update(D, sizes, ste_i, ste_j, nid, M)

    # ------------------------------------------------------------------
    # 6. Update active mask: deactivate i* and j*, activate nid
    # ------------------------------------------------------------------
    active_new = active.at[i_star].set(False)
    active_new = active_new.at[j_star].set(False)
    active_new = active_new.at[nid].set(True)

    # ------------------------------------------------------------------
    # 7. Update sizes: size[nid] = size[i*] + size[j*]
    # ------------------------------------------------------------------
    sizes_new = sizes.at[nid].set(sizes[i_star] + sizes[j_star])

    # ------------------------------------------------------------------
    # 8. Record edges in pre-allocated buffers
    # ------------------------------------------------------------------
    # Two edges per step: (nid→i*, nid→j*)
    # ptr increments by 2 each step
    parent_buf = state.parent_buf.at[ptr].set(nid).at[ptr + 1].set(nid)
    child_buf  = state.child_buf.at[ptr].set(i_star).at[ptr + 1].set(j_star)
    branch_buf = state.branch_buf.at[ptr].set(branch_i).at[ptr + 1].set(branch_j)

    new_state = UPGMAState(
        D=D_new,
        active=active_new,
        sizes=sizes_new,
        next_id=nid + 1,
        parent_buf=parent_buf,
        child_buf=child_buf,
        branch_buf=branch_buf,
        edge_ptr=ptr + 2,
    )
    return new_state, None


# ---------------------------------------------------------------------------
# Public API: soft_upgma
# ---------------------------------------------------------------------------

def soft_upgma(
    D: jax.Array,
    tau: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Differentiable UPGMA decoder via Soft-argmin + Straight-Through Estimator.

    Runs N-1 merge steps using jax.lax.scan.  Each step is O(N²) in the
    distance matrix operations; total complexity O(N³) per forward pass —
    identical to classical UPGMA.

    The STE ensures:
      - Forward pass: exact discrete UPGMA topology (correct tree structure).
      - Backward pass: smooth gradients from the soft temperature-weighted
        version, enabling gradient-based optimisation of Z → D → tree.

    Args:
        D   : (N, N) pairwise distance matrix, D[i,i] = 0 ≥ 0.
        tau : Gumbel-Softmax temperature τ.  High τ ≈ uniform (explore);
              low τ ≈ hard argmin (exploit).  Annealed externally.

    Returns:
        parents  : (2*(N-1),) int array  — parent node index for each edge.
        children : (2*(N-1),) int array  — child  node index for each edge.
        branches : (2*(N-1),) float array — branch length for each edge.
        final_D  : (2N-1, 2N-1) final pairwise distance matrix (includes
                   internal nodes), useful for debugging / convergence checks.

    Notes
    -----
    Node indexing convention:
      - Leaf nodes:     indices 0 … N-1   (input taxa)
      - Internal nodes: indices N … 2N-2  (created one per merge step)
      - Root:           index 2N-2        (last merge)
    """
    N = D.shape[0]
    max_nodes = 2 * N - 1        # Total nodes (leaves + internals)
    n_edges   = 2 * (N - 1)      # Two edges per merge step

    # ------------------------------------------------------------------
    # Expand D to (max_nodes, max_nodes) — pad with 0s for internal nodes.
    # Internal rows/cols will be filled in during the scan.
    # ------------------------------------------------------------------
    D_full = jnp.zeros((max_nodes, max_nodes))
    D_full = D_full.at[:N, :N].set(D)

    # ------------------------------------------------------------------
    # Initialise state
    # ------------------------------------------------------------------
    active = jnp.array(
        [True] * N + [False] * (N - 1), dtype=bool
    )  # (max_nodes,): only leaves initially active
    sizes  = jnp.array(
        [1] * N + [0] * (N - 1), dtype=jnp.int32
    )  # (max_nodes,)

    init_state = UPGMAState(
        D=D_full,
        active=active,
        sizes=sizes,
        next_id=jnp.array(N, dtype=jnp.int32),
        parent_buf=jnp.zeros(n_edges, dtype=jnp.int32),
        child_buf=jnp.zeros(n_edges, dtype=jnp.int32),
        branch_buf=jnp.zeros(n_edges, dtype=jnp.float32),
        edge_ptr=jnp.array(0, dtype=jnp.int32),
    )

    # ------------------------------------------------------------------
    # Bind tau via a lambda closure — NOT functools.partial.
    # jax.lax.scan calls body(carry, x); xs=None passes x=None each step.
    # partial(fn, tau=tau) would cause 'multiple values for argument tau'
    # because scan also fills the second positional slot.
    # ------------------------------------------------------------------
    step_fn = lambda state, x: _upgma_step(state, x, tau=tau)  # noqa: E731

    # ------------------------------------------------------------------
    # Run N-1 merge steps
    # xs=None with length=N-1: scan passes None as x each step (ignored).
    # ------------------------------------------------------------------
    final_state, _ = jax.lax.scan(
        step_fn,
        init_state,
        xs=None,
        length=N - 1,
    )

    return (
        final_state.parent_buf,
        final_state.child_buf,
        final_state.branch_buf,
        final_state.D,
    )


# ---------------------------------------------------------------------------
# Hard tree extraction (for validation / Robinson-Foulds)
# ---------------------------------------------------------------------------

def extract_hard_tree(
    D: jax.Array,
) -> tuple[list[tuple[int, int, float]], int]:
    """
    Run standard (non-differentiable) UPGMA on a distance matrix and return
    the hard tree as a list of (parent, child, branch_length) triples.

    This is used in validation to compare against known ground-truth trees.

    Args:
        D: (N, N) numpy-compatible distance matrix.

    Returns:
        edges: List of (parent, child, branch_length) triples.
        root:  Index of the root node (= 2N-2).
    """
    import numpy as np
    D_np = np.array(D)
    N = D_np.shape[0]
    max_nodes = 2 * N - 1

    D_full = np.full((max_nodes, max_nodes), _INF)
    D_full[:N, :N] = D_np
    np.fill_diagonal(D_full, 0.0)

    active = list(range(N))
    sizes  = {i: 1 for i in range(N)}
    edges  = []
    next_id = N

    for _ in range(N - 1):
        # Find closest pair among active nodes
        best_d = _INF
        i_star = j_star = -1
        for ii, i in enumerate(active):
            for j in active[ii + 1:]:
                if D_full[i, j] < best_d:
                    best_d = D_full[i, j]
                    i_star, j_star = i, j

        nid = next_id
        next_id += 1
        bl = 0.5 * best_d
        edges.append((nid, i_star, bl))
        edges.append((nid, j_star, bl))

        # UPGMA distance update
        ni, nj = sizes[i_star], sizes[j_star]
        nn = ni + nj
        for k in active:
            if k == i_star or k == j_star:
                continue
            d_new = (ni * D_full[i_star, k] + nj * D_full[j_star, k]) / nn
            D_full[nid, k] = d_new
            D_full[k, nid] = d_new
        D_full[nid, nid] = 0.0

        # Update active list and sizes
        active.remove(i_star)
        active.remove(j_star)
        active.append(nid)
        sizes[nid] = nn

    root = next_id - 1
    return edges, root
