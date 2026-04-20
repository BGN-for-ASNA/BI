"""
likelihood.py
=============
JAX-compiled Felsenstein Pruning Algorithm with the JC69 substitution model.

The Felsenstein pruning algorithm computes the marginal likelihood of a
multiple sequence alignment Y = {y_1, …, y_N} given a tree topology T
and branch lengths b:

    P(Y | T, b) = Π_{sites s} P(Y_s | T, b)

For each site, we perform a post-order traversal (tips to root), computing
partial likelihoods L_v(a) = P(Y below v | node v is in state a).

At a tip node v with observed state y_v:
    L_v(a) = 1{a == y_v}

At an internal node v with children c_1, c_2 and branch lengths b_1, b_2:
    L_v(a) = [Σ_b P(b|a, b_1) L_{c1}(b)] × [Σ_b P(b|a, b_2) L_{c2}(b)]

At the root (uniform stationary distribution under JC69, π_a = 1/4):
    P(Y_s | T, b) = Σ_a π_a L_root(a)

JC69 Substitution Model
-----------------------
For nucleotides (4 states), the Jukes-Cantor 1969 model has a single rate
parameter λ (set to 1.0 for the standard parameterisation):

    P_{same}(t) = 1/4 + 3/4 · exp(-4λt/3)
    P_{diff}(t) = 1/4 - 1/4 · exp(-4λt/3)

Implementation Notes
--------------------
- Tree traversal is implemented with jax.lax.scan over DEPTH LEVELS of the
  tree, vectorising over all nodes at the same depth simultaneously.
  This replaces Python for loops and enables XLA compilation.
- Log-space arithmetic with jax.nn.logsumexp prevents numerical underflow
  for long sequences or many taxa.
- All arrays have static shapes determined at trace time (required by JAX JIT).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES = 4       # |{A, C, G, T}|
_JC69_RATE = 1.0   # JC69 rate parameter λ (standard = 1)


# ---------------------------------------------------------------------------
# JC69 Transition Matrix
# ---------------------------------------------------------------------------

def jc69_transition_matrix(t: jax.Array, rate: float = _JC69_RATE) -> jax.Array:
    """
    Compute the 4×4 JC69 transition probability matrix P(t).

    P_ij(t) = 1/4 + 3/4 · exp(-4λt/3)   if i == j
    P_ij(t) = 1/4 - 1/4 · exp(-4λt/3)   if i ≠ j

    Args:
        t    : Branch length (scalar or batched array), t ≥ 0.
        rate : Rate parameter λ (default 1.0).

    Returns:
        P: (4, 4) transition probability matrix, rows sum to 1.
    """
    exponent = jnp.exp(-4.0 * rate * t / 3.0)
    p_same = 0.25 + 0.75 * exponent
    p_diff = 0.25 - 0.25 * exponent

    # Build 4×4 matrix: diagonal = p_same, off-diagonal = p_diff
    P = jnp.full((N_STATES, N_STATES), p_diff)
    P = P.at[jnp.arange(N_STATES), jnp.arange(N_STATES)].set(p_same)
    return P


def batched_jc69(branch_lengths: jax.Array, rate: float = _JC69_RATE) -> jax.Array:
    """
    Compute a batch of JC69 transition matrices.

    Args:
        branch_lengths: (E,) array of branch lengths.
        rate: JC69 rate λ.

    Returns:
        (E, 4, 4) array of transition matrices.
    """
    return jax.vmap(lambda t: jc69_transition_matrix(t, rate))(branch_lengths)


# ---------------------------------------------------------------------------
# Log-space partial likelihood update
# ---------------------------------------------------------------------------

def _log_conditional_child(
    log_partial: jax.Array,   # (seq_len, 4)   log L_{child}(a)
    P: jax.Array,             # (4, 4)         transition matrix
) -> jax.Array:
    """
    Compute the log of the conditional likelihood contribution from one child.

    For each parent state a and each site s:
        C_s(a) = log Σ_b P(b|a) · L_child(b)
               = log Σ_b exp[ log P(b|a) + log L_child(b) ]

    Using log-sum-exp for numerical stability:
        C_s(a) = logsumexp_{b} (log P(a,b) + log_partial_s(b))

    Args:
        log_partial : (seq_len, 4) — log partial likelihoods at child.
        P           : (4, 4)      — transition matrix P[parent_state, child_state].

    Returns:
        (seq_len, 4) log conditional contributions.
    """
    log_P = jnp.log(P + 1e-30)    # (4, 4) avoid log(0)
    # log_P[a, b] = log P(b | a, t)
    # log_partial[s, b] for each site s
    # We want: out[s, a] = logsumexp_b (log_P[a, b] + log_partial[s, b])
    # Broadcast: (1, 4, 4) + (seq_len, 1, 4) → (seq_len, 4, 4)
    log_joint = log_P[None, :, :] + log_partial[:, None, :]  # (seq_len, 4, 4)
    # logsumexp over b (last dim)
    return jax.nn.logsumexp(log_joint, axis=-1)   # (seq_len, 4)


# ---------------------------------------------------------------------------
# Build a level-order traversal schedule
# ---------------------------------------------------------------------------

def _build_traversal_schedule(
    parents: jax.Array,
    children: jax.Array,
    n_taxa: int,
    max_nodes: int,
) -> tuple[list[list[tuple[int, int]]], int]:
    """
    Build a bottom-up (post-order) traversal schedule grouped by depth levels.

    Each level contains nodes that can be processed in parallel because all
    their children have already been computed.

    This is a Python-time computation (not JAX-traced); it runs once during
    JIT tracing to produce the static schedule that scan iterates over.

    Args:
        parents  : (2*(N-1),) int — parent indices.
        children : (2*(N-1),) int — child indices.
        n_taxa   : int — number of leaf nodes.
        max_nodes: int — total nodes (2N-1).

    Returns:
        levels: List of lists, each containing (parent, child) edge tuples
                at the same depth level.
        root: Index of the root node.
    """
    import numpy as np
    parents_np  = np.array(parents,  dtype=int)
    children_np = np.array(children, dtype=int)

    # Build: parent → list of children
    parent_to_children: dict[int, list[int]] = {}
    child_to_parent: dict[int, int] = {}
    for p, c in zip(parents_np, children_np):
        parent_to_children.setdefault(p, []).append(c)
        child_to_parent[c] = p

    # Root = node with no parent
    all_nodes = set(range(max_nodes))
    non_roots = set(child_to_parent.keys())
    internal_nodes = set(parents_np.tolist())
    root_candidates = internal_nodes - non_roots
    root = max(root_candidates) if root_candidates else max_nodes - 1

    # BFS from root to assign depths
    depths: dict[int, int] = {root: 0}
    queue = [root]
    max_depth = 0
    while queue:
        node = queue.pop(0)
        for ch in parent_to_children.get(node, []):
            d = depths[node] + 1
            depths[ch] = d
            max_depth = max(max_depth, d)
            queue.append(ch)

    # Group edges by the depth of the CHILD (process leaves first)
    # Level 0 = deepest (tips or near-tips); we'll reverse later
    depth_to_edges: dict[int, list[tuple[int, int]]] = {}
    for p, c in zip(parents_np, children_np):
        d = depths.get(c, 0)
        depth_to_edges.setdefault(d, []).append((p, c))

    # Sort from deepest to shallowest (post-order)
    levels = [depth_to_edges[d] for d in sorted(depth_to_edges.keys(), reverse=True)]
    return levels, root


# ---------------------------------------------------------------------------
# Main Felsenstein log-likelihood
# ---------------------------------------------------------------------------

def felsenstein_log_likelihood(
    alignment_oh: jax.Array,
    parents: jax.Array,
    children: jax.Array,
    branch_lengths: jax.Array,
    n_taxa: int,
) -> jax.Array:
    """
    Compute the Felsenstein log-likelihood log P(Y | tree) for all sites.

    Uses JAX-compiled post-order tree traversal (tips to root).
    Numerical stability via log-sum-exp at every internal node.

    Args:
        alignment_oh  : (N_taxa, seq_len, 4) one-hot encoded alignment.
        parents       : (2*(N-1),) int parent node indices.
        children      : (2*(N-1),) int child node indices.
        branch_lengths: (2*(N-1),) float branch lengths.
        n_taxa        : int — number of leaf nodes N.

    Returns:
        log_lik: scalar — total log-likelihood summed over all sites.
    """
    seq_len  = alignment_oh.shape[1]
    max_nodes = 2 * n_taxa - 1

    # ------------------------------------------------------------------
    # Initialise log partial likelihoods
    # Tip nodes: L_v(a) = 1{a == y_v}  →  log L_v(a) = 0 if a==y_v else -inf
    # ------------------------------------------------------------------
    # alignment_oh: (N, seq_len, 4), values in {0, 1}
    # log_partials[node, site, state]
    tip_log = jnp.where(
        alignment_oh > 0.5,
        0.0,
        -jnp.inf,
    )  # (N, seq_len, 4)

    # Pad to max_nodes by appending zeros for internal node slots
    pad = jnp.zeros((max_nodes - n_taxa, seq_len, N_STATES))
    log_partials = jnp.concatenate([tip_log, pad], axis=0)  # (max_nodes, seq_len, 4)

    # ------------------------------------------------------------------
    # Post-order traversal: scan over depth levels bottom-up
    # ------------------------------------------------------------------
    # Build level schedule at Python-trace time (not inside JAX scan)
    parents_np  = jax.device_get(parents)
    children_np = jax.device_get(children)
    levels, root = _build_traversal_schedule(
        parents_np, children_np, n_taxa, max_nodes
    )

    # Process each level: for each (parent, child) edge in the level,
    # accumulate the child's contribution into the parent's log_partial.
    for level_edges in levels:
        for (par, chd) in level_edges:
            # Get the branch length for this edge
            # Find the edge index where (parents[e], children[e]) == (par, chd)
            edge_idx = _find_edge_idx(parents_np, children_np, par, chd)
            bl = branch_lengths[edge_idx]

            P  = jc69_transition_matrix(bl)                           # (4, 4)
            lp_child = log_partials[chd]                              # (seq_len, 4)
            contrib  = _log_conditional_child(lp_child, P)            # (seq_len, 4)

            # Accumulate: log(a × b) = log(a) + log(b)
            # For root/internal node: start from 0 (multiply first child contribution)
            # We check if internal node has been initialised by seeing if it's all 0s
            old = log_partials[par]
            new = old + contrib
            log_partials = log_partials.at[par].set(new)

    # ------------------------------------------------------------------
    # Root: sum over states with uniform stationary distribution (JC69: π=1/4)
    # ------------------------------------------------------------------
    log_pi = jnp.log(0.25)                             # scalar
    log_root = log_partials[root]                       # (seq_len, 4)
    log_site_liks = jax.nn.logsumexp(
        log_root + log_pi, axis=-1
    )  # (seq_len,)

    return jnp.sum(log_site_liks)  # scalar


def _find_edge_idx(
    parents: object,
    children: object,
    par: int,
    chd: int,
) -> int:
    """Find the edge index for a (parent, child) pair. Python-time only."""
    import numpy as np
    p = np.asarray(parents)
    c = np.asarray(children)
    idxs = np.where((p == par) & (c == chd))[0]
    if len(idxs) == 0:
        raise ValueError(f"Edge ({par}, {chd}) not found in tree.")
    return int(idxs[0])


# ---------------------------------------------------------------------------
# Combined: Z → distances → tree → log-likelihood
# ---------------------------------------------------------------------------

def vine_log_likelihood(
    Z: jax.Array,
    alignment_oh: jax.Array,
    tau: float,
    embed_dim: int | None = None,
) -> jax.Array:
    """
    End-to-end differentiable log-likelihood: embeddings → tree → likelihood.

    This is the function passed into the NumPyro model as `log_likelihood_fn`.
    It chains:
      1. pairwise_euclidean(Z) → D
      2. soft_upgma(D, tau)    → parents, children, branches
      3. felsenstein_log_likelihood(alignment_oh, ...)

    Args:
        Z            : (N, D) taxon embeddings.
        alignment_oh : (N, seq_len, 4) one-hot alignment.
        tau          : Temperature for Soft-UPGMA.
        embed_dim    : Unused (for API compatibility).

    Returns:
        Scalar log-likelihood.
    """
    # Support both package import and direct script execution
    try:
        from .embeddings import pairwise_euclidean
        from .decoder    import soft_upgma
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from embeddings import pairwise_euclidean
        from decoder    import soft_upgma

    n_taxa = Z.shape[0]
    D = pairwise_euclidean(Z)                           # (N, N)
    parents, children, branches, _ = soft_upgma(D, tau) # decode tree
    return felsenstein_log_likelihood(
        alignment_oh, parents, children, branches, n_taxa
    )
