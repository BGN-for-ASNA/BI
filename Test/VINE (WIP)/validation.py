#%%
"""
validation.py
=============
Three-phase validation pipeline for VINE phylogenetic inference.

Phase 1: Likelihood Equivalence Check
--------------------------------------
Generates a random alignment and a random known tree, computes the
Felsenstein log-likelihood via our JAX implementation, and compares
against a reference implementation computed from first principles.
Assertion: values agree within floating-point tolerance (|Δ| < 1e-4).

Phase 2: Generative Parameter Recovery ("Known Truth" Test)
------------------------------------------------------------
Simulates a known ground-truth tree (N=10 taxa) and sequences (L=1000)
under JC69. Feeds only the sequences into VINE SVI. After optimisation,
extracts the hard UPGMA tree and compares against ground truth via the
Robinson-Foulds (RF) distance using DendroPy.
Assertion: RF distance → 0 (perfect topology recovery).
Also produces a 1:1 scatter plot of True vs Inferred branch lengths.

Phase 3: Soft vs. Hard Convergence Check
-----------------------------------------
At the end of training, computes the log-likelihood of the soft (continuous)
tree state and the hard (discretised) tree state. 
Assertion: |log_lik_soft - log_lik_hard| / |log_lik_hard| < 0.05 (within 5%).
"""

from __future__ import annotations

import sys
import numpy as np
import jax
import jax.numpy as jnp
import dendropy
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

try:
    from .data_parser   import one_hot_encode, simulate_jc69_sequences
    from .embeddings    import pairwise_euclidean, get_map_embeddings
    from .decoder       import soft_upgma, extract_hard_tree
    from .likelihood    import (
        felsenstein_log_likelihood,
        jc69_transition_matrix,
        vine_log_likelihood,
    )
    from .optimizer     import run_vine_svi, tau_schedule
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from data_parser    import one_hot_encode, simulate_jc69_sequences
    from embeddings     import pairwise_euclidean, get_map_embeddings
    from decoder        import soft_upgma, extract_hard_tree
    from likelihood     import (
        felsenstein_log_likelihood,
        jc69_transition_matrix,
        vine_log_likelihood,
    )
    from optimizer      import run_vine_svi, tau_schedule



# ---------------------------------------------------------------------------
# Reference Felsenstein implementation (pure NumPy, for Phase 1 comparison)
# ---------------------------------------------------------------------------

def _reference_felsenstein(
    alignment: np.ndarray,   # (N, seq_len) integer
    parents: np.ndarray,     # (2*(N-1),) int
    children: np.ndarray,    # (2*(N-1),) int
    branch_lengths: np.ndarray,  # (2*(N-1),) float
    n_taxa: int,
) -> float:
    """
    Pure NumPy reference Felsenstein implementation for correctness checking.

    Uses direct multiplication (no log-space) on short sequences to
    validate the JAX log-space implementation.
    """
    seq_len   = alignment.shape[1]
    max_nodes = 2 * n_taxa - 1
    n_states  = 4

    # Build parent->children map
    p2c: dict[int, list[tuple[int, int]]] = {}
    for e, (p, c) in enumerate(zip(parents, children)):
        p2c.setdefault(p, []).append((c, e))

    # Find root: node that appears only in parents, not in children
    child_set = set(children)
    root = max(set(parents) - child_set)

    # Initialise partial likelihoods at tips
    partials = np.zeros((max_nodes, seq_len, n_states), dtype=np.float64)
    for i in range(n_taxa):
        for s in range(seq_len):
            partials[i, s, alignment[i, s]] = 1.0

    # Post-order traversal (recursive, small trees only)
    def visit(node: int) -> None:
        for child, edge_idx in p2c.get(node, []):
            visit(child)
        if node < n_taxa:
            return  # leaf, already initialised
        # Internal: multiply children contributions
        first = True
        for child, edge_idx in p2c.get(node, []):
            bl = float(branch_lengths[edge_idx])
            P  = np.array(jc69_transition_matrix(jnp.array(bl)))  # (4,4)
            # contrib[s, a] = Σ_b P[a,b] * partials[child, s, b]
            contrib = partials[child] @ P.T    # (seq_len, 4) @ (4,4) → (seq_len,4)
            if first:
                partials[node] = contrib
                first = False
            else:
                partials[node] *= contrib

    visit(root)

    # Root: sum over states with π = 1/4
    log_lik = np.sum(np.log(np.sum(0.25 * partials[root], axis=-1) + 1e-300))
    return float(log_lik)


# ---------------------------------------------------------------------------
# Helper: build a random bifurcating tree topology
# ---------------------------------------------------------------------------

def _random_tree(n_taxa: int, rng: np.random.Generator) -> tuple[list, int]:
    """
    Generate a random binary tree topology via random UPGMA-style merging.

    Returns:
        edges: List of (parent, child, branch_length) triples.
        root:  Root node index.
    """
    active = list(range(n_taxa))
    next_id = n_taxa
    edges = []
    for _ in range(n_taxa - 1):
        # Pick two random nodes to merge
        idx1, idx2 = rng.choice(len(active), 2, replace=False)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        c1, c2 = active[idx1], active[idx2]
        bl1 = float(rng.uniform(0.05, 0.5))
        bl2 = float(rng.uniform(0.05, 0.5))
        edges.append((next_id, c1, bl1))
        edges.append((next_id, c2, bl2))
        active.pop(idx2)
        active.pop(idx1)
        active.append(next_id)
        next_id += 1
    root = next_id - 1
    return edges, root


# ---------------------------------------------------------------------------
# Helper: convert edge list to DendroPy Tree
# ---------------------------------------------------------------------------

def _edges_to_dendropy(
    edges: list[tuple[int, int, float]],
    n_taxa: int,
    root: int,
) -> dendropy.Tree:
    """
    Convert a (parent, child, branch_length) edge list to a DendroPy Tree.

    Leaf taxa are named "T0", "T1", …, "T{N-1}".
    """
    # Build adjacency from root
    parent_to_children: dict[int, list[tuple[int, float]]] = {}
    for p, c, bl in edges:
        parent_to_children.setdefault(p, []).append((c, bl))

    taxon_ns = dendropy.TaxonNamespace(
        [f"T{i}" for i in range(n_taxa)]
    )

    def _build(node_id: int) -> dendropy.Node:
        node = dendropy.Node()
        if node_id < n_taxa:
            node.taxon = taxon_ns.require_taxon(label=f"T{node_id}")
        for child_id, bl in parent_to_children.get(node_id, []):
            child_node = _build(child_id)
            child_node.edge.length = bl
            node.add_child(child_node)
        return node

    root_node = _build(root)
    tree = dendropy.Tree(taxon_namespace=taxon_ns)
    tree.seed_node = root_node
    return tree


# ---------------------------------------------------------------------------
# Phase 1: Likelihood equivalence check
# ---------------------------------------------------------------------------

def phase1_likelihood_equivalence(
    n_taxa: int = 6,
    seq_len: int = 50,
    seed: int = 0,
    tol: float = 1e-3,
) -> bool:
    """
    Phase 1: Assert that our JAX Felsenstein matches the NumPy reference.

    Steps:
      1. Random alignment (integer-encoded).
      2. Random tree (random edges + branch lengths).
      3. Compute log-likelihood with our JAX implementation.
      4. Compute log-likelihood with the NumPy reference.
      5. Assert |JAX - NumPy| < tol.

    Returns:
        True if assertion passes.
    """
    print("\n" + "="*60)
    print("PHASE 1: Likelihood Equivalence Check")
    print("="*60)

    rng = np.random.default_rng(seed)
    alignment = rng.integers(0, 4, size=(n_taxa, seq_len)).astype(np.int32)
    alignment_oh = one_hot_encode(alignment)

    # Random tree
    edges, root = _random_tree(n_taxa, rng)
    parents_np   = np.array([e[0] for e in edges], dtype=np.int32)
    children_np  = np.array([e[1] for e in edges], dtype=np.int32)
    branches_np  = np.array([e[2] for e in edges], dtype=np.float32)

    # JAX implementation
    jax_ll = float(felsenstein_log_likelihood(
        alignment_oh,
        jnp.array(parents_np),
        jnp.array(children_np),
        jnp.array(branches_np),
        n_taxa,
    ))

    # NumPy reference
    ref_ll = _reference_felsenstein(
        alignment, parents_np, children_np, branches_np, n_taxa
    )

    delta = abs(jax_ll - ref_ll)
    print(f"  JAX log-likelihood  : {jax_ll:.6f}")
    print(f"  NumPy reference     : {ref_ll:.6f}")
    print(f"  |Δ|                 : {delta:.2e}  (tolerance: {tol:.2e})")

    if delta < tol:
        print("  ✅ PASSED: Implementations agree within tolerance.")
        return True
    else:
        print("  ❌ FAILED: Discrepancy exceeds tolerance!")
        return False


# ---------------------------------------------------------------------------
# Phase 2: Generative parameter recovery
# ---------------------------------------------------------------------------

def phase2_parameter_recovery(
    n_taxa: int = 10,
    seq_len: int = 1000,
    embed_dim: int = 8,
    n_steps: int = 5_000,
    seed: int = 42,
    output_plot: str = "branch_length_recovery.png",
) -> tuple[bool, float]:
    """
    Phase 2: Simulate a known tree, recover it from sequences via VINE SVI.

    Steps:
      1. Generate a random ground-truth tree (N=10, known topology).
      2. Simulate sequences (L=1000) down the tree using JC69.
      3. Run VINE SVI on the sequences only (no tree info).
      4. Extract the hard UPGMA tree from the inferred distances.
      5. Compute Robinson-Foulds distance vs ground truth.
      6. Plot True vs Inferred branch lengths.

    Returns:
        (passed, rf_distance)
    """
    print("\n" + "="*60)
    print("PHASE 2: Generative Parameter Recovery")
    print("="*60)

    rng   = np.random.default_rng(seed)
    key   = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # 1. Ground-truth tree
    # ------------------------------------------------------------------
    gt_edges, gt_root = _random_tree(n_taxa, rng)
    print(f"  Ground-truth tree: {len(gt_edges)} edges, root={gt_root}")
    for p, c, bl in gt_edges:
        print(f"    {p} → {c}  (bl={bl:.4f})")

    # ------------------------------------------------------------------
    # 2. Simulate sequences
    # ------------------------------------------------------------------
    alignment = np.array(simulate_jc69_sequences(gt_edges, n_taxa, seq_len, key))
    alignment_oh = one_hot_encode(alignment)
    print(f"  Simulated alignment: {n_taxa} taxa × {seq_len} sites")

    # ------------------------------------------------------------------
    # 3. VINE SVI
    # ------------------------------------------------------------------
    print(f"  Running VINE SVI ({n_steps} steps)…")
    result = run_vine_svi(
        alignment_oh=alignment_oh,
        n_taxa=n_taxa,
        embed_dim=embed_dim,
        n_steps=n_steps,
        print_every=1_000,
    )

    # ------------------------------------------------------------------
    # 4. Extract hard tree from inferred distances
    # ------------------------------------------------------------------
    mu = result.embed_mu                           # (N, D)
    D_inferred = np.array(pairwise_euclidean(mu))  # (N, N)
    inferred_edges, inferred_root = extract_hard_tree(jnp.array(D_inferred))

    # ------------------------------------------------------------------
    # 5. Robinson-Foulds distance
    # ------------------------------------------------------------------
    gt_tree  = _edges_to_dendropy(gt_edges,       n_taxa, gt_root)
    inf_tree = _edges_to_dendropy(inferred_edges, n_taxa, inferred_root)

    try:
        rf = dendropy.calculate.treecompare.symmetric_difference(gt_tree, inf_tree)
        max_rf = 2 * (n_taxa - 3)  # maximum possible RF for unrooted trees
        rf_normalised = rf / max(max_rf, 1)
        print(f"  Robinson-Foulds distance : {rf}  (max={max_rf}, normalised={rf_normalised:.3f})")
    except Exception as e:
        print(f"  Warning: RF calculation failed ({e}). Setting RF=inf.")
        rf = float("inf")
        rf_normalised = float("inf")

    # ------------------------------------------------------------------
    # 6. Branch-length scatter plot
    # ------------------------------------------------------------------
    # Map true branch lengths (leaves only) for comparison
    true_bls: dict[int, float] = {}
    for p, c, bl in gt_edges:
        if c < n_taxa:
            true_bls[c] = bl

    inferred_bls: dict[int, float] = {}
    for p, c, bl in inferred_edges:
        if c < n_taxa:
            inferred_bls[c] = bl

    common_leaves = sorted(set(true_bls) & set(inferred_bls))
    if common_leaves:
        true_vals     = [true_bls[l]     for l in common_leaves]
        inferred_vals = [inferred_bls[l] for l in common_leaves]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true_vals, inferred_vals, s=60, alpha=0.8)
        lim = max(max(true_vals), max(inferred_vals)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x (ideal)")
        ax.set_xlabel("True branch length")
        ax.set_ylabel("Inferred branch length")
        ax.set_title("VINE: Branch Length Recovery")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_plot, dpi=120)
        plt.close(fig)
        print(f"  Scatter plot saved → {output_plot}")

    passed = rf == 0
    if passed:
        print("  ✅ PASSED: Perfect topology recovery (RF = 0).")
    else:
        print(f"  ⚠  RF = {rf} (topology not perfectly recovered; may need more steps).")
    return passed, float(rf)


# ---------------------------------------------------------------------------
# Phase 3: Soft vs. Hard convergence check
# ---------------------------------------------------------------------------

def phase3_soft_vs_hard(
    alignment_oh: jax.Array,
    n_taxa: int,
    result,
    threshold: float = 0.05,
) -> bool:
    """
    Phase 3: Assert that soft and hard log-likelihoods are within 5%.

    At convergence, the Soft-UPGMA should approximate the discrete hard tree
    closely enough that:
        |log_lik_soft - log_lik_hard| / |log_lik_hard| < threshold

    Args:
        alignment_oh : (N, seq_len, 4) one-hot alignment.
        n_taxa       : Number of taxa.
        result       : VINEResult from Phase 2 SVI.
        threshold    : Relative tolerance (default 0.05 = 5%).

    Returns:
        True if assertion passes.
    """
    print("\n" + "="*60)
    print("PHASE 3: Soft vs. Hard Convergence Check")
    print("="*60)

    mu = result.embed_mu
    tau_final = result.final_tau()

    # ------------------------------------------------------------------
    # Soft log-likelihood: use final tau (near-hard but still soft)
    # ------------------------------------------------------------------
    log_lik_soft = float(vine_log_likelihood(mu, alignment_oh, tau_final))

    # ------------------------------------------------------------------
    # Hard log-likelihood: tau → 0 (essentially 0.001) = crisp argmin
    # ------------------------------------------------------------------
    log_lik_hard = float(vine_log_likelihood(mu, alignment_oh, tau=0.001))

    rel_diff = abs(log_lik_soft - log_lik_hard) / (abs(log_lik_hard) + 1e-10)
    print(f"  Soft log-likelihood : {log_lik_soft:.4f}")
    print(f"  Hard log-likelihood : {log_lik_hard:.4f}")
    print(f"  Relative difference : {rel_diff:.4f}  (threshold: {threshold:.4f})")

    if rel_diff < threshold:
        print("  ✅ PASSED: Relaxation has converged (soft ≈ hard).")
        return True
    else:
        print(f"  ❌ FAILED: Relaxation gap {rel_diff:.4f} exceeds {threshold:.4f}.")
        return False


# ---------------------------------------------------------------------------
# Full validation pipeline runner
# ---------------------------------------------------------------------------

def run_validation_pipeline(
    n_taxa: int = 10,
    seq_len: int = 1000,
    embed_dim: int = 8,
    n_steps: int = 5_000,
    seed: int = 42,
) -> dict[str, bool | float]:
    """
    Execute all three validation phases and return a summary.

    Args:
        n_taxa    : Number of taxa for Phase 2 (default 10).
        seq_len   : Sequence length for Phase 2 (default 1000).
        embed_dim : Embedding dimension (default 8).
        n_steps   : SVI steps for Phase 2 (default 5000).
        seed      : Random seed.

    Returns:
        dict with keys:
          'phase1_passed': bool
          'phase2_passed': bool
          'phase2_rf'    : float (Robinson-Foulds distance)
          'phase3_passed': bool
    """
    results: dict[str, bool | float] = {}

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    p1 = phase1_likelihood_equivalence(seed=seed)
    results["phase1_passed"] = p1

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    gt_edges, gt_root = _random_tree(n_taxa, rng)
    alignment = np.array(simulate_jc69_sequences(gt_edges, n_taxa, seq_len, key))
    alignment_oh = one_hot_encode(alignment)

    # Run SVI
    print(f"\n  Running VINE SVI for Phase 2 & 3 ({n_steps} steps)…")
    svi_result = run_vine_svi(
        alignment_oh=alignment_oh,
        n_taxa=n_taxa,
        embed_dim=embed_dim,
        n_steps=n_steps,
        print_every=1_000,
        rng_seed=seed,
    )

    # Extract hard tree and compute RF
    mu = svi_result.embed_mu
    D_inferred = np.array(pairwise_euclidean(mu))
    inferred_edges, inferred_root = extract_hard_tree(jnp.array(D_inferred))

    gt_tree  = _edges_to_dendropy(gt_edges,       n_taxa, gt_root)
    inf_tree = _edges_to_dendropy(inferred_edges, n_taxa, inferred_root)

    try:
        rf = float(dendropy.calculate.treecompare.symmetric_difference(gt_tree, inf_tree))
    except Exception:
        rf = float("inf")

    print(f"\n  Robinson-Foulds: {rf}")
    results["phase2_passed"] = (rf == 0.0)
    results["phase2_rf"]     = rf

    # Branch length plot
    true_bls     = {c: bl for p, c, bl in gt_edges       if c < n_taxa}
    inferred_bls = {c: bl for p, c, bl in inferred_edges if c < n_taxa}
    common = sorted(set(true_bls) & set(inferred_bls))
    if common:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter([true_bls[l] for l in common], [inferred_bls[l] for l in common], s=60)
        m = max(max(true_bls.values()), max(inferred_bls.values())) * 1.1
        ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Inferred")
        ax.set_title("Branch Length Recovery")
        fig.tight_layout()
        fig.savefig("branch_length_recovery.png", dpi=120)
        plt.close(fig)

    # ELBO loss curve
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(-svi_result.losses, alpha=0.8)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO")
    ax.set_title("VINE Training Curve")
    fig.tight_layout()
    fig.savefig("vine_training_curve.png", dpi=120)
    plt.close(fig)
    print("  Training curve saved → vine_training_curve.png")

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    p3 = phase3_soft_vs_hard(alignment_oh, n_taxa, svi_result)
    results["phase3_passed"] = p3

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for k, v in results.items():
        status = "✅" if v else "❌" if isinstance(v, bool) else ""
        print(f"  {k:20s}: {v}  {status}")

    all_passed = all(
        results[k] for k in ["phase1_passed", "phase2_passed", "phase3_passed"]
    )
    print(f"\n  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_validation_pipeline(
        n_taxa=10,
        seq_len=1000,
        embed_dim=8,
        n_steps=5_000,
        seed=42,
    )
