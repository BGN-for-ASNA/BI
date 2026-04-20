"""
data_parser.py
==============
DNA sequence parsing and one-hot encoding utilities.

Handles FASTA input via BioPython and converts nucleotide sequences into
JAX arrays suitable for the Felsenstein pruning algorithm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from Bio import SeqIO

# Nucleotide alphabet: A=0, C=1, G=2, T=3
NUCLEOTIDES = {"A": 0, "C": 1, "G": 2, "T": 3}
N_STATES = 4  # |{A, C, G, T}|


def parse_fasta(filepath: str) -> tuple[list[str], np.ndarray]:
    """
    Parse a FASTA file into taxon names and an integer-encoded alignment matrix.

    Args:
        filepath: Path to the FASTA file.

    Returns:
        taxa_names: List of sequence identifiers.
        alignment:  Integer array of shape (N_taxa, seq_len), values in {0,1,2,3}.
    """
    records = list(SeqIO.parse(filepath, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {filepath}")

    taxa_names = [r.id for r in records]
    seq_len = len(records[0].seq)

    # Encode sequences as integers; unknown bases default to 0 (A)
    alignment = np.zeros((len(records), seq_len), dtype=np.int32)
    for i, rec in enumerate(records):
        for j, base in enumerate(str(rec.seq).upper()):
            alignment[i, j] = NUCLEOTIDES.get(base, 0)

    return taxa_names, alignment


def one_hot_encode(alignment: np.ndarray) -> jax.Array:
    """
    Convert integer-encoded alignment to one-hot arrays for Felsenstein pruning.

    The Felsenstein algorithm initialises the partial likelihood at each tip
    as a one-hot vector over {A, C, G, T}.

    Args:
        alignment: Integer array of shape (N_taxa, seq_len).

    Returns:
        JAX array of shape (N_taxa, seq_len, N_states=4), dtype float32.
    """
    n_taxa, seq_len = alignment.shape
    oh = np.zeros((n_taxa, seq_len, N_STATES), dtype=np.float32)
    for i in range(n_taxa):
        oh[i, np.arange(seq_len), alignment[i]] = 1.0
    return jnp.array(oh)


def simulate_jc69_sequences(
    tree_topology: list[tuple[int, int, float]],
    n_taxa: int,
    seq_len: int,
    rng_key: jax.Array,
) -> jax.Array:
    """
    Simulate DNA sequences down a tree under the JC69 model.

    This is used in validation to generate ground-truth alignments with a
    known tree so we can check parameter recovery.

    Args:
        tree_topology: List of (parent_idx, child_idx, branch_length) triples.
                       Nodes 0..n_taxa-1 are tips; node n_taxa is the root.
        n_taxa:        Number of leaf taxa.
        seq_len:       Number of sites to simulate.
        rng_key:       JAX PRNG key.

    Returns:
        Integer alignment array of shape (n_taxa, seq_len).
    """
    # Total nodes = 2*n_taxa - 1 for a bifurcating tree
    n_nodes = 2 * n_taxa - 1
    # node_seqs[node, site] -> nucleotide index {0,1,2,3}
    node_seqs = np.zeros((n_nodes, seq_len), dtype=np.int32)

    # Root node index is n_nodes - 1
    root = n_nodes - 1
    key, subkey = jax.random.split(rng_key)
    # Draw root sequence from uniform stationary distribution
    node_seqs[root] = np.array(
        jax.random.randint(subkey, (seq_len,), 0, N_STATES), dtype=np.int32
    )

    # BFS from root through topology
    # Build adjacency: parent -> [children with lengths]
    children: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for parent, child, bl in tree_topology:
        children[parent].append((child, bl))

    queue = [root]
    while queue:
        node = queue.pop(0)
        for child, bl in children[node]:
            key, subkey = jax.random.split(key)
            parent_seq = node_seqs[node]
            child_seq = np.zeros(seq_len, dtype=np.int32)
            # JC69 transition probability
            p_same = 0.25 + 0.75 * np.exp(-4.0 / 3.0 * bl)
            p_diff = (1.0 - p_same) / 3.0
            uniforms = np.array(jax.random.uniform(subkey, (seq_len,)))
            for site in range(seq_len):
                base = parent_seq[site]
                u = uniforms[site]
                if u < p_same:
                    child_seq[site] = base
                elif u < p_same + p_diff:
                    child_seq[site] = (base + 1) % 4
                elif u < p_same + 2 * p_diff:
                    child_seq[site] = (base + 2) % 4
                else:
                    child_seq[site] = (base + 3) % 4
            node_seqs[child] = child_seq
            queue.append(child)

    # Return only leaf sequences (indices 0..n_taxa-1)
    return jnp.array(node_seqs[:n_taxa], dtype=jnp.int32)
