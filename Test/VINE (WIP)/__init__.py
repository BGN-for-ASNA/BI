"""
bi_phylogenetics.src
====================
VINE: Variational Inference with Node Embeddings for Bayesian Phylogenetic Inference.

A scalable, fully differentiable, GPU-accelerated phylogenetic inference pipeline.

Modules
-------
data_parser   : DNA sequence parsing and one-hot encoding.
embeddings    : Variational posterior q_φ(Z) = N(μ, diag(σ²)) over taxon embeddings.
decoder       : Differentiable Soft-UPGMA decoder via Straight-Through Estimator.
likelihood    : JAX-compiled Felsenstein pruning algorithm (JC69 substitution model).
optimizer     : NumPyro SVI training loop with temperature annealing.
validation    : Three-phase correctness validation pipeline.
"""

from .data_parser  import parse_fasta, one_hot_encode, simulate_jc69_sequences
from .embeddings   import (
    pairwise_euclidean,
    vine_model,
    vine_guide,
    get_map_embeddings,
    get_posterior_distances,
)
from .decoder      import soft_upgma, extract_hard_tree
from .likelihood   import (
    jc69_transition_matrix,
    felsenstein_log_likelihood,
    vine_log_likelihood,
)
from .optimizer    import run_vine_svi, tau_schedule, VINEResult
from .validation   import run_validation_pipeline

__all__ = [
    # data_parser
    "parse_fasta",
    "one_hot_encode",
    "simulate_jc69_sequences",
    # embeddings
    "pairwise_euclidean",
    "vine_model",
    "vine_guide",
    "get_map_embeddings",
    "get_posterior_distances",
    # decoder
    "soft_upgma",
    "extract_hard_tree",
    # likelihood
    "jc69_transition_matrix",
    "felsenstein_log_likelihood",
    "vine_log_likelihood",
    # optimizer
    "run_vine_svi",
    "tau_schedule",
    "VINEResult",
    # validation
    "run_validation_pipeline",
]
