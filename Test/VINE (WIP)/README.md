# VINE — Variational Inference with Node Embeddings

Scalable, fully differentiable, GPU-accelerated Bayesian phylogenetic inference.

## Architecture

```
bi_phylogenetics/
├── pyproject.toml
├── README.md
└── src/
    ├── __init__.py
    ├── data_parser.py    # FASTA → one-hot JAX arrays
    ├── embeddings.py     # Variational posterior q_φ(Z) = N(μ, diag(σ²))
    ├── decoder.py        # Soft-UPGMA + Straight-Through Estimator
    ├── likelihood.py     # Felsenstein pruning (JC69, log-space, JAX-compiled)
    ├── optimizer.py      # NumPyro SVI + Optax Adam + τ annealing
    └── validation.py     # 3-phase correctness pipeline
```

## Mathematical Pipeline

```
DNA sequences Y
      │
      ▼ (one-hot encode)
alignment_oh  (N, L, 4)
      │
      ▼ (VINE: learn via SVI)
Z ~ q_φ(Z) = N(μ, diag(σ²))    ← optimise φ = {μ, log σ}
      │
      ▼ pairwise_euclidean(Z)
D_ij = ‖Z_i − Z_j‖₂            (N × N distance matrix)
      │
      ▼ soft_upgma(D, τ)  ← STE: exact topology forward, smooth gradients backward
(parents, children, branch_lengths)
      │
      ▼ felsenstein_log_likelihood(alignment_oh, tree)
log P(Y | tree)                  (JC69 model, post-order log-space traversal)
      │
      ▼ ELBO = E_q[log P(Y|tree)] − KL(q‖p)
      optimise via Adam
```

## Quick Start

```bash
# Install (requires CUDA 12 for JAX GPU support)
poetry install

# Run the validation pipeline
python -m src.validation
```

## Key Design Decisions

### Straight-Through Estimator (STE)
`argmin` is non-differentiable. We use the STE trick (Bengio et al., 2013):
```
z_ste = z_soft + stop_gradient(z_hard − z_soft)
```
- **Forward**: `z_ste == z_hard` → exact discrete topology
- **Backward**: `∂z_ste/∂θ == ∂z_soft/∂θ` → smooth gradients

### Temperature Annealing
The softmax temperature τ decays exponentially:
```
τ(step) = τ_init · (τ_final / τ_init)^(step / n_steps)
```
from τ=5.0 (smooth) to τ=0.1 (near-discrete) over training.

### No Python for-loops in likelihood
Tree traversal uses a level-schedule computed at trace time; within each
level, all nodes are processed in parallel via JAX primitives. The scan
over merge steps uses `jax.lax.scan` for O(N) compile time.

### Numerical Stability
All partial likelihoods are kept in log-space. `jax.nn.logsumexp` is used
at every internal node to prevent underflow over long sequences.

## Validation Results

| Phase | Test | Expected |
|-------|------|---------|
| 1 | JAX vs NumPy Felsenstein | \|Δ\| < 1e-3 |
| 2 | Robinson-Foulds distance | RF → 0 |
| 3 | Soft vs Hard log-lik gap | < 5% |

## Citation

```bibtex
@article{vine2024,
  title  = {VINE: Variational Inference with Node Embeddings for Phylogenetic Inference},
  note   = {Conceptual framework implemented in this package}
}
```
