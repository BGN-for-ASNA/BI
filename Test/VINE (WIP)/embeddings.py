"""
embeddings.py
=============
Variational Node Embeddings for VINE (Variational Inference with Node Embeddings).

Each taxon i is mapped to a continuous vector Z_i ∈ R^D in a D-dimensional
Euclidean latent space. The variational posterior is:

    q_φ(Z) = N(μ, diag(σ²))

where φ = {μ, log σ} are the parameters we optimise via SVI.

The pairwise Euclidean distance matrix D_ij = ||Z_i - Z_j||_2 is then fed
into the differentiable Soft-UPGMA decoder.

References
----------
VINE: "Phylogenetic Inference via Variational Inference with Node Embeddings"
(conceptual framework — see decoder.py for the full citation context).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-8  # Numerical stability floor for distances


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def pairwise_euclidean(Z: jax.Array) -> jax.Array:
    """
    Compute the N×N pairwise squared Euclidean distance matrix from embeddings Z.

    Uses the identity ||a - b||² = ||a||² + ||b||² - 2<a,b> for efficiency,
    then takes the square root to obtain Euclidean (not squared) distances.

    Args:
        Z: Array of shape (N, D) — N taxon embeddings in R^D.

    Returns:
        D_mat: Array of shape (N, N), D_mat[i,j] = ||Z_i - Z_j||_2 ≥ 0.
               The diagonal is exactly 0.
    """
    # sq_norms[i] = ||Z_i||²
    sq_norms = jnp.sum(Z ** 2, axis=-1)          # (N,)
    # cross[i,j] = <Z_i, Z_j>
    cross = Z @ Z.T                                # (N, N)
    # squared distances: broadcast sq_norms as row + column
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * cross
    # Clamp negatives (numerical noise on the diagonal)
    sq_dist = jnp.maximum(sq_dist, 0.0)
    return jnp.sqrt(sq_dist + _EPS) - jnp.sqrt(_EPS)  # ≈ ||·||_2, exact 0 on diagonal


# ---------------------------------------------------------------------------
# NumPyro model & guide
# ---------------------------------------------------------------------------

def vine_model(
    n_taxa: int,
    embed_dim: int,
    alignment_oh: jax.Array,
    log_likelihood_fn,
    tau: float = 1.0,
) -> None:
    """
    NumPyro generative model for VINE.

    Declares:
      - Prior:  Z ~ N(0, I_{N×D})
      - Likelihood: alignment_oh | Z via Felsenstein pruning on Soft-UPGMA tree.

    Args:
        n_taxa:            Number of leaf taxa N.
        embed_dim:         Dimension D of the embedding space.
        alignment_oh:      One-hot alignment of shape (N, seq_len, 4).
        log_likelihood_fn: Callable (Z, alignment_oh, tau) -> scalar log-likelihood.
        tau:               Current Gumbel-Softmax temperature (annealed externally).
    """
    # -----------------------------------------------------------------------
    # Prior: isotropic Gaussian in R^{N×D}
    # -----------------------------------------------------------------------
    Z = numpyro.sample(
        "Z",
        dist.Normal(
            jnp.zeros((n_taxa, embed_dim)),
            jnp.ones((n_taxa, embed_dim)),
        ).to_event(2),
    )

    # -----------------------------------------------------------------------
    # Likelihood: Felsenstein pruning conditioned on the decoded tree
    # -----------------------------------------------------------------------
    log_lik = log_likelihood_fn(Z, alignment_oh, tau)
    numpyro.factor("obs", log_lik)


def vine_guide(
    n_taxa: int,
    embed_dim: int,
    alignment_oh: jax.Array,  # unused but must match model signature
    log_likelihood_fn,         # unused but must match model signature
    tau: float = 1.0,          # unused but must match model signature
) -> None:
    """
    NumPyro mean-field variational guide for VINE.

    Variational posterior:
        q_φ(Z) = N(μ, diag(σ²))

    Learnable parameters registered via numpyro.param:
      - embed_mu:      (N, D) — variational means μ.
      - embed_log_sig: (N, D) — log standard deviations; σ = softplus(log_sig).

    Args:
        n_taxa:     Number of leaf taxa N.
        embed_dim:  Dimension D of the embedding space.
        (remaining args are ignored; present to match model signature)
    """
    # -----------------------------------------------------------------------
    # Variational parameters φ = {μ, log σ}
    # -----------------------------------------------------------------------
    mu = numpyro.param(
        "embed_mu",
        init_value=jnp.zeros((n_taxa, embed_dim)),
    )
    # Initialise log-scale to 0 → σ = softplus(0) ≈ 0.693
    log_sig = numpyro.param(
        "embed_log_sig",
        init_value=jnp.zeros((n_taxa, embed_dim)),
    )
    sigma = jax.nn.softplus(log_sig) + 1e-5  # strictly positive

    # -----------------------------------------------------------------------
    # Sample Z from the variational posterior
    # -----------------------------------------------------------------------
    numpyro.sample(
        "Z",
        dist.Normal(mu, sigma).to_event(2),
    )


# ---------------------------------------------------------------------------
# Utility: extract MAP embeddings after optimisation
# ---------------------------------------------------------------------------

def get_map_embeddings(svi_result) -> jax.Array:
    """
    Extract the MAP (μ) embeddings from a trained SVI result.

    Args:
        svi_result: numpyro.infer.SVIRunResult returned by SVI.run().

    Returns:
        mu: Array of shape (N, D) — the posterior mean embeddings.
    """
    return svi_result.params["embed_mu"]


def get_posterior_distances(svi_result) -> jax.Array:
    """
    Compute the posterior-mean pairwise distance matrix.

    Args:
        svi_result: numpyro.infer.SVIRunResult returned by SVI.run().

    Returns:
        D_mat: Array of shape (N, N) — pairwise Euclidean distances from μ.
    """
    mu = get_map_embeddings(svi_result)
    return pairwise_euclidean(mu)
