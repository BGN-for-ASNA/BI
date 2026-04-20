"""
optimizer.py
============
NumPyro SVI training loop for VINE with temperature annealing.

ELBO Objective
--------------
We maximise the Evidence Lower Bound (ELBO):

    L(φ) = E_{q_φ(Z)} [log P(Y | Tree(Z))] - KL(q_φ(Z) || p(Z))

where:
  - q_φ(Z) = N(μ, diag(σ²))  is the variational posterior over embeddings.
  - p(Z)   = N(0, I)          is the standard Gaussian prior.
  - log P(Y | Tree(Z)) is the Felsenstein log-likelihood after decoding Z
    through Soft-UPGMA.
  - The KL term encourages the posterior to stay close to the prior and is
    computed analytically for Gaussian q and p.

Temperature Annealing Schedule
-------------------------------
The Gumbel-Softmax temperature τ controls how "soft" the tree decoding is:
  - High τ (early training): near-uniform merge weights → smooth loss surface,
    easier for gradients to flow.
  - Low τ (late training): near-hard argmin → precise discrete topology.

We use exponential decay:
    τ(step) = τ_init * (τ_final / τ_init)^(step / n_steps)
            = τ_init * exp(-step * ln(τ_init / τ_final) / n_steps)

This anneals from τ_init=5.0 to τ_final=0.1 over 10,000 steps.

Implementation
--------------
NumPyro SVI with Trace_ELBO handles the ELBO estimation automatically,
using the reparameterisation trick (pathwise gradient estimator) for the
Gaussian variational family.

The temperature τ is passed as a keyword argument to the model/guide at each
step via a custom SVI run loop (rather than using SVI.run(), which doesn't
support step-varying hyperparameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

try:
    from .embeddings import vine_model, vine_guide
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from embeddings import vine_model, vine_guide


# ---------------------------------------------------------------------------
# Temperature annealing
# ---------------------------------------------------------------------------

def tau_schedule(
    step: int,
    n_steps: int,
    tau_init: float = 5.0,
    tau_final: float = 0.1,
) -> float:
    """
    Exponential decay schedule for the Gumbel-Softmax temperature.

    τ(step) = τ_init · (τ_final / τ_init)^(step / n_steps)

    Args:
        step    : Current optimisation step (0-indexed).
        n_steps : Total number of steps.
        tau_init: Starting temperature (default 5.0).
        tau_final: Final temperature (default 0.1).

    Returns:
        tau: Float temperature for the current step.
    """
    progress = min(step / max(n_steps - 1, 1), 1.0)
    tau = tau_init * (tau_final / tau_init) ** progress
    return float(tau)


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class VINEResult:
    """Container for VINE optimisation results.

    Attributes
    ----------
    params        : dict mapping param name → final value (e.g. 'embed_mu').
    losses        : (n_steps,) array of ELBO loss values (= -ELBO).
    taus          : (n_steps,) temperature values used each step.
    n_taxa        : Number of taxa N.
    embed_dim     : Embedding dimension D.
    n_steps       : Number of training steps.
    """
    params:    dict
    losses:    np.ndarray
    taus:      np.ndarray
    n_taxa:    int
    embed_dim: int
    n_steps:   int

    @property
    def embed_mu(self) -> jax.Array:
        """Return the posterior mean embeddings μ of shape (N, D)."""
        return self.params["embed_mu"]

    @property
    def embed_sigma(self) -> jax.Array:
        """Return the posterior std devs σ of shape (N, D)."""
        return jax.nn.softplus(self.params["embed_log_sig"]) + 1e-5

    def final_tau(self) -> float:
        """Return the temperature used at the last training step."""
        return float(self.taus[-1])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_vine_svi(
    alignment_oh: jax.Array,
    n_taxa: int,
    embed_dim: int = 8,
    n_steps: int = 10_000,
    lr: float = 3e-3,
    tau_init: float = 5.0,
    tau_final: float = 0.1,
    log_likelihood_fn: Callable | None = None,
    rng_seed: int = 42,
    print_every: int = 500,
) -> VINEResult:
    """
    Run VINE variational inference via NumPyro SVI.

    Optimises the ELBO:
        L(φ) = E_{q_φ}[log P(Y | Tree(Z))] - KL(q_φ(Z) || p(Z))

    using Adam with step-varying temperature τ for the Soft-UPGMA decoder.

    Args:
        alignment_oh     : (N, seq_len, 4) one-hot alignment (JAX array).
        n_taxa           : Number of taxa N.
        embed_dim        : Latent dimension D (default 8).
        n_steps          : Total SVI steps (default 10,000).
        lr               : Adam learning rate (default 3e-3).
        tau_init         : Initial temperature (default 5.0).
        tau_final        : Final temperature (default 0.1).
        log_likelihood_fn: Custom log-likelihood function. If None, uses the
                           default vine_log_likelihood from likelihood.py.
        rng_seed         : JAX random seed for reproducibility.
        print_every      : Print ELBO every N steps (0 = silent).

    Returns:
        VINEResult with final parameters, loss history, and tau history.
    """
    if log_likelihood_fn is None:
        try:
            from .likelihood import vine_log_likelihood
        except ImportError:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from likelihood import vine_log_likelihood
        log_likelihood_fn = vine_log_likelihood

    rng_key = jax.random.PRNGKey(rng_seed)

    # ------------------------------------------------------------------
    # Set up NumPyro SVI
    # ------------------------------------------------------------------
    optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))

    svi = SVI(
        model=vine_model,
        guide=vine_guide,
        optim=optimizer,
        loss=Trace_ELBO(num_particles=1),
    )

    # ------------------------------------------------------------------
    # Initialise SVI state with tau = tau_init
    # ------------------------------------------------------------------
    rng_key, init_key = jax.random.split(rng_key)
    svi_state = svi.init(
        init_key,
        n_taxa=n_taxa,
        embed_dim=embed_dim,
        alignment_oh=alignment_oh,
        log_likelihood_fn=log_likelihood_fn,
        tau=tau_init,
    )

    # ------------------------------------------------------------------
    # Training loop with temperature annealing
    # ------------------------------------------------------------------
    losses = np.zeros(n_steps, dtype=np.float32)
    taus   = np.zeros(n_steps, dtype=np.float32)

    for step in range(n_steps):
        # Compute current temperature
        tau = tau_schedule(step, n_steps, tau_init, tau_final)
        taus[step] = tau

        # SVI update step
        rng_key, step_key = jax.random.split(rng_key)
        svi_state, loss = svi.update(
            svi_state,
            n_taxa=n_taxa,
            embed_dim=embed_dim,
            alignment_oh=alignment_oh,
            log_likelihood_fn=log_likelihood_fn,
            tau=tau,
        )
        losses[step] = float(loss)

        if print_every > 0 and (step % print_every == 0 or step == n_steps - 1):
            print(
                f"Step {step:>6d}/{n_steps}  |  "
                f"ELBO = {-loss:>10.3f}  |  "
                f"τ = {tau:.4f}"
            )

    # ------------------------------------------------------------------
    # Extract final parameters
    # ------------------------------------------------------------------
    params = svi.get_params(svi_state)

    return VINEResult(
        params=params,
        losses=losses,
        taus=taus,
        n_taxa=n_taxa,
        embed_dim=embed_dim,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Convenience: compute ELBO components at a specific parameter state
# ---------------------------------------------------------------------------

def compute_elbo_components(
    result: VINEResult,
    alignment_oh: jax.Array,
    log_likelihood_fn: Callable | None = None,
    rng_seed: int = 0,
) -> dict[str, float]:
    """
    Decompose the ELBO into its two components at the final parameters.

    ELBO = E_{q}[log P(Y | tree)] - KL(q || p)

    The KL for a diagonal Gaussian q = N(μ, σ²) vs p = N(0, I) is:
        KL = 0.5 * Σ_i (σ_i² + μ_i² - 1 - log σ_i²)

    The expected log-likelihood is estimated via a single Monte Carlo sample.

    Args:
        result         : Trained VINEResult.
        alignment_oh   : (N, seq_len, 4) one-hot alignment.
        log_likelihood_fn: Log-likelihood function (defaults to vine_log_likelihood).
        rng_seed       : Random seed for MC sample.

    Returns:
        dict with keys: 'elbo', 'expected_log_lik', 'kl_divergence'.
    """
    if log_likelihood_fn is None:
        try:
            from .likelihood import vine_log_likelihood
        except ImportError:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from likelihood import vine_log_likelihood
        log_likelihood_fn = vine_log_likelihood

    mu    = result.embed_mu
    sigma = result.embed_sigma

    # KL divergence (analytic)
    kl = 0.5 * jnp.sum(sigma ** 2 + mu ** 2 - 1.0 - jnp.log(sigma ** 2 + 1e-10))

    # MC estimate of expected log-likelihood
    key = jax.random.PRNGKey(rng_seed)
    eps = jax.random.normal(key, mu.shape)
    Z_sample = mu + sigma * eps

    tau_final = result.final_tau()
    log_lik = log_likelihood_fn(Z_sample, alignment_oh, tau_final)

    elbo = float(log_lik) - float(kl)

    return {
        "elbo":               elbo,
        "expected_log_lik":   float(log_lik),
        "kl_divergence":      float(kl),
    }
