"""Exact posterior sampling of discrete latent sites, conditioned on data.

numpyro.infer.Predictive(model, posterior_samples, infer_discrete=True)
unconditionally wraps the model with `numpyro.handlers.mask(model, mask=False)`
before enumerating (see numpyro/infer/util.py:_predictive). `mask(fn, mask=False)`
zeroes `log_prob` for *every* sample site, including observed ones — so the
funsor factor graph used to draw the discrete site loses the likelihood term
and collapses to a uniform draw over the site's support, independent of the
data. This module reimplements the same forward-filter/backward-sample logic
without that masking step, so discrete sites are drawn from their true
posterior, conditioned on the observed data.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from numpyro.contrib.funsor import config_enumerate
from numpyro.contrib.funsor.discrete import _sample_posterior
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import _guess_max_plate_nesting
from numpyro.util import soft_vmap


def sample_discrete_posterior(model, posterior_samples, rng_key, model_kwargs=None):
    """Draw exact posterior samples of a model's discrete sites.

    Args:
        model: the (unmasked) model function.
        posterior_samples: dict of posterior draws for the model's
            continuous/parameter sites, each array shaped (S, ...).
        rng_key: JAX PRNGKey.
        model_kwargs: keyword arguments to pass to the model (must include
            the observed data so the discrete sites are conditioned on it).

    Returns:
        dict of sampled sites (discrete + everything else in the trace),
        each shaped (S, ...).
    """
    model_kwargs = model_kwargs or {}
    batch_shape = jax.tree_util.tree_leaves(posterior_samples)[0].shape[:1]

    rng_key, subkey = random.split(rng_key)
    prototype_sample = jax.tree.map(lambda x: x[0], posterior_samples)
    prototype_trace = trace(
        seed(substitute(model, prototype_sample), subkey)
    ).get_trace(**model_kwargs)
    first_available_dim = -_guess_max_plate_nesting(prototype_trace) - 1

    def single_prediction(val):
        key, samples = val
        substituted_model = substitute(model, samples)
        return _sample_posterior(
            config_enumerate(substituted_model),
            first_available_dim,
            1,  # temperature=1 -> sample (not MAP)
            key,
            **model_kwargs,
        )

    num_samples = int(np.prod(batch_shape))
    key_shape = rng_key.shape
    if num_samples > 1:
        rng_key = random.split(rng_key, num_samples)
    rng_key = rng_key.reshape(batch_shape + key_shape)

    return soft_vmap(
        single_prediction, (rng_key, posterior_samples), len(batch_shape), num_samples
    )
