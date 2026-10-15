"""Random-effect building blocks (m.effects)."""
import jax.numpy as jnp
import numpy as np

from BayesForge import bf


def test_phylo_effect_covariance_and_tree_stack():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    cov = jnp.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
    L = jnp.linalg.cholesky(cov)
    draws = np.stack([np.asarray(m.effects.phylo_effect(L, 2.0, sample=True, name="z"))
                      for _ in range(4000)])
    assert draws.shape == (4000, 3)
    np.testing.assert_allclose(np.cov(draws.T), 4.0 * np.asarray(cov), atol=0.35)

    # a stack of T trees with one sd per tree -> one effect vector per tree
    Ls = jnp.stack([L, jnp.eye(3)])
    u = m.effects.phylo_effect(Ls, jnp.array([1.0, 0.5]), sample=True, name="z")
    assert u.shape == (2, 3)
