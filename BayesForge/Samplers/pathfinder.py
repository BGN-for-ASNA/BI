"""Multi-path Pathfinder (Zhang et al. 2022, JMLR 23:306) on a NumPyro model.

Each path runs L-BFGS from a random point and fits a Gaussian approximation
along the optimisation path (BlackJAX ``blackjax.vi.pathfinder``). The draws
of all paths are pooled and resampled with Pareto-smoothed importance
weights (log p - log q), which corrects part of the approximation error and
is the multi-path variant of the paper. The Pareto k of these weights tells
how far the approximation is from the posterior (k > 0.7: unreliable).

Requires the optional dependency ``blackjax`` (pip install blackjax, or
pip install "bayesforge[pathfinder]").
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import initialize_model


def _blackjax():
    try:
        import blackjax
    except ImportError as e:
        raise ImportError(
            "Pathfinder needs the optional dependency blackjax: "
            "pip install blackjax (or pip install 'bayesforge[pathfinder]')."
        ) from e
    return blackjax


class Pathfinder_BF:
    """Sampler-like object (same interface as SVI_BF) for multi-path Pathfinder.

    Args:
        model: NumPyro model.
        num_paths: Number of independent single-path runs (default 4).
        num_samples: Number of draws returned after importance resampling.
        num_draws_per_path: Draws taken from each path's approximation.
        maxiter: Maximum L-BFGS iterations per path.
        init_radius: Initial points are uniform in (-r, r) on the
            unconstrained scale (as NumPyro's init_to_uniform).
        **lbfgs_kwargs: Passed to ``blackjax.vi.pathfinder.approximate``.
    """

    def __init__(self, model, num_paths=4, num_samples=1000, num_draws_per_path=1000,
                 maxiter=100, init_radius=2.0, **lbfgs_kwargs):
        self.model = model
        self.num_paths = int(num_paths)
        self.num_samples = int(num_samples)
        self.num_draws_per_path = int(num_draws_per_path)
        self.maxiter = int(maxiter)
        self.init_radius = float(init_radius)
        self.lbfgs_kwargs = lbfgs_kwargs
        # interface shared with the MCMC / SVI wrappers
        self.num_chains = 1
        self.num_warmup = 0
        self.thinning = 1
        self.sampler = self
        self._args, self._kwargs = (), {}
        self.samples = None
        self.unconstrained_samples = None
        self.pareto_k = None
        self.path_elbo = None

    def run(self, rng_key, *args, **kwargs):
        blackjax = _blackjax()
        self._args, self._kwargs = args, kwargs
        k_init, k_start, k_paths, k_res = jax.random.split(rng_key, 4)

        param_info, potential_fn, postprocess_fn, _ = initialize_model(
            k_init, self.model, model_args=args, model_kwargs=kwargs, dynamic_args=False)
        z0 = param_info.z
        flat0, unravel = ravel_pytree(z0)

        def logdensity(flat):
            return -potential_fn(unravel(flat))

        starts = jax.random.uniform(k_start, (self.num_paths, flat0.size),
                                    minval=-self.init_radius, maxval=self.init_radius)
        draws, logq, elbo = [], [], []
        for k, start in zip(jax.random.split(k_paths, self.num_paths), starts):
            k_fit, k_draw = jax.random.split(k)
            state, _ = blackjax.vi.pathfinder.approximate(
                k_fit, logdensity, start, maxiter=self.maxiter, **self.lbfgs_kwargs)
            d, lq = blackjax.vi.pathfinder.sample(k_draw, state, self.num_draws_per_path)
            draws.append(d)
            logq.append(lq)
            elbo.append(float(state.elbo))
        draws = jnp.concatenate(draws)                      # (paths * draws, dim)
        logq = jnp.concatenate(logq)
        logp = jax.vmap(logdensity)(draws)
        logw = np.array(logp - logq, dtype=np.float64)   # writable copy
        logw[~np.isfinite(logw)] = -np.inf
        if not np.isfinite(logw).any():
            raise RuntimeError("Pathfinder: every draw has a non-finite density.")

        from arviz_stats.base import array_stats
        finite = np.isfinite(logw)
        lw = np.full_like(logw, -np.inf)
        # psislw takes log-likelihood-like input (it smooths exp(-input), as in
        # PSIS-LOO), so pass the negated log importance weights.
        lw[finite], k = array_stats.psislw(-logw[finite])
        self.pareto_k = float(k)
        w = np.exp(lw - lw.max())
        idx = jax.random.choice(k_res, draws.shape[0], (self.num_samples,),
                                replace=True, p=jnp.asarray(w / w.sum()))
        flat_samples = draws[idx]
        self.unconstrained_samples = jax.vmap(unravel)(flat_samples)
        self.samples = jax.vmap(postprocess_fn)(self.unconstrained_samples)
        self.path_elbo = elbo
        if self.pareto_k > 0.7:
            warnings.warn(
                f"Pathfinder: Pareto k = {self.pareto_k:.2f} > 0.7, the approximation "
                "is far from the posterior; use it as an initialisation for "
                "m.fit(..., init_strategy='pathfinder') rather than as a result.",
                stacklevel=3)
        return self

    def get_samples(self, group_by_chain=False, **kwargs):
        if group_by_chain:
            return {k: v[None, ...] for k, v in self.samples.items()}
        return dict(self.samples)

    def get_extra_fields(self, group_by_chain=False, **kwargs):
        return {}

    def init_params(self, num_chains):
        """One distinct unconstrained draw per chain, for MCMC.run(init_params=)."""
        z = self.unconstrained_samples
        n = next(iter(z.values())).shape[0]
        if n < num_chains:
            raise ValueError(f"Pathfinder returned {n} draws for {num_chains} chains.")
        idx = jnp.linspace(0, n - 1, num_chains).astype(int)
        return {k: (v[idx] if num_chains > 1 else v[idx[0]]) for k, v in z.items()}
