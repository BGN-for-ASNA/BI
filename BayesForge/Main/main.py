import inspect
import ast

# import jax.random as random
# import jax.numpy as jnp
# from jax import vmap
# from jax import jit
import jax


from numpyro.infer import Predictive
from numpyro import handlers
import numpyro
try:
    import arviz as az
except Exception:
    # matplotlib>=3.11 removed matplotlib.style.core, which arviz_plots imports.
    # Shim it (the paths live on matplotlib.style directly) and retry; arviz is
    # only needed by summary(), so defer to None if it still fails.
    import sys as _sys, types as _t
    import matplotlib.style as _ms
    if not hasattr(_ms, "core"):
        _shim = _t.ModuleType("matplotlib.style.core")
        _shim.USER_LIBRARY_PATHS = getattr(_ms, "USER_LIBRARY_PATHS", [])
        _shim.STYLE_EXTENSION = getattr(_ms, "STYLE_EXTENSION", "mplstyle")
        _shim.reload_library = getattr(_ms, "reload_library", lambda: None)
        _ms.core = _shim
        _sys.modules["matplotlib.style.core"] = _shim
    try:
        import arviz as az
    except Exception:
        az = None

import functools
import cloudpickle

from BayesForge.SetDevice.set import setup_device, setup_multi_device

from BayesForge.Data.manip import manip
from BayesForge.Resources.datasets import load

from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.Effects import effects
from BayesForge.Utils.link import link
from BayesForge.Utils.infer_discrete import sample_discrete_posterior

from BayesForge.Diagnostic.Diag2 import diagWIP as diag
#from BayesForge.Diagnostic.Diag import diag as diag

from BayesForge.Network.Net import net, net2
from BayesForge.NBDA.NBDA import NBDA
from BayesForge.Causal.dag import causal, DAG

from BayesForge.Models.models import models
from BayesForge.Models.surv import survival
from BayesForge.Models.GMM import *
from BayesForge.Models.DPMM import *
from BayesForge.ML.ml import ml
from BayesForge.BNN.bnn import bnn


class bf(manip):
    def __init__(
        self,
        platform="cpu",
        cores=None,
        gpu_index=None,
        rand_seed=True,
        deallocate=False,
        print_devices_found=True,
        backend="numpyro",
        diag_backend="arviz",
        float_precision=64,
        loss=None,
        optim=None,
        guide=None,
    ):
        """
        Initialize the BF class with platform, cores, gpu_index, deallocate, print_devices_found, and backend parameters.

        Args:
            platform (str, optional): Platform to use. Defaults to 'cpu'.
            cores (int, optional): Number of cores. Defaults to None.
            gpu_index (int, optional): Index of the GPU to use. Defaults to None.
            float_precision (int or str, optional): JAX floating-point precision — 32 or 64 (also 'float32'/'float64'). Defaults to 64.
            deallocate (bool, optional): Whether to deallocate. Defaults to False.
            print_devices_found (bool, optional): Whether to print devices found. Defaults to True.
            backend (str, optional): Backend to use. Defaults to 'numpyro'.
            diag_backend (str, optional): Engine behind the convergence metrics
                exposed by ``m.summary()`` and ``m.diag.{summary,rhat,ess,mcse,
                loo,WAIC}``. ``'arviz'`` (default) uses ArviZ, the reference
                implementation. ``'jax'`` selects the experimental
                JAX/NumPy diagnostics and emits a warning. Plots, PPC,
                sensitivity and regression overlays are unaffected either way.
                Defaults to 'arviz'.
            shard (bool, optional): Enable EXPERIMENTAL data sharding (split the
                leading axis of data arrays across devices). Requires more than
                one device (cores>1 on CPU or multiple GPUs). Off by default;
                when enabled a warning describes the model classes for which it
                is not yet correct. Equivalent to setting BF_SHARD=1. Defaults to
                False.
        """
        manip.__init__(self)
        setup_device(
            platform=platform,
            cores=cores,
            gpu_index=gpu_index,
            deallocate=deallocate,
            print_devices_found=print_devices_found,
            float_precision=float_precision,
        )

        # Multi-device setup ----------------------------------------------------
        # Works on both CPU (virtual devices via XLA_FLAGS) and GPU (physical).
        # Provides m.n_devices, m.n_gpus, m.shard(), m.replicate().
        import jax
        self.n_gpus    = len(jax.devices('gpu')) if platform == 'gpu' else 0
        self.n_devices = len(jax.devices(platform))
        self._data_sharding = None
        self._rep_sharding  = None
        self.mesh = None
        self._last_shard_report = None
        if self.n_devices > 1:
            self.mesh, self._data_sharding, self._rep_sharding = \
                setup_multi_device(platform)
        # -----------------------------------------------------------------------

        # Sharding is declared per-fit via fit(shard=True); it is NOT a
        # construction-time concern (the device mesh above is built whenever
        # cores>1 regardless). BF_SHARD=1 only sets the *default* for
        # fit(shard=None). The EXPERIMENTAL-limits warning is printed lazily on
        # the first sharded fit (see _maybe_warn_shard_limits).
        import os as _os
        self._auto_shard = _os.environ.get('BF_SHARD', '0') == '1'
        self._shard_warned = False

        self.seed = rand_seed
        from BayesForge.Utils.rng import RNG

        self._rng = RNG(rand_seed)
        self._rng.seed_globals()
        self.data_on_model = None
        self.priors_name = None
        self.tab_summary = None
        self.posteriors_full = None
        self.posteriors_by_chain_full = None

        self.nbdaModel = False
        self.obs_args = None
        self.model2 = None
        self.trace = None
        self.history = {}
        self.backend = backend
        # Engine behind the convergence metrics. ArviZ is the default and the
        # reference; "jax" selects the experimental jax_diagnostics path.
        from BayesForge.Diagnostic.patch_diag import _resolve_backend
        self.diag_backend = _resolve_backend(diag_backend)
        self.loss = loss
        self.optim = optim
        self.guide = guide

        self.gaussian = gaussian
        self.survival = survival(self)
        self.effects = effects
        self.link = link

        self.dpmm = dpmm
        self.gmm = gmm
        self.NBDA = NBDA

        self.models = models(self)

        self.model_name = None
        self.run_model_name = None

        self.net = net()
        self.net2 = net2()
        self.ml = ml()
        self.bnn = bnn()
        self.causal = causal()
        self.dag = DAG

        self.load = load()

        if backend == "numpyro":
            from BayesForge.Distributions.np_dists import UnifiedDist as np_dists, DistProxy

            self.dist = DistProxy(np_dists(rand_seed=self.seed, rng=self._rng))

            # effects/model_effects/gaussian/DPMM/GMM/surv each hold their own
            # module-level `dist` singleton, instantiated with UnifiedDist's
            # default (rand_seed=True, rng=None). Left alone that: (1) ignores
            # whatever rand_seed was requested here (always time.time_ns() on
            # sample=True), and (2) even once seeded, _draw_key falls back to
            # a literal fixed PRNGKey(seed) reused on every call rather than
            # _rng.stream()'s fresh independent sub-key -- so e.g. lambda_i and
            # delta_ij from separate m.effects.varying_effects(sample=True)
            # calls would be correlated (same key), not independent draws.
            # Share this instance's RNG manager with those singletons so the
            # whole DGP -- not just m.dist -- is both reproducible and
            # properly decorrelated.
            import BayesForge.Utils.Effects as _effects_mod
            import BayesForge.Utils.Gaussian as _gaussian_mod
            import BayesForge.Network.model_effects as _model_effects_mod
            import BayesForge.Models.DPMM as _dpmm_mod
            import BayesForge.Models.GMM as _gmm_mod
            import BayesForge.Models.surv as _surv_mod

            for _mod in (
                _effects_mod,
                _gaussian_mod,
                _model_effects_mod,
                _dpmm_mod,
                _gmm_mod,
                _surv_mod,
            ):
                if hasattr(_mod, "dist"):
                    _mod.dist.seed = self.seed
                    _mod.dist._rng = self._rng

        elif backend == "tfp":
            from BayesForge.Distributions.tfp_dists import UnifiedDist as tfp_dists
            from BayesForge.Samplers.mcmc_tfp import mcmc as mcmc_tfp

            self.dist = tfp_dists
            self.sampler = mcmc_tfp()
            jax.config.update("jax_enable_x64", False)

    def _resolve_key(self, seed, label):
        """Resolve a JAX PRNGKey for an inference call.

        If ``seed`` is given explicitly it is honoured verbatim; otherwise the
        central RNG manager supplies a stable, label-derived key governed by
        ``rand_seed`` (reproducible for False/int, entropy-based for True).
        """
        if seed is None:
            return self._rng.key(label)
        return jax.random.PRNGKey(int(seed))

    def get_seed(self):
        """Return the resolved base seed governing reproducibility."""
        return self._rng.get_seed()

    def set_seed(self, seed):
        """Reset the reproducibility base seed for subsequent draws."""
        self._rng.set_seed(seed)
        self.seed = seed

    def latex(self):
        from BayesForge.PostModel.to_latex import to_latex

        return to_latex(self.model)

    def graph(self, model=None, figsize=(8, 6), title=None, draw=True):
        """Build (and draw) the directed graph of a probabilistic model.

        Parses the model's source to reconstruct latent, observed,
        deterministic and data nodes and the dependencies between them.

        Args:
            model: Model function to graph. Defaults to the current model.
            figsize: Figure size passed to the renderer.
            title: Optional plot title.
            draw: If True (default) render with matplotlib; otherwise just
                return the :class:`ModelGraph` object.

        Returns:
            The :class:`ModelGraph` instance (inspect ``.nodes``/``.edges``).
        """
        from BayesForge.PostModel.model_graph import model_to_graph

        if model is None:
            model = self.model
        g = model_to_graph(model)
        if draw:
            g.draw(figsize=figsize, title=title)
        return g

    def fit(
        self,
        model=None,
        obs=None,
        potential_fn=None,
        kinetic_fn=None,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=None,
        max_tree_depth=10,
        init_strategy=numpyro.infer.init_to_uniform,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        thinning=1,
        postprocess_fn=None,
        chain_method="parallel",
        shard=None,
        progress_bar=True,
        jit_model_args=False,
        seed=None,
        extra_fields=None,
    ):
        """
        Fit the model using the specified backend and parameters.

        Args:
            model: Model to fit.
            obs: Observed data.
            potential_fn: Potential function.
            kinetic_fn: Kinetic function.
            step_size: Step size for the sampler.
            inverse_mass_matrix: Inverse mass matrix.
            adapt_step_size: Whether to adapt step size.
            adapt_mass_matrix: Whether to adapt mass matrix.
            dense_mass: Whether to use dense mass.
            target_accept_prob: Target acceptance probability.
            trajectory_length: Length of the trajectory.
            max_tree_depth: Maximum tree depth.
            init_strategy: Initialization strategy.
            find_heuristic_step_size: Whether to find heuristic step size.
            forward_mode_differentiation: Whether to use forward mode differentiation.
            regularize_mass_matrix: Whether to regularize mass matrix.
            num_warmup: Number of warmup samples.
            num_samples: Number of samples.
            num_chains: Number of chains.
            thinning: Thinning factor.
            postprocess_fn: Postprocess function.
            chain_method: Chain method.
            progress_bar: Whether to show progress bar.
            jit_model_args: Whether to JIT model arguments.
            seed: Random seed.
        """
        self.num_chains = num_chains

        # Sharding is declared here, per fit. fit(shard=None) (the default)
        # falls back to the BF_SHARD env default; fit(shard=True/False) is
        # explicit. There is no construction-time shard flag.
        if shard is None:
            shard = self._auto_shard

        if shard:
            if self._data_sharding is None:
                import warnings
                warnings.warn(
                    "shard=True has no effect: no multi-device mesh available "
                    f"(n_devices={self.n_devices}). "
                    "Pass cores>1 when creating bf() to enable sharding.",
                    stacklevel=2,
                )
                shard = False
            else:
                self._maybe_warn_shard_limits()   # print limits once, lazily
                chain_method = "vectorized"

        if obs is None:
            obs = self.data_on_model
        else:
            self.data_on_model = obs

        if model is None:
            if self.nbdaModel == False:
                print("Argument model can't be None")

            else:
                self.model = self.nbda.model
                self.model_name = "NBDA"
        else:
            self.model = model
            if isinstance(model, functools.partial):
                self.model_name = model.func.__name__
            else:
                if self.model_name is None:
                    self.model_name = model.__name__

        if self.nbdaModel == False:
            if self.data_on_model is None:
                if self.backend == "numpyro":
                    self.data_on_model = self.pd_to_jax(self.model)
                else:
                    self.data_on_model = self.pd_to_jax(self.model, bit="32")

        if self.model_name == "gmm":
            if "initial_means" not in self.data_on_model:
                self.ml.KMEANS(
                    self.data_on_model["data"], n_clusters=self.data_on_model["K"]
                )
                self.data_on_model["initial_means"] = self.ml.results["centroids"]

        if self.model_name == "pca":
            self.model = model.get_model(model.type)
            tmp = self.data_on_model.keys()

            if "X" not in tmp:
                return print("X is required")
            else:
                self.data_on_model["X"] = self.data_on_model["X"].T
            if "data_dim" not in tmp:
                self.data_on_model["data_dim"] = self.data_on_model["X"].shape[0]
            if "latent_dim" not in tmp:
                self.data_on_model["latent_dim"] = self.data_on_model["X"].shape[0]
            if "num_data_points" not in tmp:
                self.data_on_model["num_data_points"] = self.data_on_model["X"].shape[1]
            # Init pca class with data
            self.models.pca = self.models.pca(
                X=self.data_on_model["X"],
                latent_dim=self.data_on_model["latent_dim"],
                type=model.type,
            )

        if self.backend == "numpyro":
            from BayesForge.Samplers.mcmc_numpyro import mcmc_numpyro

            self.sampler = mcmc_numpyro(
                model=self.model,
                potential_fn=potential_fn,
                kinetic_fn=kinetic_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                adapt_step_size=adapt_step_size,
                adapt_mass_matrix=adapt_mass_matrix,
                dense_mass=dense_mass,
                target_accept_prob=target_accept_prob,
                trajectory_length=trajectory_length,
                max_tree_depth=max_tree_depth,
                init_strategy=init_strategy,
                find_heuristic_step_size=find_heuristic_step_size,
                forward_mode_differentiation=forward_mode_differentiation,
                regularize_mass_matrix=regularize_mass_matrix,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                thinning=thinning,
                postprocess_fn=postprocess_fn,
                chain_method=chain_method,
                progress_bar=progress_bar,
                jit_model_args=jit_model_args,
            )

            # Improvement 2 — gate data sharding on the explicit *shard* intent
            # only, NEVER on chain_method. Choosing chain_method='vectorized' for
            # its own reasons must not silently trigger data parallelism.
            # Improvement 6 — do not mutate self.data_on_model: keep the canonical
            # (host) copy intact so subsequent fits stay independent. The sharded
            # arrays live only in the local `run_data` passed to this run.
            if shard and self._data_sharding is not None:
                run_data = self._auto_shard_data(self.data_on_model)
                # Safeguards (static HLO + runtime value) run ONCE here, before
                # the MCMC loop — they add a one-time setup cost, no per-step cost.
                # A ✗ runtime verdict (sharding changes the answer) falls back to
                # replicated; ⚠ verdicts (collectives → deadlock/cost) only warn.
                if self._shard_check(self.model, run_data, self.data_on_model):
                    pass  # static-only warnings; keep sharding
                else:
                    run_data = self.data_on_model  # correctness fallback
            else:
                run_data = self.data_on_model
            # These were already being requested from NUTS but never read back,
            # so leapfrog counts and energy were collected and thrown away.
            # step_size is added so the adapted value can be reported too.
            if extra_fields is None:
                extra_fields = ('num_steps', 'energy', 'diverging',
                                'accept_prob', 'adapt_state.step_size')
            self._extra_fields_requested = tuple(extra_fields)
            self._max_tree_depth = max_tree_depth
            self.sampler.run(
                self._resolve_key(seed, "fit"),
                extra_fields=self._extra_fields_requested,
                **run_data,
            )
            self.extra_fields = self.sampler.get_extra_fields()
            self.extra_fields_by_chain = self.sampler.get_extra_fields(
                group_by_chain=True)
            self.posteriors = self.sampler.get_samples()
            self.posteriors_by_chain = self.sampler.get_samples(group_by_chain=True)
            self.posteriors_full = dict(self.posteriors)
            self.posteriors_by_chain_full = dict(self.posteriors_by_chain)
            self.diag = diag(sampler=self.sampler)
            from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
            bind_diag_to_model(self.diag, self, backend=self.diag_backend)
            self.get_history()

        elif self.backend == "tfp":
            print(
                "[WARNING] This function is still in development. Use it with caution. [WARNING]"
            )
            from BayesForge.Distributions.tfp_dists import UnifiedDist as tfp_dists
            from BayesForge.Samplers.mcmc_tfp import mcmc as mcmc_tfp
            from BayesForge.Samplers.Model_handler import model_handler

            # self.mcmc= mcmc_tfp()
            self.sampler.data_on_model = self.data_on_model
            self.sampler.model = self.model

            # Pass the fit parameters to the TFP sampler
            self.sampler.run(
                seed=self._resolve_key(seed, "fit"),
                model=self.model,
                n_chains=num_chains,
                num_results=num_samples,
                num_burnin_steps=num_warmup,
                **self.data_on_model,
            )
            self.diag = diag(sampler=self.sampler)
            self.posteriors = self.sampler.get_samples()
            self.posteriors_by_chain = self.posteriors  # TFP always returns chain-structured
            self.posteriors_full = dict(self.posteriors)
            self.posteriors_by_chain_full = dict(self.posteriors_by_chain)
            from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
            bind_diag_to_model(self.diag, self, backend=self.diag_backend)
            self.get_history()

        if self.model_name == "pca":
            self.models.pca.posterior = self.posteriors
            self.models.pca.get_attributes(self.models.pca.X.T)

    def svi(
        self,
        model=None,
        guide=None,
        optim=None,
        loss=None,
        num_steps=1000,
        num_samples=1000,
        seed=None,
        **kwargs,
    ):
        """
        Fit the model using SVI (Stochastic Variational Inference).
        """
        if model is None:
            model = self.model
        else:
            self.model = model
            if isinstance(model, functools.partial):
                self.model_name = model.func.__name__
            else:
                if self.model_name is None:
                    self.model_name = model.__name__

        if self.data_on_model is None:
            self.data_on_model = self.pd_to_jax(self.model)

        if optim is None:
            optim = self.optim
            if optim is None:
                import numpyro

                optim = numpyro.optim.Adam(step_size=1e-3)

        if loss is None:
            loss = self.loss
            if loss is None:
                from numpyro.infer import Trace_ELBO

                loss = Trace_ELBO(num_particles=10)

        if guide is None:
            guide = self.guide
            if guide is None:
                from numpyro.infer.autoguide import AutoDiagonalNormal

                guide = AutoDiagonalNormal(self.model)
        elif isinstance(guide, str):
            from numpyro.infer import autoguide

            guides = {
                "diagonal": autoguide.AutoDiagonalNormal,
                "multivariate": autoguide.AutoMultivariateNormal,
                "laplace": autoguide.AutoLaplaceApproximation,
                "delta": autoguide.AutoDelta,
                "normal": autoguide.AutoNormal,
                "low_rank": autoguide.AutoLowRankMultivariateNormal,
            }
            if guide.lower() in guides:
                guide = guides[guide.lower()](self.model)
            else:
                raise ValueError(
                    f"Unknown autoguide: {guide}. Available options: {list(guides.keys())}"
                )

        if self.backend == "numpyro":
            from BayesForge.Samplers.SVI.svi_numpyro import svi_numpyro

            self.sampler = svi_numpyro(
                model=self.model,
                guide=guide,
                optim=optim,
                loss=loss,
                num_steps=num_steps,
                num_samples=num_samples,
                **kwargs,
            )

            self.sampler.run(
                self._resolve_key(seed, "svi"),
                **self.data_on_model,
            )
            self.posteriors = self.sampler.get_samples(
                seed=(self.get_seed() if seed is None else int(seed)),
            )
            self.diag = diag(sampler=self.sampler)
            self.get_history()

        self.run_model_name = self.model_name
        self.model_name = None

    # Random number generator ----------------------------------------------------------------

    def randint(self, low, high, shape):
        """
        Generate random integers in the given range.

        Args:
            low: Lowest possible value.
            high: Highest possible value.
            shape: Shape of the output array.

        Returns:
            Array of random integers.
        """
        return jax.random.randint(self._rng.stream(), shape, low, high)

    # Sampler diagnostics -----------------------------------------------------------------------
    def sampler_stats(self, per_chain=False):
        """HMC sampler diagnostics for the last fit: work done and how well it went.

        These describe the *sampler*, not the posterior. `summary()` reports
        ESS and R-hat; this reports what the sampler had to do to get them,
        which is what explains a slow or inefficient run.

        Parameters
        ----------
        per_chain : bool
            Return one row per chain instead of pooled totals.

        Returns
        -------
        dict (or list of dict when per_chain=True) with

          leapfrog_total          summed gradient evaluations over sampling
          leapfrog_per_iter       mean leapfrog steps per iteration
          pct_at_max_treedepth    % of iterations that saturated max_tree_depth.
                                  High values mean the trajectory was truncated
                                  before it turned, so the cost per effective
                                  sample is being capped, not chosen.
          divergences             count of divergent transitions
          ebfmi                   E-BFMI, same definition Stan reports. Below
                                  0.3 indicates poor energy exploration, the
                                  usual signature of a funnel.
          accept_prob             mean Metropolis acceptance probability
          step_size               adapted step size at the end of warmup

        Requires a numpyro-backend fit; `extra_fields` must have included
        'num_steps' and 'energy' (both are on by default).
        """
        import numpy as _np

        ex = getattr(self, "extra_fields_by_chain", None)
        if not ex:
            raise RuntimeError(
                "No sampler diagnostics recorded. sampler_stats() needs a "
                "numpyro-backend fit; if extra_fields was overridden, include "
                "'num_steps' and 'energy'."
            )

        def _get(name):
            v = ex.get(name)
            return None if v is None else _np.asarray(v)

        steps = _get("num_steps")
        energy = _get("energy")
        div = _get("diverging")
        acc = _get("accept_prob")
        ss = _get("adapt_state.step_size")
        cap = 2 ** getattr(self, "_max_tree_depth", 10) - 1

        def _ebfmi(e):
            # mean successive squared energy difference over marginal variance
            var = _np.var(e)
            return float("nan") if var == 0 else float(
                _np.sum(_np.diff(e) ** 2) / (len(e) * var))

        def _row(sl):
            r = {}
            if steps is not None:
                s = steps[sl]
                r["leapfrog_total"] = int(s.sum())
                r["leapfrog_per_iter"] = float(s.mean())
                r["pct_at_max_treedepth"] = float(100.0 * (s >= cap).mean())
            if div is not None:
                r["divergences"] = int(_np.asarray(div[sl]).sum())
            if energy is not None:
                e = _np.asarray(energy[sl]).ravel()
                r["ebfmi"] = round(_ebfmi(e), 4)
            if acc is not None:
                r["accept_prob"] = round(float(_np.asarray(acc[sl]).mean()), 4)
            if ss is not None:
                r["step_size"] = float(_np.asarray(ss[sl]).ravel()[-1])
            return r

        if per_chain:
            return [dict(chain=i, **_row(i)) for i in range(len(steps))]

        out = _row(slice(None))
        if energy is not None:            # worst chain, as Stan warns on
            out["ebfmi"] = round(min(_ebfmi(_np.asarray(e).ravel())
                                     for e in energy), 4)
        return out

    # Get posteriors ----------------------------------------------------------------------------
    def summary(
        self,
        round_to=2,
        hdi_prob=0.89,
        include=None,
        exclude=None,
        var_names=None,
        exclude_vars=None,
        filter_regex=None,
        group_by_chain=True,
        backend=None,
        kind="all",
    ):
        """Posterior summary table.

        Args:
            round_to: Decimal places to round.
            hdi_prob: HDI probability mass.
            include: str or list — keep only these parameters.
            exclude: str or list — remove these parameters.
            var_names: Legacy alias for include.
            exclude_vars: Legacy alias for exclude.
            filter_regex: Regex applied to base parameter names (jax backend only).
            group_by_chain: Use chain-structured posteriors (default True).
            backend: "arviz" (default) or "jax". Overrides the model's
                ``diag_backend`` for this call; "jax" warns that the
                JAX/NumPy diagnostics are experimental.
            kind: Passed to az.summary ("all", "stats", "diagnostics") when the
                arviz backend is in use.
        """
        from BayesForge.Diagnostic.patch_diag import (
            ARVIZ, _resolve_backend, _warn_jax_experimental,
        )

        _inc = include or var_names
        _exc = exclude or exclude_vars
        if _inc is not None or _exc is not None:
            self.filter_posteriors(include=_inc, exclude=_exc)

        _backend = _resolve_backend(
            backend if backend is not None else getattr(self, "diag_backend", ARVIZ))
        if backend is not None and _backend != ARVIZ:
            _warn_jax_experimental()

        if _backend == ARVIZ:
            self.tab_summary = self.summary_old(
                round_to=round_to, kind=kind, hdi_prob=hdi_prob,
                var_names=_inc, exclude_vars=_exc,
            )
            return self.tab_summary

        from BayesForge.Diagnostic.jax_diagnostics import summary as _jax_summary
        posteriors = (self.posteriors_by_chain if group_by_chain
                      and hasattr(self, 'posteriors_by_chain')
                      else self.posteriors)
        self.tab_summary = _jax_summary(
            posteriors,
            round_to=round_to,
            hdi_prob=hdi_prob,
            filter_regex=filter_regex,
            group_by_chain=group_by_chain,
        )
        return self.tab_summary

    def summary_jax(self, **kwargs):
        """Posterior summary via the experimental JAX/NumPy diagnostics."""
        kwargs.setdefault("backend", "jax")
        return self.summary(**kwargs)

    def summary_old(
        self,
        round_to=2,
        kind="all",
        hdi_prob=0.89,
        var_names=None,
        exclude_vars=None,
        *args,
        **kwargs,
    ):
        """ArviZ-based posterior summary (the default backend for summary())."""
        if az is None:
            raise ImportError(
                "arviz is unavailable, so the arviz summary backend cannot run. "
                "Use m.summary(backend='jax') or fix the arviz import.")
        if getattr(self.diag, "trace", None) is None:
            self.diag.to_az(backend=self.backend)
        # arviz renamed this argument: 0.x takes hdi_prob=, 1.x takes ci_prob=.
        # Hard-coding ci_prob raised TypeError on every arviz 0.x install.
        import inspect as _inspect
        _prob_kw = ("ci_prob" if "ci_prob" in _inspect.signature(az.summary).parameters
                    else "hdi_prob")
        self.tab_summary = az.summary(
            self.diag.trace,
            *args,
            round_to=round_to,
            kind=kind,
            **{_prob_kw: hdi_prob},
            **kwargs,
        )
        if var_names is not None or exclude_vars is not None:
            base_names = self.tab_summary.index.str.split("[").str[0]
            vars_to_keep = base_names.unique().tolist()
            if var_names is not None:
                if isinstance(var_names, str):
                    var_names = [var_names]
                vars_to_keep = [v for v in vars_to_keep if v in var_names]
            if exclude_vars is not None:
                if isinstance(exclude_vars, str):
                    exclude_vars = [exclude_vars]
                vars_to_keep = [v for v in vars_to_keep if v not in exclude_vars]
            self.tab_summary = self.tab_summary.loc[base_names.isin(vars_to_keep)]
        return self.tab_summary

    def filter_posteriors(self, include=None, exclude=None):
        """Filter m.posteriors in-place by parameter base name.

        Always filters from the original full posteriors (m.posteriors_full),
        so repeated calls reset rather than stack.

        Matching is exact on base name: exclude='var' removes 'var' (all its
        elements) but never 'var_1'. Index suffixes are stripped from filter
        names, so 'var[0,0]' and 'var' are equivalent.

        Args:
            include: str or list — keep only these parameters.
            exclude: str or list — remove these parameters.
        """
        if self.posteriors_full is None:
            raise RuntimeError("No posteriors found. Run fit() first.")

        def _base(name):
            return name.split('[')[0]

        def _as_set(arg):
            if arg is None:
                return None
            if isinstance(arg, str):
                arg = [arg]
            return {_base(n) for n in arg}

        inc = _as_set(include)
        exc = _as_set(exclude)

        def _filter(d):
            return {k: v for k, v in d.items()
                    if (inc is None or k in inc) and (exc is None or k not in exc)}

        self.posteriors = _filter(self.posteriors_full)
        if self.posteriors_by_chain_full is not None:
            self.posteriors_by_chain = _filter(self.posteriors_by_chain_full)

    def get_posterior_means(self):
        """
        Get the posterior means of the variables.

        Returns:
            Dictionary of variable names and their posterior means.
        """
        d = self.summary()
        posterior_means = d["mean"].values
        posterior_names = d.index.tolist()
        return {var: mean for var, mean in zip(posterior_names, posterior_means)}

    # Sample model ----------------------------------------------------------------------------
    def visit_call(self, node, obs_args):
        """
        Parse a function call node to find `lk` calls with `obs` arguments and add those argument names to the `obs_args` list.

        Args:
            node: AST node to parse.
            obs_args: List to store observed argument names.
        """
        # Check if the function called is `lk`
        if isinstance(node.func, ast.Name) and node.func.id == "lk":
            # Check for keyword arguments in the `lk` function
            for kw in node.keywords:
                if kw.arg == "obs":  # Look for `obs=`
                    # Add the variable name (if available) to obs_args
                    if isinstance(kw.value, ast.Name):
                        obs_args.append(kw.value.id)

    def find_obs_in_model(self, model_func):
        """
        Extract observed argument names from `obs` in `lk` calls in `model_func`.

        Args:
            model_func: Model function to analyze.

        Returns:
            List of observed argument values.
        """
        obs_values = []
        try:
            # Get the source code of the function
            source_code = inspect.getsource(self.model)
            # Parse the source code into an Abstract Syntax Tree
            tree = ast.parse(source_code)

            # Traverse the AST to find all function calls with 'obs' keyword argument
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "obs":
                            # Extract the value passed to 'obs'
                            value = ast.unparse(keyword.value)
                            obs_values.append(value)
        except (IndentationError, SyntaxError, OSError, TypeError) as e:
            # Fallback for dynamic/reticulate functions where source cannot be inspected
            pass

        self.obs_args = obs_values
        return obs_values

    # Create a new model function with the modified signature
    def build_model_with_Y_None(self, model):
        """
        Modify the original model function to make the observed arguments optional.

        Args:
            model: Model function to modify.

        Returns:
            Modified model function with optional observed arguments.
        """
        # Extract `obs` argument names
        obs = self.find_obs_in_model(model)
        # Modify the function's signature to make the observed argument optional
        sig = inspect.signature(model)
        parameters = []
        for name, param in sig.parameters.items():
            if name in obs:
                parameters.append(
                    inspect.Parameter(
                        name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    )
                )
            else:
                parameters.append(param)

        def model_with_None(*args, **kwargs):
            # Default values for obs arguments if not passed
            for obs_name in obs:
                if obs_name not in kwargs:
                    kwargs[obs_name] = None
            # Call the original model function with the modified arguments
            return model(*args, **kwargs)

        # Update the signature of the new model
        model_with_None.__signature__ = sig.replace(parameters=parameters)
        self.model2 = model_with_None
        return model_with_None

    def sample(self, data=None, remove_obs=True, posterior=True, samples=1, seed=0, infer_discrete=False):
        """
        Sample from the model.

        Args:
            data: Data to use for sampling. Defaults to None.
            remove_obs: Whether to remove observed data. Defaults to True.
                Forced to False when infer_discrete=True, since exact
                posterior sampling of discrete sites requires conditioning
                on the observed data.
            posterior: Whether to use posterior. Defaults to True.
            samples: Number of samples. Defaults to 1. Ignored when
                infer_discrete=True (sample count follows the posterior).
            seed: Random seed. Defaults to 0.
            infer_discrete: If True, draw exact posterior samples of
                discrete latent sites (e.g. auto-enumerated Bernoulli /
                Categorical sites) conditioned on the observed data,
                instead of forward-sampling them from their prior. Use
                this to recover discrete latents (e.g. mixture indicators)
                rather than the plain prior-predictive draw.

        Returns:
            Samples from the model.
        """
        rng_key = jax.random.PRNGKey(int(seed))
        self.build_model_with_Y_None(self.model)

        if data is None:
            data = self.data_on_model.copy()

        if infer_discrete:
            remove_obs = False

        if remove_obs:
            for intem in self.obs_args:
                del data[intem]

        if posterior == False:
            posterior = None
        else:
            posterior = self.sampler.get_samples()

        if infer_discrete:
            # numpyro.infer.Predictive(..., infer_discrete=True) unconditionally
            # masks every sample site's log_prob before enumerating (see
            # numpyro/infer/util.py:_predictive), which silently drops the
            # observed-data likelihood and collapses discrete sites to a
            # uniform draw. sample_discrete_posterior conditions correctly.
            return sample_discrete_posterior(
                self.model, posterior, rng_key, model_kwargs=data
            )

        predictive = Predictive(
            self.model2, posterior_samples=posterior, num_samples=samples
        )
        return predictive(rng_key, **data)
    # Log probability ----------------------------------------------------------------------------
    def log_prob(self, model, seed=None, **kwargs):
        """
        Compute the log probability of a model.

        Args:
            model: Model to compute log probability for.
            seed: Random seed. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing init_params, potential_fn, constrain_fn, and model_trace.
        """
        # getting log porbability
        rng_key = self._resolve_key(seed, "log_prob")
        init_params, potential_fn, constrain_fn, model_trace = (
            numpyro.infer.util.initialize_model(rng_key, model, model_args=(kwargs))
        )
        print("init_params:  ", init_params)
        print("constrain_fn: ", constrain_fn(init_params.z))
        print("potential_fn: ", -potential_fn(init_params.z))  # log prob
        print("grad:         ", jax.grad(potential_fn)(init_params.z))
        return init_params, potential_fn, constrain_fn, model_trace

    def get_history(self):
        """
        Get the history of the model fit.

        Returns:
            Dictionary containing model, model_name, data, sampler, and posteriors.
        """
        self.history = {
            "model": self.model,
            "model_name": self.run_model_name,
            "data": self.data_on_model,
            "sampler": self.sampler,
            "posteriors": self.posteriors,
        }

    def plot(self, X, y=None, figsize=(10, 6), **kwargs):
        """
        Plot the model results.

        Args:
            X: Data to plot.
            y: Target variable. Defaults to None.
            figsize: Figure size. Defaults to (10, 6).
            **kwargs: Additional keyword arguments.
        """
        if self.run_model_name == "gmm":
            plot_gmm(X, sampler=self.sampler, figsize=figsize)

        elif self.run_model_name == "dpmm":
            self.models.dpmm.plot(X, sampler=self.sampler, figsize=figsize)

        elif self.run_model_name == "pca":
            self.models.pca.plot()

    def save(self, path=None):
        """
        Save the BF object state to a file using cloudpickle.
        Includes data_on_model, model, posteriors, and other attributes.

        Args:
            path (str, optional): File path to save the object. Defaults to '{model_name}_bi.pkl' or 'BF_model.pkl' in CWD.
        """
        if path is None:
            path = f"{self.model_name}_bf.pkl" if self.model_name else "bf_model.pkl"

        try:
            with open(path, "wb") as f:
                cloudpickle.dump(self, f)
            print(f"BF object successfully saved to {path}")
        except Exception as e:
            print(f"Error saving BF object: {e}")

    @staticmethod
    def load(path=None):
        """
        Load a BF object state from a file using cloudpickle.

        Args:
            path (str, optional): File path to load the object from. Defaults to 'bf_model.pkl'.

        Returns:
            BF: The loaded BF instance.
        """
        if path is None:
            path = "bf_model.pkl"

        try:
            with open(path, "rb") as f:
                obj = cloudpickle.load(f)
            print(f"BF object successfully loaded from {path}")
            return obj
        except Exception as e:
            print(f"Error loading BF object: {e}")
            return None

    # ------------------------------------------------------------------
    # Multi-GPU helpers
    # ------------------------------------------------------------------

    def _maybe_warn_shard_limits(self):
        """Print the EXPERIMENTAL data-sharding limits warning once per instance.

        Called lazily from ``fit`` on the first sharded fit (not at
        construction). Sharding splits the leading (axis-0) dimension of each
        eligible data array across devices. This is only numerically correct
        when the model is a composition of *map* (per-row) and *reduce*
        (sum/mean/logsumexp over axis 0) operations. Operations that couple rows
        across the device boundary are NOT yet handled and can crash, deadlock
        (under vectorized chains), or — worst — bias the posterior silently.
        """
        if self._shard_warned:
            return
        self._shard_warned = True
        import warnings
        if self._data_sharding is None:
            warnings.warn(
                f"shard=True but only {self.n_devices} device is available — "
                "data sharding will have no effect. Create bf() with cores>1 "
                "(CPU) or run on multiple GPUs to enable it.",
                stacklevel=3,
            )
            return
        warnings.warn(
            "EXPERIMENTAL data sharding is ENABLED (axis-0 split across "
            f"{self.n_devices} devices). It is correct ONLY for models built "
            "from per-row (map) and axis-0 reductions (sum/mean/logsumexp). It "
            "can crash, deadlock, or SILENTLY bias the posterior for models "
            "that couple rows across the shard boundary, including:\n"
            "  1. cross-position ops: diff/lag/cumsum/scan, AR/Kalman/HMM/RNN/"
            "ODE, convolution, sort, FFT;\n"
            "  2. all-to-all/pairwise: outer products on axis 0, Gram/distance/"
            "kernel matrices, attention, reciprocity (Y + Y.T);\n"
            "  3. contraction over the sharded axis (dot products on the wrong "
            "side);\n"
            "  4. whole-array linear algebra: cholesky/inv/solve/qr/svd/eig/det;"
            "\n  5. index/segment ops crossing partitions: scatter, "
            "segment_sum, bincount, permutation gather;\n"
            "  6. a leading axis that is NOT the observation axis (feature- or "
            "time-major arrays).\n"
            "Control placement explicitly with m.shard(arr)/m.replicate(arr), "
            "and inspect m.shard_report() after fitting. These cases are being "
            "addressed incrementally.",
            stacklevel=3,
        )

    def _auto_shard_data(self, data):
        """Shard every eligible array in *data* across available devices.

        Called automatically by :meth:`fit` before passing ``data_on_model``
        to the sampler.  An array is sharded when:

        * ``_data_sharding`` is set (i.e. ``n_devices > 1``), **and**
        * the array has at least one dimension, **and**
        * its leading dimension is divisible by ``n_devices``
          (so the split is exact — no padding needed).

        Scalars, 0-D arrays, strings, and arrays whose leading dimension is
        not divisible by ``n_devices`` are passed through unchanged; XLA
        replicates them automatically on all devices.

        Behaviour notes
        ---------------
        * An array already placed by the caller with :meth:`shard` or
          :meth:`replicate` is left untouched — an explicit placement is never
          overridden (Improvement 4).
        * Arrays whose leading dimension is not divisible by ``n_devices`` are
          left for XLA to replicate, but the decision is reported via a warning
          rather than being silent (Improvement 3 / 5).
        * The per-array decisions are recorded in ``self._last_shard_report``
          (see :meth:`shard_report`).

        .. warning::
           This routine shards purely on leading-axis divisibility; it has no
           knowledge of how the model *consumes* each array. An array whose
           axis 0 is a "needs-all-N" axis (a broadcast partner in an outer
           product, a gather-index source, or a normalisation axis) will be
           sharded if its shape happens to divide, which can crash or, worse,
           silently bias the posterior. Keep such arrays replicated explicitly
           with :meth:`replicate`. Note also that for the matrix-form SRM the
           dyadic random-effect state (``dr_raw`` (2, N_dyads), ``dr_rf_mat``
           (N, N)) is sampled globally and stays replicated on every device, so
           sharding the data does not reduce that O(N^2) parameter memory
           (Improvement 7).

        Parameters
        ----------
        data : dict
            The ``data_on_model`` dictionary.

        Returns
        -------
        dict
            A new dictionary with eligible arrays replaced by sharded copies.
            The input dictionary is not mutated.
        """
        if self._data_sharding is None or data is None:
            return data
        import jax
        import jax.numpy as jnp
        import warnings

        out = {}
        sharded = []            # device-sharded along axis 0
        replicated_user = []    # left as-is because the caller already placed them
        not_divisible = []      # (name, leading_dim) left replicated by XLA
        passthrough = []        # scalars / static args / non-arrays

        for k, v in data.items():
            # Non-array values — python scalars used as static model args (N, K,
            # latent_dim), strings, None — are left untouched. device_put would
            # turn a static arg into a traced array and XLA replicates scalars
            # automatically anyway.
            if not hasattr(v, "shape") and not isinstance(v, (list, tuple)):
                out[k] = v
                passthrough.append(k)
                continue
            try:
                arr = jnp.asarray(v)
            except (TypeError, ValueError):
                out[k] = v
                passthrough.append(k)
                continue

            if arr.ndim == 0:                       # 0-D array → replicated scalar
                out[k] = v
                passthrough.append(k)
                continue

            # Improvement 4 — respect a sharding the caller already chose via
            # m.shard()/m.replicate(). Never override an explicit placement.
            existing = getattr(v, "sharding", None)
            if existing is not None and (
                existing == self._data_sharding or existing == self._rep_sharding
            ):
                out[k] = v
                (sharded if existing == self._data_sharding
                 else replicated_user).append(k)
                continue

            if arr.shape[0] % self.n_devices == 0:
                # device_put errors (e.g. OOM) are intentionally NOT swallowed —
                # they propagate so failures are visible (Improvement 5).
                out[k] = jax.device_put(arr, self._data_sharding)
                sharded.append(k)
            else:
                # Improvement 3 — never a silent no-op. The array is left for
                # XLA to replicate (memory behaviour unchanged) but the decision
                # is recorded and surfaced in the warning below.
                out[k] = v
                not_divisible.append((k, int(arr.shape[0])))

        # Improvement 5 — auditable report instead of silent guesswork.
        if not_divisible:
            details = ", ".join(f"{k}(leading={n})" for k, n in not_divisible)
            warnings.warn(
                f"Auto-shard: {len(not_divisible)} array(s) have a leading "
                f"dimension not divisible by n_devices={self.n_devices} and were "
                f"REPLICATED instead of sharded: {details}. Each device then "
                "computes these in full (no parallel speed-up). Choose an "
                "n_devices that divides the leading dimension to shard them.",
                stacklevel=2,
            )
        self._last_shard_report = {
            "n_devices": self.n_devices,
            "sharded": sharded,
            "replicated_not_divisible": [k for k, _ in not_divisible],
            "replicated_by_caller": replicated_user,
            "passthrough": passthrough,
        }
        return out

    def _shard_check(self, model, sharded_data, replicated_data):
        """Run the static + runtime shard safeguards (best-effort).

        Returns True if it is safe to proceed with sharding (no ✗ correctness
        verdict), False if the sharded log-density differs from the replicated
        reference (caller should fall back to replicated). Never raises; on any
        internal failure it warns and returns True (does not block the fit).
        """
        import os as _os
        if _os.environ.get("BF_SHARD_CHECK", "1") == "0":
            return True   # safeguards explicitly disabled (e.g. for benchmarking)
        try:
            from BayesForge.Diagnostic.sharding_safeguards import run_shard_check
            key = self._resolve_key(None, "shard_check")
            report = run_shard_check(model, sharded_data, replicated_data, key,
                                     name=str(self.model_name or ""))
        except Exception as _e:
            import warnings
            warnings.warn(f"shard-check skipped ({_e})", stacklevel=2)
            return True
        if report is None:
            return True
        if report.get("runtime_unsafe"):
            import warnings
            warnings.warn(
                "shard-check: sharded log-density differs from the replicated "
                "reference — running REPLICATED instead to preserve correctness.",
                stacklevel=2,
            )
            return False
        if report.get("perf_unsafe") and _os.environ.get("BF_SHARD_PERF", "1") != "0":
            import warnings
            back = report.get("backward", {}) or {}
            warnings.warn(
                "shard-check: backward-pass communication is O(N²)-dominant "
                f"(ratio {back.get('ratio', float('nan')):.2f} ≥ "
                f"{back.get('ratio_threshold', 0.25):.2f}) — the gradient all-reduce "
                "of a replicated/coupled parameter will not shrink with device count, "
                "so sharding cannot pay off. Running REPLICATED instead. Set "
                "BF_SHARD_PERF=0 to override.",
                stacklevel=2,
            )
            return False
        return True

    def shard_report(self):
        """Return the per-array sharding decisions from the most recent fit.

        After a ``fit(..., shard=True)`` on a multi-device setup, this returns a
        dict with the keys ``sharded``, ``replicated_not_divisible``,
        ``replicated_by_caller`` and ``passthrough`` listing the array names in
        each category, plus ``n_devices``. Returns ``None`` if no sharded fit
        has run yet. Use it to audit what the automatic sharder actually did.
        """
        return getattr(self, "_last_shard_report", None)

    def shard(self, array):
        """Shard *array* along its leading axis across all available GPUs.

        Wraps ``jax.device_put(array, data_sharding)`` using the mesh
        created at initialisation (only available when ``platform='gpu'``
        and more than one GPU is present).  Pass the returned sharded array
        as an observation or covariate to the model; XLA will automatically
        compute each GPU's slice of the log-likelihood in parallel.

        On a single-GPU or CPU setup the array is returned unchanged.

        Parameters
        ----------
        array : array-like
            Any array convertible by JAX (numpy, jax, list …).

        Returns
        -------
        jax.Array
            The input array, sharded across GPUs when available.

        Example
        -------
        >>> m = bf(platform='gpu')
        >>> Y_sharded = m.shard(jnp.array(Y))
        >>> m.fit(model, obs=dict(Y=Y_sharded), num_chains=m.n_gpus)
        """
        import jax
        import jax.numpy as jnp
        arr = jnp.asarray(array)
        if self._data_sharding is None:
            return arr
        return jax.device_put(arr, self._data_sharding)

    def replicate(self, array):
        """Place an identical copy of *array* on every GPU.

        Useful for small parameter arrays or constants that every GPU needs
        in full rather than a sharded slice.  On a single-GPU or CPU setup
        the array is returned unchanged.

        Parameters
        ----------
        array : array-like
            Any array convertible by JAX.

        Returns
        -------
        jax.Array
            The input array, replicated on all GPUs when available.
        """
        import jax
        import jax.numpy as jnp
        arr = jnp.asarray(array)
        if self._rep_sharding is None:
            return arr
        return jax.device_put(arr, self._rep_sharding)

    # ------------------------------------------------------------------
    # Parallel simulation scenarios
    # ------------------------------------------------------------------
    # Exposed as staticmethods (like `load` above) because each simulation
    # builds its *own* bf instance in its own process -- there is no `self` to
    # speak of. Imported lazily to keep BayesForge.Parallel free of a circular
    # import back into this module.

    @staticmethod
    def run_simulations(*args, **kwargs):
        """Run independent simulations in parallel. See
        :func:`BayesForge.Parallel.simulate.run_simulations`."""
        from BayesForge.Parallel.simulate import run_simulations as _run
        return _run(*args, **kwargs)

    @staticmethod
    def grid(*args, **kwargs):
        """Build a scenario grid. See
        :func:`BayesForge.Parallel.simulate.grid`."""
        from BayesForge.Parallel.simulate import grid as _grid
        return _grid(*args, **kwargs)


# Alias for backward compatibility
BF = bf
