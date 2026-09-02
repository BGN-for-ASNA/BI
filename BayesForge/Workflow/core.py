"""Workflow: multi-fit orchestration for BF.

``m.fit()`` fits a model once, on data you hand it. Every workflow pattern
in Gelman/Vehtari/McElreath's *Bayesian Workflow* that needs many fits over
many simulated datasets -- parameter recovery, simulation-based calibration
checking, prior-vs-posterior contraction -- currently has to be hand-rolled
in a loop around ``m.fit``. ``Workflow`` packages those loops as methods on
``m.workflow``, in the same spirit as ``m.models``/``m.diag`` (a thin
accessor bound to the parent ``bf`` instance).

    from BayesForge import bf
    m = bf(platform="cpu")
    result = m.workflow.recover(model=my_model, dgp=my_dgp,
                                 param_names=["alpha", "beta"], n_sim=100)
    print(result.summary())
    m.workflow.plot_recovery(result).show()

See ``parallel.py`` for what "parallel" means here and why it has two paths.
"""
import os

from .metrics import recovery_metrics, sbc_uniformity
from .results import RecoveryResult, SBCResult, ContrastResult
from .parallel import run_parallel, probe_parallel_capabilities
from .diagnostics import annotated_summary, advise
from .visualization import plot_recovery, plot_sbc_rank, plot_annotated_summary, save_figure


class Workflow:
    """Multi-fit workflow orchestration bound to one ``bf`` instance.

    Args:
        parent: The ``bf`` instance this accessor is attached to (``self`` in
            ``bf.__init__``, exposed to users as ``m.workflow``).
    """

    def __init__(self, parent):
        self.m = parent
        self.last_recovery = None
        self.last_sbc = None

    # ------------------------------------------------------------------
    # Parameter recovery
    # ------------------------------------------------------------------
    def recover(self, model=None, dgp=None, param_names=None, n_sim=100,
                model_factory=None, dgp_factory=None, n_jobs=1,
                platform="cpu", cores=None, fit_kwargs=None, hdi_prob=0.89,
                results_dir=None, verbose=True):
        """Simulation study: repeatedly simulate -> fit -> check recovery.

        Two calling conventions, matching the two parallelization paths in
        ``parallel.py``:

        * ``n_jobs=1`` (default): pass a ``model``/``dgp`` already bound to
          *this* ``m`` (the normal case -- cheap, reuses this instance).
        * ``n_jobs>1``: BF model closures capture one specific ``bf``
          instance and can't be pickled to a worker process, so instead pass
          ``model_factory``/``dgp_factory`` -- callables that take a fresh
          ``bf`` instance and return a bound model/dgp. Each of the ``n_jobs``
          worker processes builds its own instance from these factories.

        Args:
            model: Model function bound to ``self.m`` (n_jobs=1 path).
            dgp: Callable ``() -> (true_params: dict, data: dict)`` bound to
                ``self.m`` (n_jobs=1 path). Every stochastic draw inside it
                should use ``sample=True`` (see ``bf://how-to/python/data-generation``).
            param_names: Parameter names to track (must match ``name=`` in
                ``model`` and keys returned by ``dgp``/``true_params``).
            n_sim: Number of simulated datasets.
            model_factory, dgp_factory: ``Callable[[bf], model|dgp]`` for the
                n_jobs>1 path.
            n_jobs: 1 (sequential, default) or >1 (multiprocess).
            platform, cores: Forwarded to each worker's ``bf(...)`` when
                n_jobs>1 (ignored otherwise -- the existing ``self.m`` is used).
            fit_kwargs: Extra kwargs forwarded to ``m.fit`` (e.g.
                ``dict(num_warmup=500, num_samples=500, num_chains=2)`` for a
                fast screening pass, or ``dict(chain_method="parallel")`` --
                already BF's default). See ``m.workflow.parallel_report()``
                to inspect what chain-level parallelism is actually available.
            hdi_prob: HDI mass used for the coverage check (default 0.89,
                matching BF's/the book's convention).
            results_dir: If given, writes ``experiment_log.csv`` and
                ``recovery_scatter.png`` there (the artifact convention used
                throughout the ``dgp_estimation``/``model_improvement``
                workflow guides).
            verbose: Print progress in the sequential (n_jobs=1) path.

        Returns:
            :class:`~BayesForge.Workflow.results.RecoveryResult`.
        """
        import jax.numpy as jnp
        import pandas as pd

        if not param_names:
            raise ValueError("recover() requires param_names.")
        fit_kwargs = dict(fit_kwargs or {})

        if n_jobs > 1:
            if model_factory is None or dgp_factory is None:
                raise ValueError(
                    "n_jobs>1 requires model_factory and dgp_factory (each a "
                    "callable taking a fresh bf instance and returning a bound "
                    "model/dgp) -- BF model closures capture a specific bf() "
                    "instance and cannot be pickled across processes. See the "
                    "Workflow.recover docstring."
                )
            rows = run_parallel(
                n_sim, n_jobs=n_jobs, model_factory=model_factory,
                dgp_factory=dgp_factory, param_names=param_names,
                fit_kwargs=fit_kwargs, platform=platform, cores=cores,
                hdi_prob=hdi_prob, mode="recover",
            )
        else:
            if model is None or dgp is None:
                raise ValueError("recover() requires model and dgp when n_jobs=1.")
            lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
            rows = []
            for i in range(n_sim):
                true_params, data = dgp()
                self.m.fit(model, obs=data, **fit_kwargs)
                row = {"sim": i}
                for name in param_names:
                    draws = jnp.asarray(self.m.posteriors[name]).reshape(-1)
                    lo, hi = jnp.quantile(draws, jnp.array([lo_q, hi_q]))
                    true_val = float(true_params[name])
                    row[f"{name}_true"] = true_val
                    row[f"{name}_mean"] = float(draws.mean())
                    row[f"{name}_hdi_lo"] = float(lo)
                    row[f"{name}_hdi_hi"] = float(hi)
                    row[f"{name}_covered"] = bool(lo <= true_val <= hi)
                rows.append(row)
                if verbose:
                    print(f"[workflow.recover] simulation {i + 1}/{n_sim} done")

        table = pd.DataFrame(rows)
        metrics = recovery_metrics(table, param_names)
        result = RecoveryResult(table=table, metrics=metrics,
                                param_names=list(param_names), hdi_prob=hdi_prob,
                                n_jobs=n_jobs)
        self.last_recovery = result

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            table.to_csv(os.path.join(results_dir, "experiment_log.csv"), index=False)
            save_figure(plot_recovery(result),
                       os.path.join(results_dir, "recovery_scatter.png"))
        return result

    # ------------------------------------------------------------------
    # Simulation-based calibration checking
    # ------------------------------------------------------------------
    def sbc(self, model=None, dgp=None, param_names=None, n_sbc=200,
            n_post_draws=1000, model_factory=None, dgp_factory=None,
            n_jobs=1, platform="cpu", cores=None, fit_kwargs=None,
            results_dir=None, verbose=True):
        """Simulation-based calibration: rank-uniformity check of the fitting
        procedure itself (not any one dataset).

        Same two calling conventions as :meth:`recover` (n_jobs=1 -> bound
        model/dgp; n_jobs>1 -> factories). Each replication draws
        ``(true_params, data)`` from ``dgp``, fits with a single chain (SBC
        needs *one* posterior per replication, not several averaged
        together), and records the rank of the true value among that
        replication's posterior draws. Across many replications those ranks
        should be uniform; a skew reveals a specific bias in the model code,
        priors, or sampler.

        Args:
            model, dgp: n_jobs=1 path -- see :meth:`recover`.
            param_names: Parameters to check.
            n_sbc: Number of replications (200 is the SBC literature's usual
                minimum for a stable rank histogram).
            n_post_draws: Posterior draws per replication (used as
                ``num_samples`` unless overridden in ``fit_kwargs``).
            model_factory, dgp_factory, n_jobs, platform, cores: n_jobs>1
                path -- see :meth:`recover`.
            fit_kwargs: Extra kwargs forwarded to ``m.fit``. Defaults to
                ``num_samples=n_post_draws, num_chains=1`` if not overridden.
            results_dir: If given, writes ``sbc_ranks.csv`` and
                ``sbc_rank_histogram.png``.
            verbose: Print progress in the sequential (n_jobs=1) path.

        Returns:
            :class:`~BayesForge.Workflow.results.SBCResult`.
        """
        import jax.numpy as jnp
        import pandas as pd

        if not param_names:
            raise ValueError("sbc() requires param_names.")
        fit_kwargs = dict(fit_kwargs or {})
        fit_kwargs.setdefault("num_samples", n_post_draws)
        fit_kwargs.setdefault("num_chains", 1)

        if n_jobs > 1:
            if model_factory is None or dgp_factory is None:
                raise ValueError(
                    "n_jobs>1 requires model_factory and dgp_factory -- see the "
                    "Workflow.recover docstring for why."
                )
            rows = run_parallel(
                n_sbc, n_jobs=n_jobs, model_factory=model_factory,
                dgp_factory=dgp_factory, param_names=param_names,
                fit_kwargs=fit_kwargs, platform=platform, cores=cores,
                hdi_prob=0.89, mode="sbc",
            )
        else:
            if model is None or dgp is None:
                raise ValueError("sbc() requires model and dgp when n_jobs=1.")
            rows = []
            for i in range(n_sbc):
                true_params, data = dgp()
                self.m.fit(model, obs=data, **fit_kwargs)
                row = {"sim": i}
                for name in param_names:
                    draws = jnp.asarray(self.m.posteriors[name]).reshape(-1)
                    true_val = float(true_params[name])
                    row[f"{name}_rank"] = int((draws < true_val).sum())
                    row[f"{name}_n_draws"] = int(draws.size)
                rows.append(row)
                if verbose:
                    print(f"[workflow.sbc] replication {i + 1}/{n_sbc} done")

        table = pd.DataFrame(rows)
        n_draws_actual = (int(table[f"{param_names[0]}_n_draws"].iloc[0])
                          if len(table) else n_post_draws)
        uniformity = sbc_uniformity(table, param_names, n_draws_actual)
        result = SBCResult(table=table, uniformity=uniformity,
                           param_names=list(param_names),
                           n_post_draws=n_draws_actual, n_jobs=n_jobs)
        self.last_sbc = result

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            table.to_csv(os.path.join(results_dir, "sbc_ranks.csv"), index=False)
            save_figure(plot_sbc_rank(result),
                       os.path.join(results_dir, "sbc_rank_histogram.png"))
        return result

    # ------------------------------------------------------------------
    # Posterior arithmetic: contrasts, poststratification, decisions
    # ------------------------------------------------------------------
    def contrast(self, expr, name=None, hdi_prob=0.89):
        """Evaluate a posterior contrast expression over ``m.posteriors``.

        Args:
            expr: A Python expression string evaluated with each entry of
                ``m.posteriors`` bound as a local variable (plus ``jnp``),
                e.g. ``"alpha[:, 0] - alpha[:, 1]"``. Evaluated with an empty
                ``__builtins__`` -- intended for trusted expressions you
                write yourself, the same trust model as the
                ``compute_contrasts`` MCP tool's ``expr`` field.
            name: Label for the result (default: ``expr`` itself).
            hdi_prob: HDI mass for the interval.

        Returns:
            :class:`~BayesForge.Workflow.results.ContrastResult`.
        """
        import jax.numpy as jnp

        if self.m.posteriors is None:
            raise RuntimeError("No posteriors found. Run m.fit() first.")
        local_ns = {k: jnp.asarray(v) for k, v in self.m.posteriors.items()}
        local_ns["jnp"] = jnp
        value = eval(expr, {"__builtins__": {}}, local_ns)
        value = jnp.asarray(value).reshape(-1)
        lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
        lo, hi = jnp.quantile(value, jnp.array([lo_q, hi_q]))
        return ContrastResult(name=name or expr, mean=float(value.mean()),
                              hdi_lo=float(lo), hdi_hi=float(hi),
                              p_positive=float((value > 0).mean()),
                              hdi_prob=hdi_prob)

    def poststratify(self, param_name, weights, hdi_prob=0.89):
        """Population-weighted average of a per-cell posterior parameter.

        Args:
            param_name: Name of a vector-valued posterior parameter in
                ``m.posteriors`` with shape ``(draws, n_cells)`` (e.g. a
                varying intercept indexed by demographic/geographic cell).
            weights: Length ``n_cells`` population weights (need not be
                pre-normalized).
            hdi_prob: HDI mass for the interval.

        Returns:
            :class:`~BayesForge.Workflow.results.ContrastResult` for the
            population-level (poststratified) estimate.
        """
        import jax.numpy as jnp

        if self.m.posteriors is None:
            raise RuntimeError("No posteriors found. Run m.fit() first.")
        draws = jnp.asarray(self.m.posteriors[param_name])
        w = jnp.asarray(weights, dtype=draws.dtype)
        w = w / jnp.sum(w)
        pooled = draws @ w if draws.ndim > 1 else draws
        lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
        lo, hi = jnp.quantile(pooled, jnp.array([lo_q, hi_q]))
        return ContrastResult(name=f"poststratified({param_name})",
                              mean=float(pooled.mean()), hdi_lo=float(lo),
                              hdi_hi=float(hi),
                              p_positive=float((pooled > 0).mean()),
                              hdi_prob=hdi_prob)

    def decide(self, utility_fn, actions, hdi_prob=0.89):
        """Rank candidate decisions by expected utility, evaluated per draw.

        Never plugs in a point estimate first: ``utility_fn`` is applied to
        every posterior draw of each action's outcome, then averaged --
        propagating full posterior uncertainty into the decision.

        Args:
            utility_fn: ``Callable[[array], array]`` mapping an array of
                per-draw outcomes to per-draw utilities.
            actions: ``dict[name -> array-like]`` of per-draw outcome arrays,
                one entry per candidate action (e.g. built from
                ``m.posteriors`` under different covariate settings).
            hdi_prob: HDI mass for each action's utility interval.

        Returns:
            A pandas DataFrame, one row per action, sorted by
            ``expected_utility`` descending, with ``hdi_lo``/``hdi_hi``
            showing how confidently that ranking is known.
        """
        import jax.numpy as jnp
        import pandas as pd

        lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
        rows = []
        for name, outcome in actions.items():
            outcome = jnp.asarray(outcome)
            util = jnp.asarray(utility_fn(outcome)).reshape(-1)
            lo, hi = jnp.quantile(util, jnp.array([lo_q, hi_q]))
            rows.append(dict(action=name, expected_utility=float(util.mean()),
                             hdi_lo=float(lo), hdi_hi=float(hi)))
        return (pd.DataFrame(rows)
                .sort_values("expected_utility", ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Diagnostics / advice / parallel capability report
    # ------------------------------------------------------------------
    def annotated_summary(self, round_to=4, hdi_prob=0.89, rhat_threshold=1.01,
                          ess_threshold=400, mcse_frac_threshold=0.1,
                          include=None, exclude=None):
        """``m.summary()`` with an added per-parameter verdict/interpretation
        column. See :func:`BayesForge.Workflow.diagnostics.annotated_summary`.
        """
        return annotated_summary(
            self.m, round_to=round_to, hdi_prob=hdi_prob,
            rhat_threshold=rhat_threshold, ess_threshold=ess_threshold,
            mcse_frac_threshold=mcse_frac_threshold, include=include, exclude=exclude,
        )

    def advise(self, data=None, n_params=None, dgp=None, model=None):
        """Pre-fit checklist for DGP conventions and MCMC-vs-SVI choice.
        See :func:`BayesForge.Workflow.diagnostics.advise`.
        """
        return advise(m=self.m, data=data, n_params=n_params, dgp=dgp, model=model)

    def parallel_report(self, verbose=True):
        """Report what parallel execution is actually available right now.
        See :func:`BayesForge.Workflow.parallel.probe_parallel_capabilities`.
        """
        return probe_parallel_capabilities(m=self.m, verbose=verbose)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_recovery(self, result=None, param_names=None):
        """Plot a :meth:`recover` result (default: the most recent one)."""
        return plot_recovery(result or self.last_recovery, param_names=param_names)

    def plot_sbc(self, result=None, param_names=None, bins=None):
        """Plot a :meth:`sbc` result (default: the most recent one)."""
        return plot_sbc_rank(result or self.last_sbc, param_names=param_names, bins=bins)

    def plot_annotated_summary(self, table=None, **kwargs):
        """Plot an :meth:`annotated_summary` table (computes one if omitted)."""
        return plot_annotated_summary(table if table is not None
                                      else self.annotated_summary(**kwargs))
