"""Run independent simulation scenarios in parallel worker processes.

A simulation study -- parameter recovery, a scenario sweep, prior sensitivity --
is a loop of *independent* fits. ``fit`` already parallelises what happens
*inside* one fit (``num_chains`` with ``chain_method='parallel'``, and the
experimental data sharding behind ``fit(shard=True)``), but the loop over
datasets stays serial, so one core works while the rest idle.

:func:`run_simulations` spreads that loop over worker processes. Each worker is
a fresh Python process with its own JAX backend, so each simulation gets its own
``bf`` instance, its own device budget and its own RNG -- exactly the isolation
a per-fit ``cores=`` argument could never provide, because the CPU device count
is written into ``XLA_FLAGS`` once per process (see
``BayesForge/SetDevice/set.py``) and is ignored on any later change.

The scenario parameters travel with the results: every row of the returned
DataFrame carries the settings that produced it, so joining true against
estimated values needs no bookkeeping of your own.

This module deliberately imports neither ``jax`` nor ``BayesForge`` at module
level -- worker processes must be able to set their XLA environment before the
backend is initialised.
"""

import hashlib
import itertools
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager

__all__ = ["grid", "run_simulations"]


_GUARD_MSG = (
    "run_simulations uses 'spawn' worker processes, which re-import the "
    "calling script. Module-level code therefore runs again in every worker. "
    "Put the call behind a main guard:\n\n"
    "    if __name__ == '__main__':\n"
    "        res = bf.run_simulations(one_sim, scenarios, workers=8)\n\n"
    "Notebooks and the REPL are unaffected; this applies to .py scripts. Use "
    "backend='serial' to run without worker processes."
)


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------

def grid(reps=1, **params):
    """Build a list of scenario dicts from the cartesian product of ``params``.

    Args:
        reps: Number of replicates of each parameter combination.
        **params: Named parameter lists to cross. Scalars are treated as
            single-element lists.

    Returns:
        list[dict]: One dict per simulation, each carrying its ``sim`` (global
        index) and ``rep`` (replicate index within its cell) alongside the
        parameter values.

    Examples:
        >>> grid(N=[50, 100], rho=[0.0, 0.3], reps=2)   # doctest: +SKIP
        [{'sim': 0, 'rep': 0, 'N': 50, 'rho': 0.0},
         {'sim': 1, 'rep': 1, 'N': 50, 'rho': 0.0}, ...]   # 8 dicts
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")

    names = list(params)
    values = [v if isinstance(v, (list, tuple)) else [v] for v in params.values()]

    scenarios = []
    for combo in itertools.product(*values) if names else [()]:
        for rep in range(reps):
            cell = dict(zip(names, combo))
            cell["sim"] = len(scenarios)
            cell["rep"] = rep
            scenarios.append(cell)
    return scenarios


def _normalise_scenarios(scenarios):
    """Accept an int, a list of dicts, or the output of :func:`grid`."""
    if isinstance(scenarios, int):
        if scenarios < 1:
            raise ValueError(f"scenarios must be >= 1, got {scenarios}")
        scenarios = [{} for _ in range(scenarios)]
    else:
        scenarios = [dict(s) for s in scenarios]
        if not all(isinstance(s, dict) for s in scenarios):
            raise TypeError("scenarios must be an int or an iterable of dicts")

    for i, s in enumerate(scenarios):
        s.setdefault("sim", i)
        s.setdefault("rep", 0)
    return scenarios


def _derive_seed(base, sim):
    """Deterministic per-simulation seed.

    Uses sha256 rather than ``hash()`` because Python salts string hashes per
    process -- workers must agree with the parent and with each other.
    """
    digest = hashlib.sha256(f"{int(base)}:{int(sim)}".encode()).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# Result normalisation
# --------------------------------------------------------------------------

def _looks_like_bf(obj):
    cls = type(obj)
    return (
        cls.__name__ in ("bf", "BF")
        and getattr(cls, "__module__", "").startswith("BayesForge")
    )


def _normalise_result(ret):
    """Turn whatever the user's function returned into a list of row dicts."""
    if _looks_like_bf(ret):
        raise TypeError(
            "run_simulations: the simulation function returned a `bf` object. "
            "A fitted bf cannot cross a process boundary -- m.diag holds "
            "closures over m, and m.dist is an unpicklable proxy. Return "
            "summaries instead, e.g. m.summary(), m.sampler_stats(), or a dict "
            "of the numbers you need."
        )

    if ret is None:
        return [{}]

    # pandas objects, imported lazily so this module stays import-light.
    mod = type(ret).__module__ or ""
    if mod.startswith("pandas"):
        import pandas as pd
        if isinstance(ret, pd.DataFrame):
            return ret.to_dict("records")
        if isinstance(ret, pd.Series):
            return [{str(k): v for k, v in ret.items()}]

    if isinstance(ret, dict):
        return [ret]

    if isinstance(ret, (list, tuple)):
        if not ret:
            return [{}]
        if all(isinstance(r, dict) for r in ret):
            return [dict(r) for r in ret]
        raise TypeError(
            "run_simulations: a list return value must contain dicts, got "
            f"{type(ret[0]).__name__}"
        )

    return [{"value": ret}]


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_STATE = {}


def _worker_env(cores_per_worker, quiet=True):
    """Environment for one worker.

    This governs the JAX *device* count and the BLAS/OpenMP pools that numpy and
    friends consult. It does NOT bound XLA's own CPU thread pool -- that follows
    the affinity mask, so :func:`_pin_worker` is what keeps workers off each
    other's cores.

    ``BF_QUIET`` must be in the environment *inherited at spawn*, not merely set
    by the pool initializer: BayesForge prints its banner while the child
    imports this module to find the initializer, which is strictly earlier.
    """
    return {
        "BF_QUIET": "1" if quiet else "0",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={cores_per_worker}",
        "BF_MAX_CORES": str(cores_per_worker),
        "OMP_NUM_THREADS": str(cores_per_worker),
        "MKL_NUM_THREADS": str(cores_per_worker),
        "OPENBLAS_NUM_THREADS": str(cores_per_worker),
        "NUMEXPR_NUM_THREADS": str(cores_per_worker),
    }


@contextmanager
def _env_override(overrides):
    """Temporarily set env vars, restoring the previous values afterwards.

    Applied in the *parent* around the pool's lifetime: ``spawn`` children copy
    ``os.environ`` at creation, and ProcessPoolExecutor creates them lazily as
    tasks are submitted, so the override has to outlive the submissions.
    """
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _pin_worker(block, block_size):
    """Confine this worker to its own ``block_size`` CPUs, once.

    This is the part that actually limits XLA. Its CPU thread pool is sized from
    the process's affinity mask -- not from OMP_NUM_THREADS, and not from any
    XLA_FLAGS thread setting. Unpinned, every worker starts a pool sized for the
    whole machine and N workers oversubscribe it N-fold (measured: ~104 threads
    per worker on a 24-core host, against 27 when pinned).

    The block index comes from the worker's *first* task rather than a shared
    counter: XLA sizes its thread pool once, when the backend comes up, so only
    the first pin can affect it -- and avoiding the shared counter avoids
    handing every worker a semaphore to unpickle, which races with pool
    teardown. Workers pick up their first tasks in submission order, so the
    blocks they claim are distinct.

    ``block_size`` is the machine divided by the worker count, NOT
    ``cores_per_worker`` -- the latter is a JAX *device* count. Pinning to
    ``cores_per_worker`` would strand every core beyond
    ``workers * cores_per_worker`` (measured: with 12 workers at 1 core each on
    a 24-core host, half the machine sat idle and the run was slower than no
    pinning at all).

    Returns the assigned CPU list, or None where affinity is unavailable
    (non-Linux) or there are too few CPUs to divide.
    """
    if "cpus" in _STATE:                      # already pinned; never re-pin
        return _STATE["cpus"]

    _STATE["cpus"] = None
    if not hasattr(os, "sched_setaffinity"):
        return None

    # sched_getaffinity, not cpu_count: respects taskset and cgroup limits.
    cpus = sorted(os.sched_getaffinity(0))
    if block_size < 1 or len(cpus) < block_size:
        return None

    start = (block * block_size) % len(cpus)
    mine = {cpus[(start + k) % len(cpus)] for k in range(block_size)}
    try:
        os.sched_setaffinity(0, mine)
    except OSError:
        return None

    _STATE["cpus"] = sorted(mine)
    return _STATE["cpus"]


def _worker_init(env, fn_bytes, fn_kwargs_bytes, quiet, block_size):
    """Pool initializer: set the environment, then unpack the payloads."""
    os.environ.update(env)
    _STATE["block_size"] = block_size

    if quiet:
        # bf.__init__ prints its device count and BayesForge prints a banner on
        # import; times a hundred simulations that is pure noise.
        sys.stdout = open(os.devnull, "w")

    # Deliberately NOT deserialised here: fn_kwargs may hold jax arrays, and
    # rebuilding those brings the XLA backend up -- which must happen after the
    # worker is pinned, or the thread pool is sized for the whole machine.
    _STATE["fn_bytes"] = fn_bytes
    _STATE["fn_kwargs_bytes"] = fn_kwargs_bytes
    _STATE["worker"] = os.getpid()


def _worker_entry(payload_bytes, block):
    """Run one scenario. Returns cloudpickled ``(rows, error)``."""
    import cloudpickle

    _pin_worker(block, _STATE.get("block_size", 1))

    if "fn" not in _STATE:                    # first task: unpack, now pinned
        _STATE["fn"] = cloudpickle.loads(_STATE["fn_bytes"])
        blob = _STATE["fn_kwargs_bytes"]
        _STATE["fn_kwargs"] = cloudpickle.loads(blob) if blob else {}

    scenario = cloudpickle.loads(payload_bytes)
    rows, error = _run_one(
        _STATE["fn"], scenario, _STATE["fn_kwargs"], _STATE.get("worker", os.getpid())
    )
    return cloudpickle.dumps((rows, error))


def _run_one(fn, scenario, fn_kwargs, worker):
    """Execute one simulation and shape its output into rows.

    Runs identically in-process (``backend='serial'``) and in a worker, so the
    serial path is a genuine debugging equivalent rather than a second code
    path that can drift.
    """
    started = time.time()
    call = dict(fn_kwargs)
    call.update(scenario)

    try:
        ret = fn(**call)
        rows = _normalise_result(ret)
        error = None
    except Exception:
        rows = [{}]
        error = traceback.format_exc()

    elapsed = time.time() - started
    meta = dict(scenario)
    meta.update(worker=worker, elapsed_s=round(elapsed, 3), error=error)

    # Scenario/meta first so a result key of the same name wins -- the
    # simulation's own output is the more specific value.
    out = []
    for row in rows:
        merged = dict(meta)
        merged.update(row)
        out.append(merged)
    return out, error


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_simulations(
    fn,
    scenarios,
    workers=None,
    num_chains=None,
    cores_per_worker=None,
    fn_kwargs=None,
    seed=0,
    on_error="record",
    quiet=True,
    progress=True,
    backend="process",
):
    """Run independent simulations in parallel and collect them into a DataFrame.

    Each scenario dict is passed to ``fn`` as keyword arguments, together with
    an injected ``sim`` (index), ``rep`` (replicate) and ``seed``. Thread that
    ``seed`` into ``bf(rand_seed=seed)`` -- ``bf.__init__`` seeds the global
    numpy/random RNGs and the shared distribution singletons from it, so
    workers that share a seed draw identical data.

    Args:
        fn: The simulation function. Called as ``fn(**scenario, **fn_kwargs)``;
            give it a ``**_`` catch-all so added keys do not break it. It may
            return a dict, a list of dicts, a ``pandas`` DataFrame or Series, or
            None. It must NOT return a ``bf`` instance.
        scenarios: An int (that many bare simulations), a list of dicts, or the
            output of :func:`grid`.
        workers: Number of worker processes, i.e. how many simulations run at
            once. Defaults to ``min(len(scenarios), n_cpu // cores_per_worker)``
            -- so it follows ``num_chains`` automatically. See
            "Chains, workers and cores".
        num_chains: Chains each fit will draw, for fits using
            ``chain_method='parallel'`` (setup B). Reserves one core and one JAX
            device per chain in every worker, and is passed to ``fn`` as a
            keyword so the core budget and the fit cannot disagree. On 24 cores,
            ``num_chains=4`` means 4 cores per worker and 6 simulations at a
            time. Omit it for single-chain fits (A) and for
            ``chain_method='vectorized'`` (C), which needs no reserved cores.
        cores_per_worker: JAX *devices* per worker, to set the budget directly
            instead of via ``num_chains``. Defaults to ``num_chains``, else 1.
            It does not decide how many CPUs a worker gets -- the machine's
            cores are always divided evenly across ``workers`` so none sit idle
            -- only how many devices JAX exposes inside one worker. Setting it
            below ``num_chains`` warns, because parallel chains would then fall
            back to sequential.
        fn_kwargs: Constants passed to every call. Serialised once per worker
            rather than once per scenario, so this is where large shared arrays
            belong.
        seed: Base seed. Per-simulation seeds are derived from it
            deterministically, so a rerun reproduces regardless of how many
            workers ran or in what order they finished.
        on_error: ``'record'`` (default) keeps going and puts the traceback in
            the row's ``error`` column; ``'raise'`` aborts the run.
        quiet: Suppress worker stdout.
        progress: Print completion progress to stdout.
        backend: ``'process'`` (default) or ``'serial'``. The serial path runs
            in-process, so ``pdb`` and full tracebacks work while debugging.

    Returns:
        pandas.DataFrame: One row per dict returned by ``fn``, carrying the
        scenario columns plus ``sim``, ``rep``, ``seed``, ``worker``,
        ``elapsed_s`` and ``error``. Sorted by ``sim``.

    Chains, workers and cores:
        Chains and simulations compete for the same cores, so ``num_chains``,
        ``workers`` and the fit's ``chain_method`` have to be chosen together.
        There are three sensible setups. All three produce the same number of
        genuine chains and equivalent ESS -- they differ only in speed and in
        how many simulations are in flight at once.

        **A. Single-chain fits** -- the default, and the fastest way through a
        large batch when you do not need per-fit R-hat::

            def one_sim(N, seed, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m), num_chains=1,
                      progress_bar=False)
                ...

            run_simulations(one_sim, scenarios)          # workers = every core

        One core per worker, 24 simulations at a time on a 24-core host.

        **B. Chains reserved per worker** -- ``chain_method='parallel'``, which
        places one chain on one JAX device and so needs a device per chain.
        Pass ``num_chains`` here and take it back in ``fn``, so the core budget
        and the fit can never disagree::

            def one_sim(N, seed, num_chains, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m),
                      num_chains=num_chains, chain_method="parallel",
                      progress_bar=False)
                ...

            run_simulations(one_sim, scenarios, num_chains=4)

        ``num_chains=4`` gives each worker 4 cores and 4 JAX devices, sets
        ``workers = n_cpu // 4``, and pins each worker to its own disjoint block
        of 4 CPUs. On 24 cores: 6 workers, so **6 simulations run at a time**
        instead of 24. Simulation concurrency is traded for chain concurrency.

        **C. Chains vectorized** -- ``chain_method='vectorized'`` vmaps the
        chains inside a *single* device, so no cores are reserved for them and
        every core stays free for a different simulation::

            def one_sim(N, seed, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m), num_chains=4,
                      chain_method="vectorized", progress_bar=False)
                ...

            run_simulations(one_sim, scenarios)   # no num_chains needed here

        Do NOT pass ``num_chains`` for this one: it would reserve cores that
        vectorized chains never use, cutting simulation concurrency for nothing.

        Measured, 24 simulations over 24 cores, each drawing 4 chains at equal
        ESS (~1900)::

            C  workers=24  cpw=1  vectorized   17.5s   <- fastest
            B  workers=6   cpw=4  parallel     21.9s   <- 6 sims at a time
               workers=6   cpw=4  vectorized   29.1s   <- reserves cores, wastes them

        Summary:

        ==================  ===========  ==========  ============  ============
        fit's chain_method  num_chains=  workers     cores/worker  sims at once
        ==================  ===========  ==========  ============  ============
        (1 chain)           omit         n_cpu       1             n_cpu
        'parallel'          N            n_cpu // N  N             n_cpu // N
        'vectorized'        omit         n_cpu       1             n_cpu
        ==================  ===========  ==========  ============  ============

        The failure mode to know: ``chain_method='parallel'`` with only one
        device finds one device, warns once per fit, and draws the chains
        **sequentially** -- correct results, needlessly slow. That is what
        happens if you ask a fit for 4 parallel chains without telling the
        runner. Passing ``num_chains`` prevents it; passing a
        ``cores_per_worker`` below ``num_chains`` warns.

    Notes:
        In a **.py script**, put the call behind ``if __name__ == '__main__':``.
        Workers are spawned, so they re-import the calling script and would
        otherwise re-run its module level. Notebooks and the REPL need no guard.

        Each worker JIT-compiles the model on its first fit. If the model
        function is built fresh inside ``fn`` (a closure), *every* simulation
        pays full compilation because the JIT cache key changes each time.
        Defining the model at module level lets a worker reuse its compiled
        executable across all the simulations it handles -- usually the single
        largest speedup available here.

    Examples:
        >>> def one_sim(N, rho, seed, **_):            # doctest: +SKIP
        ...     m = bf(cores=1, rand_seed=seed, print_devices_found=False)
        ...     m.fit(model, obs=dgp(N, rho, m), num_chains=1, progress_bar=False)
        ...     s = m.summary()
        ...     return {"rho_hat": s.loc["rho", "mean"], "rho_true": rho}
        >>> res = run_simulations(                      # doctest: +SKIP
        ...     one_sim, grid(N=[50, 100, 500], rho=[0.0, 0.3], reps=10), workers=12)
        >>> res.groupby(["N", "rho"]).rho_hat.mean()    # doctest: +SKIP
    """
    import pandas as pd

    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    if backend not in ("process", "serial"):
        raise ValueError(f"backend must be 'process' or 'serial', got {backend!r}")
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")

    scenarios = _normalise_scenarios(scenarios)
    fn_kwargs = dict(fn_kwargs or {})
    n = len(scenarios)

    for s in scenarios:
        s.setdefault("seed", _derive_seed(seed, s["sim"]))

    # num_chains is the single knob for multi-chain fits: it reserves one core
    # per chain in every worker and is handed to fn so the fit uses the same
    # number. Asking for 4 chains on a 24-core host therefore gives each worker
    # 4 cores and runs 6 simulations at a time.
    if num_chains is not None:
        num_chains = int(num_chains)
        if num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {num_chains}")
        if cores_per_worker is None:
            cores_per_worker = num_chains
        elif cores_per_worker < num_chains:
            warnings.warn(
                f"cores_per_worker={cores_per_worker} is below num_chains="
                f"{num_chains}: chain_method='parallel' needs one device per "
                "chain, so numpyro will draw the chains sequentially. Drop "
                "cores_per_worker to let num_chains set it, or use "
                "chain_method='vectorized' in the fit.",
                stacklevel=2,
            )
        for s in scenarios:
            s.setdefault("num_chains", num_chains)

    cores_per_worker = max(1, int(cores_per_worker or 1))
    n_cpu = os.cpu_count() or 1
    if workers is None:
        workers = min(n, max(1, n_cpu // cores_per_worker))
    workers = max(1, min(int(workers), n))

    if workers * cores_per_worker > n_cpu:
        capped = max(1, n_cpu // cores_per_worker)
        warnings.warn(
            f"workers*cores_per_worker = {workers}*{cores_per_worker} = "
            f"{workers * cores_per_worker} exceeds the {n_cpu} available cores; "
            f"capping workers to {capped}. Oversubscribing makes every "
            f"simulation slower, not faster.",
            stacklevel=2,
        )
        workers = capped

    if backend == "serial" or workers == 1:
        rows = _serial(fn, scenarios, fn_kwargs, on_error, progress)
    else:
        rows = _parallel(
            fn, scenarios, fn_kwargs, workers, cores_per_worker,
            on_error, quiet, progress,
        )

    df = pd.DataFrame(rows)
    if "sim" in df.columns:
        df = df.sort_values("sim", kind="stable").reset_index(drop=True)

    if progress:
        n_failed = int(df["error"].notna().sum()) if "error" in df.columns else 0
        msg = f"run_simulations: {n} simulations done"
        if n_failed:
            msg += f", {n_failed} row(s) carry an error -- see df[df.error.notna()]"
        print(msg)
    return df


def _report(progress, done, total):
    """Progress line: rewritten in place on a terminal, thinned in a log file.

    ``\\r`` only overwrites on a TTY; redirected to a file it concatenates every
    update onto one enormous line, so there we emit occasional real lines
    instead.
    """
    if not progress:
        return

    try:
        tty = sys.stdout.isatty()
    except Exception:                        # stdout replaced (e.g. quiet worker)
        tty = False

    if tty:
        print(f"\rrun_simulations: {done}/{total} complete", end="", flush=True)
        if done == total:
            print()
        return

    step = max(1, total // 10)
    if done == total or done % step == 0:
        print(f"run_simulations: {done}/{total} complete", flush=True)


def _serial(fn, scenarios, fn_kwargs, on_error, progress):
    rows = []
    worker = os.getpid()
    for i, scenario in enumerate(scenarios, 1):
        out, error = _run_one(fn, scenario, fn_kwargs, worker)
        if error is not None and on_error == "raise":
            raise RuntimeError(
                f"run_simulations: simulation {scenario.get('sim')} failed:\n{error}"
            )
        rows.extend(out)
        _report(progress, i, len(scenarios))
    return rows


def _parallel(fn, scenarios, fn_kwargs, workers, cores_per_worker,
              on_error, quiet, progress):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    import cloudpickle

    if multiprocessing.parent_process() is not None:
        # We are inside a spawned child that re-executed its parent script's
        # module level -- i.e. the script has no __main__ guard. Stop here
        # rather than recursing into a pool of pools.
        raise RuntimeError(_GUARD_MSG)

    # cloudpickle, not pickle: model and DGP functions are routinely closures
    # over a bf instance, and spawn's default serialiser cannot carry those.
    fn_bytes = cloudpickle.dumps(fn)
    fn_kwargs_bytes = cloudpickle.dumps(fn_kwargs) if fn_kwargs else b""
    payloads = [cloudpickle.dumps(s) for s in scenarios]

    env = _worker_env(cores_per_worker, quiet=quiet)
    # spawn, not fork: setup_device must configure XLA before the backend comes
    # up, and forking a process that has already initialised JAX is unsafe.
    ctx = multiprocessing.get_context("spawn")

    # Spread every available CPU across the workers, so none sit idle. At
    # least cores_per_worker, since that many JAX devices need somewhere to
    # run.
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 1)
    block_size = max(cores_per_worker, n_cpu // workers)

    rows = []
    try:
        with _env_override(env):
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(env, fn_bytes, fn_kwargs_bytes, quiet, block_size),
            ) as pool:
                futures = {pool.submit(_worker_entry, p, i % workers): s
                           for i, (p, s) in enumerate(zip(payloads, scenarios))}
                for i, future in enumerate(as_completed(futures), 1):
                    scenario = futures[future]
                    out, error = cloudpickle.loads(future.result())
                    if error is not None and on_error == "raise":
                        # cancel_futures drops the queued work so the pool's
                        # exit only waits on what is already running.
                        pool.shutdown(wait=False, cancel_futures=True)
                        if progress:
                            print()
                        raise RuntimeError(
                            f"run_simulations: simulation {scenario.get('sim')} "
                            f"failed:\n{error}"
                        )
                    rows.extend(out)
                    _report(progress, i, len(scenarios))
    except BrokenProcessPool as exc:
        raise RuntimeError(
            f"run_simulations: a worker process died.\n\n{_GUARD_MSG}\n\n"
            "If the script already has that guard, the worker was most likely "
            "killed by the OS (out of memory) -- lower `workers`."
        ) from exc
    return rows
