"""run_simulations agrees across Python, R and Julia.

The R (BayesForge, ``m$run_simulations``) and Julia (PyBayesForge.jl,
``m.run_simulations``) runners schedule their own worker processes but take
the scenarios, the seeds and the core budget from Python's ``_plan``. So the
same grid must give the same ``sim``, ``rep``, scenario values and ``seed``
in all three languages, and the same deterministic results. Python's serial
run is the reference; R and Julia use 2 real worker processes.

Skipped when ``Rscript`` / ``julia`` or the corresponding package (with
``run_simulations``) is not installed -- the R and Julia CI jobs install them.
``BF_JULIA`` and ``BF_JULIA_PROJECT`` select the Julia binary and project, as
for Test/0.Documentation.
"""

import csv
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from BayesForge.Parallel.simulate import grid, run_simulations

pytestmark = pytest.mark.slow

COLS = ["sim", "rep", "N", "b", "seed", "twice"]


def _reference():
    """Python, serial: one row per scenario, sorted by sim."""
    def fn(N, b, seed, **_):
        return {"twice": 2 * N}

    df = run_simulations(fn, grid(N=[1, 2, 3], b=[-0.5, 0.5], reps=2),
                         backend="serial", progress=False)
    return [{c: str(row[c]) for c in COLS} for _, row in df.iterrows()]


def _norm(rows):
    """Compare as numbers: 2 and 2.0 are the same value in every language."""
    return [{c: float(r[c]) for c in COLS} for r in rows]


def _env():
    root = pathlib.Path(__file__).resolve().parents[2]
    return dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg",
                PYTHONPATH=os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])))


def _check(verify, lang, out_csv, reference):
    with open(out_csv, newline="") as f:
        got = list(csv.DictReader(f))
    mismatches = sum(a != b for a, b in zip(_norm(got), _norm(reference)))
    mismatches += abs(len(got) - len(reference))
    verify.record(quantity=f"{lang}: rows differing from Python (sim, rep, N, b, seed, result)",
                  reference="Python run_simulations, serial", criterion="== 0",
                  value=mismatches, passed=mismatches == 0)
    assert len(got) == len(reference)
    assert _norm(got) == _norm(reference)


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------

R_SCRIPT = r"""
suppressMessages(library(BayesForge))
if (!exists(".bf_run_simulations", envir = asNamespace("BayesForge"))) quit(status = 3)
m <- importBF(print_devices_found = FALSE)
f <- function(N, b, seed) list(twice = 2 * N)
res <- m$run_simulations(f, m$grid(N = c(1, 2, 3), b = c(-0.5, 0.5), reps = 2),
                         workers = 2, progress = FALSE)
stopifnot(length(unique(res$worker)) == 2, all(is.na(res$error)))
write.csv(res[c("sim", "rep", "N", "b", "seed", "twice")],
          commandArgs(TRUE)[1], row.names = FALSE)
"""


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not found")
def test_r_matches_python(tmp_path, verify):
    script, out = tmp_path / "sim.R", tmp_path / "r.csv"
    script.write_text(R_SCRIPT)
    proc = subprocess.run(["Rscript", str(script), str(out)], env=_env(),
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == 3 or "there is no package called" in proc.stderr:
        pytest.skip("R package BayesForge with run_simulations not installed")
    assert proc.returncode == 0, proc.stderr[-3000:]
    _check(verify, "R", out, _reference())


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------

JL_SCRIPT = r"""
using PyBayesForge
isdefined(PyBayesForge, :bf_run_simulations) || exit(3)
m = importBF(print_devices_found = false)
@sim double(x) = 2x
@sim function f(; N, b, seed)
    (twice = double(N),)
end
res = m.run_simulations(f, m.grid(N = [1, 2, 3], b = [-0.5, 0.5], reps = 2);
                        workers = 2, progress = false)
@assert length(unique(r.worker for r in res)) == 2
@assert all(ismissing(r.error) for r in res)
open(ARGS[1], "w") do io
    println(io, "sim,rep,N,b,seed,twice")
    for r in res
        println(io, join((r.sim, r.rep, r.N, r.b, r.seed, r.twice), ","))
    end
end
"""


def _find_julia():
    if os.environ.get("BF_JULIA"):
        return os.environ["BF_JULIA"]
    juliaup = pathlib.Path.home() / ".juliaup" / "bin" / "julia"
    return shutil.which("julia") or (str(juliaup) if juliaup.exists() else None)


@pytest.mark.skipif(_find_julia() is None, reason="julia not found")
def test_julia_matches_python(tmp_path, verify):
    script, out = tmp_path / "sim.jl", tmp_path / "jl.csv"
    script.write_text(JL_SCRIPT)
    env = _env()
    if "JULIA_PYTHONCALL_EXE" not in env:
        env.update(JULIA_PYTHONCALL_EXE=sys.executable, JULIA_CONDAPKG_BACKEND="Null")
    cmd = [_find_julia()]
    if os.environ.get("BF_JULIA_PROJECT"):
        cmd.append(f"--project={os.environ['BF_JULIA_PROJECT']}")
    proc = subprocess.run(cmd + [str(script), str(out)], env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode == 3 or "Package PyBayesForge not found" in proc.stderr:
        pytest.skip("PyBayesForge.jl with run_simulations not installed")
    assert proc.returncode == 0, proc.stderr[-3000:]
    _check(verify, "Julia", out, _reference())
