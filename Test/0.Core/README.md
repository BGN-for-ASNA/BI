# Test/0.Core: tests of the core BayesForge functions

Unit, regression and verification tests of the Python package, including the
features of Tiers 1 and 2 of the roadmap after MEE-26-09-855
(`BF_MS/Reply to reviewers/feature_roadmap.md`). Every verification test
compares a feature against a reference that does not depend on it (exact
posterior, NUTS, the implementation it replaces).

```bash
python -m pytest Test/0.Core -q                 # everything (~2 min)
python -m pytest Test/0.Core -q -m "not slow"   # skip the minute-long statistical checks
python -m pytest Test/0.Core/test_enumeration.py -s
```

`conftest.py` collects the checks recorded through the `verify` fixture,
prints them as a table and writes `Test/0.Core/_report/` (`summary.md`,
`checks.csv`, `comparisons.png`, `z_scores.png`).

## Roadmap features and their tests

| # | Feature | Test | Where else |
|---|---|---|---|
| 1.1 | R `loo` object, `loo::loo(m)` | — | BFR `tests/testthat/test-loo.R` |
| 1.2 | bayesplot ready, `as.array(m)` | — | BFR `tests/testthat/test-as-array.R` |
| 1.3 | ArviZ.jl, `InferenceData(m)` | — | PyBayesForge.jl `test/runtests.jl` (not committed yet) |
| 1.4 | No hidden fixed seeds | `test_rng.py` | |
| 1.5 | Save/load: explicit errors | `test_save_load.py` | |
| 1.6 | `enum=True` on discrete sites | `test_enumeration.py` | BFR `test-enum-samplers.R`, PyBayesForge.jl `test/runtests.jl` |
| 1.7 | MixedHMC / DiscreteHMCGibbs / HMCGibbs via `fit(sampler=...)` | `test_discrete_samplers.py` | BFR `test-enum-samplers.R` |
| 2.1 | Automatic parameter names in R, `bf_model()` | — | BFR `test-bf-model.R` |
| 2.2 | R math functions on JAX arrays | — | BFR `test-jax-methods.R` |
| 2.3 | Tidy data in and out (`m.index()` on categorical columns) | `test_index_categorical.py`, `test_data_input.py` | BFR `test-tidy.R`, PyBayesForge.jl `test/runtests.jl` |
| 2.4 | One compatibility matrix | `test_compat.py` (+ `tools/check_compat.py` in CI) | CI of BFR and PyBayesForge.jl |
| 2.5 | Doc examples are tests | — | `Test/3.Documentation/` |
| 2.6 | Docstrings as single source of the R help | `test_docstring_examples.py` | BFR CI (`tools/build_r_docs.py --check`) |
| 2.7 | Observed data found by tracing | `test_obs_args_tracing.py` | |
| — | Observed sites named after the `obs` variable | `test_obs_names.py` | BFR `test-obs-names.R` |
| — | `run_simulations` in R (`m$run_simulations`) and Julia (`m.run_simulations`): same grid, seeds and results as Python | `test_run_simulations_languages.py` (slow; skipped without `Rscript` / `julia`) | BFR `test-simulate.R`, PyBayesForge.jl `test/runtests.jl`, `Test/13.Parallel/` |

Other regression tests: `test_diag_api.py` (every `m.diag` method is
documented and runs), `test_diagnostics_ic.py` (LOO/WAIC/compare with
ArviZ 1.x), `test_effects.py` (`m.effects`).
