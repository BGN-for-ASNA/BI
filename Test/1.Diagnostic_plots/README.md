# Test/1.Diagnostic_plots: bayesplot cross-check of the PPC plots

Numerical cross-checks of the posterior predictive check plots in
`BayesForge/Diagnostic/ppc.py` (exposed as `m.diag.ppc_*`) against
R **bayesplot 1.16.0** (with **loo 2.10.1** / **rstantools 2.7.1** for
LOO-PIT, ggplot2 4.0.3 for the boxplot and histogram statistics).
`make_input.py` writes a shared fixture (`fixtures/input.json`: `y`, `yrep`,
log-likelihood, x, count data); `make_reference.R` feeds the same arrays to
the bayesplot `*_data` functions and stores the numbers; the pytest module
recomputes them in Python and compares. Quantile conventions: `np.percentile`
(linear interpolation) equals R's `stats::quantile(type = 7)`, which is what
bayesplot uses everywhere except where noted below.

## How to run

```bash
cd /home/sosa/work/BF && PYTHONPATH=$PWD /home/sosa/work/3.12venv/bin/python -m pytest -q Test/1.Diagnostic_plots -p no:cacheprovider
```

## How to regenerate the references

```bash
cd /home/sosa/work/BF
PYTHONPATH=$PWD /home/sosa/work/3.12venv/bin/python Test/1.Diagnostic_plots/make_input.py   # regenerates fixtures/input.json
Rscript Test/1.Diagnostic_plots/make_reference.R                                            # regenerates fixtures/reference.json
```

## Mapping: BayesForge function -> bayesplot function -> numbers compared

| BayesForge (`m.diag.*`) | bayesplot 1.16.0 | numbers compared |
|---|---|---|
| `ppc_density` | `ppc_dens_overlay` (`bw="nrd0"`, `adjust=1`, gaussian kernel, `n=1024` grid points, all draws) | KDE ordinates on a shared grid (BF uses Scott's rule and `n=50` sampled draws, so match bandwidth/draws first) |
| `ppc_hist` | `ppc_hist` (`binwidth`/`bins`/`breaks` NULL -> ggplot2 default 30 bins) | bin counts per dataset on shared explicit `breaks` |
| `ppc_boxplot` | `ppc_boxplot` (ggplot2 `stat_boxplot`: `quantile.type=7`, `coef=1.5`, `notch=TRUE`) | min, Q1, median, Q3, max; whisker ends = most extreme non-outlier; outlier set; notch = median +/- 1.58*IQR/sqrt(n) |
| `ppc_stat` | `ppc_stat` / `ppc_stat_data` (`stat="mean"`, name resolved by `match.fun`) | T(y) and T(yrep) per draw; BF's p-value annotation (`mean(T_rep >= T_y)`, two-sided) has no bayesplot counterpart |
| `ppc_stat_2d` | `ppc_stat_2d` (`stat=c("mean","sd")`) | the two statistic vectors over draws + observed point |
| `ppc_intervals` | `ppc_intervals_data` (`prob=0.5`, `prob_outer=0.9`; `quantile(type=7)` at `(1-prob_outer)/2`, `(1-prob)/2`, 0.5, `(1+prob)/2`, `(1+prob_outer)/2`; inner/outer probs are swapped if given in the wrong order) | `ll, l, m, h, hh` per observation |
| `ppc_ribbon` | `ppc_ribbon_data` (same `.ppd_intervals_data` as intervals) | same interval bounds (bayesplot's plot then loess-smooths; BF does not) |
| `ppc_intervals_sorted` | `ppc_intervals_data` on rows sorted by `y` (reference: `prob=prob_outer=0.9`) | interval bounds in sorted order; bayesplot's `ppc_loo_intervals` is a different thing (true LOO via PSIS, `order="median"`) |
| `ppc_loo_intervals` | deprecated alias of `ppc_intervals_sorted` | same as above (no LOO is computed) |
| `ppc_error_scatter` | `ppc_error_scatter_avg` (`stat="mean"`) | x = `colMeans(yrep)`, y = `y` |
| `ppc_error_hist` | `ppc_error_hist` (errors `y - yrep[s,i]` per draw via `rstantools::predictive_error`, faceted per draw) | `colMeans(errors)` vs BF's `y - E[yrep]`; BF's single-histogram counts are not bayesplot's per-draw facets |
| `ppc_scatter` | `ppc_scatter` (all draws, `ref_line=TRUE`) | point coordinates `(yrep[s,i], y_i)` for matched draws (BF samples 9 draws with seed 0) |
| `ppc_rootogram` | `ppc_rootogram` (`style="hanging"`, `prob=0.9`) | `xpos = 0:max(y,yrep)`, `tyexp = sqrt(colMeans(counts))`, `ty = sqrt(counts_obs)`, hanging tile base/top; band `sqrt(quantile(counts, .05/.95))` (type 7) not drawn by BF |
| `ppc_bars` | `ppc_bars_data` (`prob=0.9`, `freq=TRUE`; `l/m/h = quantile(counts, .05/.5/.95)` type 7; categories = union of `y`/`yrep` values, zero-filled) | observed counts, 5%/95% count bounds; bayesplot's point is the **median**, BF's is the **mean** |
| `ppc_loo_pit` | `ppc_loo_pit_data` (PIT via `rstantools::loo_pit(yrep, y, lw)`, `lw` from `loo::psis_loo(...)$psis_object`; density with `boundary_correction=TRUE`, `bw="nrd0"`, `grid_len=512`, `samples=100`) | PIT values before smoothing (BF computes PSIS itself, ArviZ-style); both sides reflect the density at 0 and 1 (BF: Scott bandwidth, reflection) |

## Known differences / xfail

(Placeholder - fill in as the tests are written. Candidates spotted while
reading the sources: KDE bandwidth rule (Scott vs nrd0) in `ppc_density` and
`ppc_loo_pit`; `ppc_bars` mean vs median point; `ppc_error_hist` mean errors
vs per-draw errors; LOO-PIT tie handling (`<=` in BF vs `<` plus uniform
randomization inside ties in `rstantools::loo_pit`) and ArviZ-style vs
R `loo` PSIS smoothing; bayesplot's boundary-corrected PIT density.)
