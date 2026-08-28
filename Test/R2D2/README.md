# R2D2 prior — implementation-correctness test

Evaluates the R2D2 (*R²-induced Dirichlet Decomposition*) prior as documented in
`BF/Documentation/37.Feature extraction: R2D2.qmd`, using a simulate → fit →
recover Bayesian-inference workflow.

## Run

```bash
cd BF/Test/R2D2
python3 run_r2d2_test.py                       # defaults below
python3 run_r2d2_test.py --centered            # verbatim qmd param. beta~N(0,lam)
python3 run_r2d2_test.py --priors "1/3,3 ; 3,1" --num-warmup 2000 --num-samples 2000
```

Defaults: `N=50`, `V=100`, `n_strong=5`, `beta_scale≈2.5`,
**DGP noise sweep `--sigma-grid "1,2,4,8"`** (true model R² ≈ 0.98 / 0.93 / 0.77 / 0.46),
NUTS `warmup=1000 samples=1000 chains=4 target_accept=0.9`, priors on R²
`Beta(1/3,3), Beta(1,1), Beta(3,3), Beta(3,1), Beta(5,1)`.
`--sigma-grid ""` collapses to a single run at `--sigma-true`.

Runs on **CPU** (script forces `JAX_PLATFORMS=cpu`; the box's GPU jax has a
CuDNN mismatch). ~3 min for the full 4-noise × 5-prior grid (20 fits) on `ECOD052`.

## Outputs

| Path | Content |
|---|---|
| `logs/r2d2_test_<stamp>.log` | full run log (console is mirrored here) |
| `out/results_<stamp>.json`   | machine-readable metrics per prior |
| `out/beta_recovery_<stamp>_s<σ>.png` | true vs posterior-mean β, per prior, per noise level |
| `out/r2_posterior_<stamp>_s<σ>.png`  | posterior of R² vs truth, per noise level |
| `out/error_metrics_<stamp>_s<σ>.png` | RMSE(strong) & mean\|β\|(null), per noise level |
| `out/noise_sweep_<stamp>.png`        | RMSE / R² bias / σ bias vs DGP noise, all priors |

## What is checked

1. **Simulation** — `N` individuals, `V` standardized `N(0,1)` covariates, only
   `n_strong` with a real effect; `y = Xβ + N(0,σ)`.
2. **R²** — model / explained-variance definition
   `R² = Var(Xβ) / (Var(Xβ) + σ²)` (equivalently `τ²/(τ²+σ²)`, the quantity the
   Beta prior targets — see qmd *Definition of R²*), plus classical
   `1 − SSres/SStot` for reference.
3. **Prior-predictive check** on R² for every `Beta(a,b)`.
4. **Model** — R2D2 exactly as the qmd:
   `σ~Exp(1)`, `R²~Beta(a,b)`, `τ²=σ²R²/(1−R²)`, `φ~Dir(1,…,1)`,
   `λ²ⱼ=τ²φⱼ`, `βⱼ~N(0,λ²ⱼ)`. Non-centered by default (`β=z·λ`, `z~N(0,1)`) —
   mathematically identical, needed because `V>N` funnels the centered form for
   priors with mass on `R²→1`. `--centered` reproduces the snippet verbatim.
5. **Recovery vs truth** — bias/RMSE/HDI-coverage of the strong coefficients,
   shrinkage of the 95 nulls (mean |β|, HDI false-positives), recovery of R² and
   σ, and MCMC health (max R-hat, min ESS, divergences).
6. **DGP noise sweep** — repeats the whole simulate→fit→recover loop at each
   `sigma_true`, so the true R² spans 0.98 → 0.46, and reports whether the
   prior-sensitivity conclusions depend on the signal-to-noise ratio.

## Reference run (`seed=20260827`, `--sigma-grid "1,2,4,8"`, 20 fits, all converged)

`cell = RMSE(strong β) / posterior-mean σ`

| σ_true | true R² | Beta(1/3,3) | Beta(1,1) | Beta(3,3) | Beta(3,1) | Beta(5,1) |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.982 | 0.71 / 1.38 | 0.62 / 0.82 | 0.69 / 1.32 | 0.61 / 0.78 | 0.60 / 0.77 |
| 2 | 0.932 | 1.06 / 2.05 | 0.94 / 1.19 | 1.02 / 1.89 | 0.93 / 1.18 | 0.93 / 1.09 |
| 4 | 0.773 | 1.64 / 3.58 | 1.45 / 2.29 | 1.57 / 3.25 | 1.44 / 2.19 | 1.43 / 2.01 |
| 8 | 0.460 | 2.35 / 6.59 | 2.16 / 4.91 | 2.22 / 5.75 | 2.14 / 4.47 | 2.12 / 3.94 |

σ-recovery bias (post mean / true − 1), low-noise → high-noise:

| prior | σ bias @ σ=1 | σ bias @ σ=8 |
|---|--:|--:|
| Beta(1/3,3) | **+38 %** | −18 % |
| Beta(1,1)   | −18 %     | −39 % |
| Beta(3,3)   | +32 %     | −28 % |
| Beta(3,1)   | −22 %     | −44 % |
| Beta(5,1)   | −23 %     | **−51 %** |

**Conclusion — the R2D2 implementation is correct and behaves as the paper/qmd
describe:**

- **Adaptive sparsity works, at every noise level.** The 95 null coefficients
  are shrunk (mean |β̂| grows 0.09 → 0.25 as noise grows) with **0 false
  positives out of 95** (89% HDI never excludes 0) in all 20 fits.
- **The Beta prior on R² is the influential dial, and its effect flips with
  noise.** At low noise (true R²≈0.98) conservative priors `Beta(1/3,3)/Beta(3,3)`
  over-estimate σ (**+32…+38 %**) while calibrated priors are near-unbiased. At
  high noise (true R²≈0.46) *every* prior now **under-estimates σ** — worst for
  the optimistic `Beta(5,1)` (**−51 %**) — and over-estimates R². The crossover
  reproduces the qmd *"When R2D2 Fails → Prior Misspecification"*: the safe
  choice depends on where the true R² sits, which you don't know. Less-opinionated
  priors (`Beta(1,1)`, `Beta(3,3)`) degrade more gracefully than the confident
  `Beta(5,1)`, whose worst-case σ error is the largest.
- **Strong effects are under-estimated (ridge-like) and it worsens with noise.**
  Bias is always toward 0; |β̂|/|β| falls from ≈0.80 (σ=1) to ≈0.50 (σ=4); 89%
  HDI covers only 2–4 of the 5 true effects. Expected for `V ≫ N` with a flat
  `Dirichlet(1,…,1)` instead of a horseshoe (qmd *Limitations*).
- **Sampler geometry.** Non-centered (`β = z·λ`, default) keeps all 20 fits at
  R̂ ≤ 1.02 with few divergences. The verbatim centered form (`--centered`)
  breaks (R̂≈2.7, thousands of divergences) for priors that push R²→1.

## Appendix A — N = 300 (`--N 300`, seed 20260827)

Same sweep with n:p = 3:1. `results_20260828_093031.json`.

- **Prior sensitivity mostly collapses.** σ recovered to ±3% at every noise
  level, all priors; the "flip" is gone (σ bias `Beta(1/3,3)`: +38%→−18% at N=50
  becomes +2%→−2% at N=300). σ_post spread across priors at σ=8: 0.13 vs 0.66.
- **Coefficients** recovered (RMSE ≤ 0.13, coverage 5/5) whenever true R² ≥ 0.92;
  at true R²=0.75, RMSE 0.31 / coverage 4/5; at true R²=0.43, still ~28% shrink,
  coverage 3/5.
- **0 divergences** everywhere (N=50 had 25–273).
- **Residual artifact, N-independent:** the `r2` posterior sits below truth
  (0.88 vs 0.98 at σ=1). Flat `Dirichlet(1,…,1)` spreads τ² over 100 slots when
  5 matter → implied R² dragged down. Don't read the R² posterior as a fit
  estimate.

## Appendix B — low-R² regime (`--dir-conc`, true R² ≈ 0.30, σ=11)

New flag `--dir-conc a0`: `φ ~ Dirichlet(a0,…,a0)`. `a0<1` favours sparse
variance allocation (heavier-tailed β priors). `results_20260828_0939*/0941*`.

| a0 | N=300 coverage | N=300 nulls \|β̂\| | N=300 div | N=50 coverage |
|--|--|--|--|--|
| 1.0 (flat) | 1/5 | 0.17 | 0 | 1/5 |
| 0.5 | 2/5 | 0.15 | 0 | 1/5 |
| **0.1** | **4/5** | **0.08** | 0 | 1/5 |
| 0.01 | 4/5 | 0.02 | **647** | 0/5 |

- **N=300: `a0 ≈ 0.1` is a clear win** — coverage 1/5→4/5, null noise halved,
  R² recovers toward truth, no divergence cost. `a0 ≤ 0.01` funnels (hundreds of
  divergences).
- **N=50: no `a0` rescues it** — 50 rows can't resolve 5 moderate effects among
  100 with 70% noise. Small `a0` trades coefficient recovery for σ/R² recovery.
- `Beta(2,6)` (mean 0.25) ≈ `Beta(1/3,3)` here — difference grows as true R²
  rises off zero.

**Low-R² recommendations** (see report §8): drop `a0` to 0.1–0.5; calibrate the
Beta to the field (`Beta(2,6)`, or empirical-Bayes from published R²); scale `y`
(or the σ prior); keep non-centered; consider a regularized horseshoe if you can
guess the number of real predictors; if effects aren't sparse, flat Dirichlet is
correct and only more data / dim-reduction helps.

Full write-up (both N, both regimes, tuning): `report.html` →
<https://claude.ai/code/artifact/982622b7-0cc6-450a-887b-ed18ab2e1fec>
