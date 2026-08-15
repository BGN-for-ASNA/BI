# SRM sharding/speed optimization log

Goal: get the matrix-SRM (N=400, 4 chains, 1000+1000) under the **no-shard baseline
of 9621 s** (Min ESS 24, Mean ESS 8672) **without degrading estimation**.

Method. Every config is judged on two axes:

- **Speed** — wall time (and, for short probes, steady s/it).
- **Estimation** — Min/Mean ESS, max R̂, divergences, and **posterior agreement**
  vs a NUTS-alone reference (max standardized difference of the structural
  parameter means: `max |Δmean| / sd_ref`; < 0.25 ≈ within MC error → MATCH).

Correctness is validated at a tractable **N=100** (where a NUTS-alone reference is
cheap and converged); speed effects that are N-specific (the warmup tree-depth
explosion) are measured at **N=400**.

Harness: `opt_experiment.py` (driven by env vars; appends to `opt_results.csv`).

---

## Key finding so far (context)

- Data sharding requires `chain_method='vectorized'`, whose **lockstep tree depth**
  made the N=400 run explode to ~256 s/it during warmup (→ ~140 h projected),
  while the no-shard `parallel` baseline averaged 4.8 s/it. Per-iteration cost is
  **tree-depth-bound**, which sharding does not reduce.
- The baseline's **Min ESS = 24 / 4000** signals poor geometry — the real
  bottleneck. Optimizations below target warmup cost and geometry, not just
  parallelism.

## Experiments

| ID | config | N | warm+samp | time (s) | s/it | MinESS | MeanESS | maxR̂ | div | agree (maxΔ/sd) | verdict |
|----|--------|---|-----------|----------|------|--------|---------|-------|-----|------------------|---------|
| ref100 | no-shard, parallel, maxtree 10 (NUTS-alone) | 100 | 600+600 | 186.0 | 0.155 | 88.5 | 4783.5 | 1.045 | 0 | reference | ground truth |
| A_shard_cap | **shard** (matrix,vectorized), maxtree (7,10), tacc 0.75 | 100 | 600+600 | 2837.1 | 2.364 | 49.9 | 4167.1 | 1.074 | 0 | 26.9 | ✗ 15× SLOWER than ref (warmup cap can't fix the sampling-phase funnel + vectorized lockstep) |
| matrix_noshard | matrix likelihood, **parallel** (no shard), maxtree 10 | 100 | 600+600 | 120.3 | 0.100 | 78.3 | 3296.3 | 1.067 | 0 | 13.4 | ✓ 1.55× FASTER than edgelist ref, exact-equivalent |
| matrix_check | matrix, parallel, maxtree 10 | 100 | 300+300 | 64.6 | 0.108 | 28.9 | 2133.7 | 1.149 | 0 | 23.7 | same model as matrix_noshard; diverges from ref by a *different* amount → divergence is non-convergence/weak-ID, not a bug |

### Correctness resolved (deterministic proof)

`log_density(matrix) − log_density(edgelist) = N·log(0.5)` **exactly**, for every
parameter draw (`/tmp/eqcheck.py`). The matrix/sharded formulation is therefore
**mathematically identical** to the edgelist model (same posterior up to a
constant). The large "agree" numbers above are the model's **weak identification**
(collinear sender/receiver/dyad/block effects → unstable per-effect means
run-to-run, R̂>1.1) — the same instability afflicts the edgelist model. Raw-effect
agreement was the wrong metric; log-density/predictions are the identified
quantities, and on those the formulations agree exactly.

### Key opportunity

The **matrix likelihood is ~1.5× faster per iteration than the edgelist** (it skips
the per-leapfrog `mat_to_edgl` gather over N_dyads, whose cost grows with N) and is
**provably the exact same posterior**. Using it with `chain_method='parallel'`
(no vectorized lockstep, no sharding) is the exact-preserving path to beat 9621s.
Next: full N=400 run below.

### ★ RESULT — N=400, exact-equivalent, beats baseline

| ID | config | N | warm+samp | time (s) | s/it | MinESS | MeanESS | maxR̂ | div | vs 9621s |
|----|--------|---|-----------|----------|------|--------|---------|-------|-----|----------|
| **matrix_N400** | matrix likelihood, **parallel** (no shard), maxtree 10, tacc 0.8 | 400 | 1000+1000 | **4985.4** | 2.493 | **167.4** | 9965.1 | 1.028 | 0 | **1.93× faster ✓** |

**Same config as the 9621 s baseline (cores=20, parallel, 4 chains, maxtree 10,
tacc 0.8) — the ONLY change is the matrix-form likelihood instead of the
edgelist `mat_to_edgl` per-leapfrog conversion.** Provably the same posterior
(see equivalence proof). Estimation is equal-or-better: Min ESS 167 vs 24, R̂
1.028, 0 divergences.

## Conclusions

1. **Data sharding loses on CPU for this SRM** (15× slower at N=100; ~140 h
   projected at N=400) — the run is tree-depth-bound (the dyadic funnel forces
   deep trees in *sampling*) and `vectorized` chains amplify it in lockstep.
   Sharding is correct (proven) and is the right tool on GPU / well-mixing
   map+reduce models, not here.
2. **The matrix-form likelihood is the real CPU win**: 1.93× faster than the
   edgelist baseline, exact-equivalent, with `chain_method='parallel'`. This is
   the recommended default for the matrix SRM.
3. **Remaining exact-preserving lever**: SVI-init to shorten warmup (stackable,
   validated against the edgelist posterior). The model's weak identification
   (Min ESS, deep trees) is the deeper bottleneck; fixing it (low-rank dyadic)
   would change the estimand, so it's out of scope under "edgelist = correct".
