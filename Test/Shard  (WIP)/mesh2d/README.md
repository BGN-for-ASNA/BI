# 2-D mesh prototype: parallel chains × data sharding

A NUTS driver that runs **independent chains** (no `vectorized` tree-depth
lockstep) while **sharding each chain's data** — the configuration numpyro's
`chain_method` cannot express. Built entirely from numpyro's **public API**
(`NUTS` kernel + `initialize_model`); numpyro itself is **not modified**.

## Why

For the matrix SRM on CPU, plain data sharding lost badly because it forces
`chain_method='vectorized'`, whose lockstep makes every iteration cost the
deepest chain's tree depth → explosion (~256 s/it at N=400). Independent chains
(`parallel`) avoid that but can't shard data within a chain. The 2-D mesh gets
both:

```
mesh = Mesh(devices.reshape(n_chains, n_data), ('chain', 'data'))
```

- **`chain` axis** — made *manual* in `shard_map(..., axis_names={'chain'})`, so
  each chain is a separate SPMD instance running its **own** tree-depth
  `while_loop` (no lockstep).
- **`data` axis** — left *auto* (GSPMD); each chain's leapfrog is data-parallel.

## Files

- `mesh2d_nuts.py`
  - `single_chain_loop` — one NUTS chain as `kernel.init` + `lax.scan(kernel.sample)`,
    then `postprocess_fn` to return constrained samples. Mirrors numpyro's own
    single-chain orchestration.
  - `run_2d_mesh` — maps the single-chain loop over the `chain` axis via
    `shard_map`, with data committed to the `data`-axis sharding.
- `bench_mesh2d.py` — measures, on the matrix SRM, three configs (1 chain
  unsharded; 1 chain N-way sharded; 4 chains × N data-shards) → speed + the
  Amdahl serial fraction for extrapolating to more cores.

## Validation

- **Correctness** (regression vs numpyro single-chain): pooled posterior means
  recover the truth (`b=[1.47,-0.71]` vs truth `[1.5,-0.7]`; `s=0.28` vs `~0.3`),
  and `single_chain_loop` matches numpyro MCMC to ~1e-3.
- **No lockstep/explosion** (N=40): the 2-D mesh runs at a stable per-iter rate
  (0.305 s/it), unlike the vectorized run that exploded.

## Findings

| N | 1 chain unsharded | 1 chain 5-way | per-chain 5-way speedup |
|---|-------------------|---------------|--------------------------|
| 40  | 0.103 s/it | 0.346 s/it | 0.30× (overhead-dominated) |
| 400 | 14.134 s/it | 14.640 s/it | **0.97× (break-even, slightly slower)** |

(N=400 used 100+100 short warmup, which inflates the *absolute* rate — adaptation
is incomplete — but the *ratio* is valid: sharded vs unsharded run the identical
GSPMD-invariant trajectory, so only per-leapfrog compute differs.)

## Conclusion

**The 2-D mesh succeeds at what it was built for and fails at what we hoped.**

1. **No lockstep / no explosion (✓).** Independent chains run at a stable rate
   (N=40: 0.305 s/it; N=400 single sharded chain: 14.6 s/it) — never the ~256 s/it
   explosion of `vectorized`. The mechanism works.
2. **Per-chain data sharding does NOT accelerate the SRM (✗).** Even at N=400,
   5-way sharding is 0.97× — *break-even, slightly slower*. Amdahl on 0.97×
   implies **no parallelizable surplus** (serial fraction ≈ 1), so extrapolating
   to 30-way (120 cores) gives ~0.96× — **more cores would not help, and would
   slightly hurt.**

**Why:** the SRM's per-leapfrog cost is dominated by (a) the **replicated** per-dyad
random effects (159,600 params; their gradient all-reduce scales as O(N²), so the
collective grows in lockstep with the sharded compute and never becomes
relatively cheap), and (b) the **tree-depth-bound** trajectories (the dyadic
funnel), which sharding cannot shorten. Data parallelism addresses neither.

**Implication for hardware:** a 120-CPU machine would *not* speed up this SRM via
sharding. The per-dyad random effects — exactly the individual-specificity the SRM
exists to estimate — are inherently replicated parameters whose gradient is the
bottleneck. The viable speedups remain: the **matrix likelihood** (1.93×, exact,
already adopted) and **GPU** (faster per-leapfrog hardware, which accelerates the
replicated work too). Sharding is not the lever for this model.

## Follow-up: more cores per chain, and large N (>1K nodes)

Two open questions: (Q1) would a >2-D mesh let us use more cores *per chain*?
(Q2) is sharding advantageous only for much larger networks? Probe:
`bench_bigN.py` — one chain, capped tree depth (uniform cheap iterations),
**compile cost removed** (build the jitted shard_map once, compile via a first
call, time a second cached call → pure steady-state s/it). It sweeps the
data-shard count directly.

**Q1 — a 3-D mesh buys nothing here.** A `(chain, row, col)` mesh block-shards the
N×N likelihood across more cores per chain, but the bottleneck is the **all-reduce
of the replicated per-dyad random-effect gradient** (≈N²/2 params: 159,600 at
N=400, ~500,000 at N=1000). That collective is O(N²) *however* you slice the data
axis — 1-D rows, 2-D blocks, or 3-D. So "more cores per chain" == "more data
shards", which we sweep. More shards split the same O(N²) collective into more
pieces with more rounds → strictly worse, not better.

**Q2 — large N does NOT flip the verdict.** Steady-state s/it (compile removed),
`speedup = unsharded / sharded`:

| ndata | N=400 s/it | speedup | N=1000 s/it | speedup |
|------:|-----------:|--------:|------------:|--------:|
| 1 (unsharded) | 0.035 | 1.00× | 0.158 | 1.00× |
| 5  | 0.045 | 0.77× | 0.194 | **0.81×** |
| 10 | 0.050 | 0.69× | 0.327 | 0.48× |
| 20 | 0.100 | 0.35× | 0.642 | 0.25× |

Best at both N is **ndata=1 (unsharded)**; every shard count >1 is slower, and the
penalty grows with shard count. Going N=400→1000 nudged the *gentlest* (5-way)
case from 0.77×→0.81× — overhead amortizes slightly — but it stayed **below
break-even**, never crossing 1.0×. This is exactly the **ratio argument**:
per-leapfrog compute is O(N²) and the replicated-gradient all-reduce is *also*
O(N²), so comm/compute-per-shard ≈ O(n_shards), **constant in N**. Bigger N only
amortizes fixed dispatch overhead (asymptoting at break-even); it never makes the
collective relatively cheap. Sharding would win only for a model whose per-leapfrog
compute outscales communication — e.g. an **O(N³)** dense solve against O(N²)
comm (ratio O(n/N) → vanishes as N grows). The SRM is O(N²); it is not that model
at any N.

### Full 4-chain head-to-head (`bench_4chain.py`)

The config a user would actually deploy: 20 cores, 4 chains → 5 cores/chain with
data sharded across each chain's dedicated cores (`Mesh(reshape(4,5))`). Compared
against giving each chain 1 core (4 used, 16 idle — the `chain_method='parallel'`
analog). Wall-time, compile removed:

| config | N=400 | N=1000 |
|---|---|---|
| A: 4 chains × 1 core (16 idle) | 0.031 s/it | 0.152 s/it |
| B: 4 chains × 5 sharded cores (all 20 busy) | 0.078 s/it | 0.620 s/it |
| **sharding the dedicated cores** | **0.40× (2.5× slower)** | **0.24× (4× slower)** |

The full mesh is *worse* than the isolated 1-chain×5-shard probe (0.40× vs 0.77× at
N=400) because the **4 chains' O(N²) all-reduces run concurrently and contend for
shared CPU memory bandwidth** — and that contention grows with N (0.40×→0.24×),
the opposite of the gentle overhead-amortization seen per-chain in isolation. So
not only does dedicating 5 sharded cores per chain fail to help, it actively hurts
*more* the larger the network. The 16 "idle" cores in config A are not wasted in
any recoverable way: the work is embarrassingly parallel across chains and
sequential (tree-depth-bound) within a chain, with no profitably-parallel
per-leapfrog surplus for the extra cores to capture.

### Sharding by DYAD instead of by row (`bench_dyadshard.py`)

All the above sharded by **row (sender)**, which forces the O(N²) dyadic random
effect to be replicated and its gradient O(N²)-all-reduced. The communication-
optimal alternative is to shard by **dyad**: lay the whole model out in edgelist
space, pin `dr_raw (2, N_dyads)` sharded on the dyad axis (`with_sharding_constraint`),
keep the O(N) nodal effects replicated and **gather** them into each shard's dyad
block. Then the *expensive* gradient is local and only the *cheap* O(N) nodal
gradient is all-reduced — communication drops O(N²) → **O(N)**, ratio O(n/N).

Steady-state s/it, single chain, compile removed:

| ndata | row-sharded N=1000 | **dyad-sharded N=400** | **dyad-sharded N=1000** |
|------:|-------------------:|-----------------------:|------------------------:|
| 1  | 1.00× | 1.00× | 1.00× |
| 5  | 0.81× | 0.80× | **0.90×** |
| 10 | 0.48× | 0.77× | 0.65× |
| 20 | 0.25× | 0.60× | 0.36× |

Dyad-sharding is **strictly better than row-sharding** at every shard count and
its 5-way case **improves with N** (0.80×→0.90×) — direct evidence the O(N)-comm
reformulation works as predicted. But it still **caps at ~0.90× (break-even)** on
this CPU host, and a control pins down why:

**The leapfrog is memory-bandwidth-bound, not compute-bound.** Unsharded N=1000
runs at 0.244 s/it pinned to **1 physical core** (`taskset -c 0`) vs 0.136 s/it on
**all cores** — only **1.8×** from the whole machine. The work streams large O(N²)
arrays through cheap arithmetic, so all cores contend for one shared memory bus and
saturate at ~1.8×. The unsharded baseline already extracts that via XLA intra-op
threads; sharding adds coordination on top of the *same* saturated bus and lands
below it. So on a **single CPU host, unsharded wins regardless of sharding axis** —
not because of communication (dyad-sharding fixed that) but because there is no
spare memory bandwidth for any parallelization scheme to capture.

**Where dyad-sharding is the right design:** genuine multi-device hardware
(multi-GPU with separate HBM, multi-host/TPU) where each shard streams its
O(N²/n) data through its **own** dedicated memory and the only cross-device traffic
is the O(N) nodal all-reduce. There the bandwidth ceiling lifts (per-device memory)
and the low communication volume lets it scale — whereas row-sharding's O(N²)
all-reduce would saturate the interconnect. Dyad-sharding is the **correct
multi-device formulation**; a single shared-memory CPU host is just the wrong place
to see its benefit. (Multi-device payoff is the cost-model prediction, not measured
here — no multi-GPU box available.)
