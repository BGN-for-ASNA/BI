"""Does data sharding help the SRM at LARGE N (>1K nodes)? And does using
more cores per chain (higher shard count) ever win?

This probe answers two questions the 2-D mesh README left open:

  Q1. "Use more cores per chain" — you do NOT need >2-D mesh for that. A 3-D
      mesh (chain, row, col) block-shards the (N,N) matrix, but the SRM's
      bottleneck is the all-reduce of the REPLICATED per-dyad random-effect
      gradient (size O(N^2)), which no amount of data-axis splitting removes.
      So "more cores per chain" reduces to "higher data-shard count" — which we
      sweep directly here on one chain: ndata in {1,5,10,20}.

  Q2. "Advantageous only for larger models?" — the compute-to-communication
      ratio decides. Per-leapfrog COMPUTE is O(N^2) (the N×N likelihood); the
      replicated-gradient all-reduce COMMUNICATION is also O(N^2). Same order →
      the ratio is ~constant in N → bigger N should NOT unlock a speedup, only
      amortize fixed dispatch overhead toward break-even. We test that by
      measuring the same sweep at two N values.

Clean measurement
------------------
- max_tree_depth is CAPPED (default 5) so every iteration is a fixed, bounded
  number of leapfrogs → uniform, cheap iterations. The sharded/unsharded ratio
  of per-leapfrog compute is GSPMD-invariant, so capping does not bias the ratio.
- Compile cost is REMOVED: we build the jitted shard_map fn once, call it once to
  compile+warm, then TIME a second cached call. s/it = T_second / n_iters.

Run:  CORES=20 N=1000 python3 bench_bigN.py
      CORES=20 N=400  python3 bench_bigN.py     # trend comparison point
"""
import os, sys, time
CORES = int(os.environ.get("CORES", "20"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from numpyro.infer import NUTS
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from mesh2d_nuts import single_chain_loop
jax.config.update("jax_enable_x64", True)

N      = int(os.environ.get("N", "1000"))
TREE   = int(os.environ.get("TREE", "5"))          # capped tree depth
ITERS  = int(os.environ.get("ITERS", "20"))        # timed iterations (post-compile)
SWEEP  = [int(x) for x in os.environ.get("SWEEP", "1,5,10,20").split(",")]
SEED   = 42
N_GA, N_GM = 1, 3
m = bf("cpu", cores=CORES, rand_seed=False, print_devices_found=True)
print(f"N={N} cores={CORES} tree_cap={TREE} timed_iters={ITERS} sweep={SWEEP}\n")
sys.stdout.flush()


def simulate():
    np.random.seed(SEED)
    mt = bf("cpu", rand_seed=False, print_devices_found=False)
    wf = np.random.normal(0,1,(N,4)); wt = np.random.normal(0,1,(N,4))
    dy = np.random.binomial(1,0.3,(N,N,3))
    for k in range(3): np.fill_diagonal(dy[:,:,k],0)
    dy[:,:,2]=dy[:,:,0]*dy[:,:,1]
    dedg=jnp.stack([mt.net.mat_to_edgl(dy[:,:,k]) for k in range(3)],axis=2)
    Any=np.zeros(N,int); Mer=np.random.choice([0,1,2],N,p=[.25,.5625,.1875])
    nm=np.array([np.sum(Mer==i) for i in range(3)])
    Bi=mt.net.block_model(Any,1,jnp.array([N]),sample=True,name="intercept")
    Bc=mt.net.block_model(Mer,3,jnp.array(nm),sample=True,name="category")
    sr=mt.net.sender_receiver(jnp.array(wf),jnp.array(wt),s_mu=.4,r_mu=-.4,sample=True)
    dr=mt.net.dyadic_effect(dedg,d_sd=2.5,sample=True)
    net=jnp.array(mt.dist.bernoulli(logits=Bi+Bc+sr+dr,sample=True))
    Any=jnp.array(Any,jnp.int32); Mer=jnp.array(Mer,jnp.int32)
    _,na=jnp.unique(Any,return_counts=True); _,nmc=jnp.unique(Mer,return_counts=True)
    return dict(Y=NeteffectMatrix.edgelist_to_matrix_outcome(net,N),
                dmat=NeteffectMatrix.edgelist_to_matrix_predictors(dedg,N),
                wf=jnp.array(wf), wt=jnp.array(wt), Any=Any, Mer=Mer, na=na, nm=nmc)

print("simulating..."); sys.stdout.flush()
t_sim = time.perf_counter()
d = simulate()
print(f"  simulated in {time.perf_counter()-t_sim:.1f}s\n"); sys.stdout.flush()
mask = (1-jnp.eye(N)).astype(jnp.float64)
EID, DIR = NeteffectMatrix.dyad_index_maps(N)
recv, Anyf, Merf, na, nm = d["wt"], d["Any"], d["Mer"], d["na"], d["nm"]


def srm(Y, dmat, sender, Any_s, Mer_s, msk, eid, dir_):
    Ba = NeteffectMatrix.block_model(Anyf, N_GA, na, name="intercept", group_row=Any_s)
    Bm = NeteffectMatrix.block_model(Merf, N_GM, nm, name="Merica", group_row=Mer_s)
    SR = m.net2.sender_receiver(sender, recv)
    Dd = NeteffectMatrix.dyadic_effect(dmat, eid=eid, dir_idx=dir_)
    m.dist.bernoulli(logits=(Ba+Bm+SR+Dd)*msk, obs=Y, name="Y")

KW = dict(Y=d["Y"], dmat=d["dmat"], sender=d["wf"], Any_s=Anyf, Mer_s=Merf,
          msk=mask, eid=EID, dir_=DIR)
ROW = {"Y":P("data"),"dmat":P("data"),"sender":P("data"),"Any_s":P("data"),
       "Mer_s":P("data"),"msk":P("data"),"eid":P("data"),"dir_":P("data")}
REP = {k:P() for k in KW}


def build_and_time(ndata, specs, label):
    """One chain, ndata-way data sharding. Build jitted fn once, compile via a
    first call, then TIME a second cached call → pure steady-state s/it."""
    if N % ndata != 0:
        print(f"{label:<22} SKIP (N={N} not divisible by {ndata})"); return None
    devs = np.array(jax.devices()[:ndata]).reshape(1, ndata)
    mesh = Mesh(devs, ("chain", "data"))
    kernel = NUTS(srm, max_tree_depth=TREE, target_accept_prob=0.8)
    keys = jax.random.split(jax.random.PRNGKey(1), 1)

    def body(rk, mk):
        return single_chain_loop(kernel, rk[0], 1, ITERS, mk)   # 1 warmup + ITERS

    in_specs = (P("chain"), {k: P() for k in KW})
    fn = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=P("chain"),
                       axis_names={"chain"}, check_vma=False)
    keys_c = jax.device_put(keys, NamedSharding(mesh, P("chain")))
    mk = {k: jax.device_put(v, NamedSharding(mesh, specs.get(k, P())))
          for k, v in KW.items()}
    jfn = jax.jit(fn)

    t0 = time.perf_counter()
    out = jfn(keys_c, mk); jax.block_until_ready([v for v in out.values()])
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jfn(keys_c, mk); jax.block_until_ready([v for v in out.values()])
    t_exec = time.perf_counter() - t0
    rate = t_exec / ITERS
    print(f"{label:<22} compile+run={t_compile:7.1f}s  exec={t_exec:7.1f}s  "
          f"steady={rate:6.3f} s/it"); sys.stdout.flush()
    return rate


if __name__ == "__main__":
    print(f">>> single-chain shard sweep at N={N} (tree cap {TREE})\n")
    base = build_and_time(1, REP, "ndata=1 (unsharded)")
    rows = [(1, base)]
    for nd in SWEEP:
        if nd == 1: continue
        r = build_and_time(nd, ROW, f"ndata={nd}")
        if r is not None: rows.append((nd, r))

    print("\n=== STEADY-STATE s/it (compile removed) ===")
    print(f"{'ndata':>6} {'s/it':>9} {'speedup vs 1':>14}")
    for nd, r in rows:
        sp = base / r if (r and base) else float("nan")
        print(f"{nd:>6} {r:>9.3f} {sp:>13.2f}x")
    if len(rows) > 1:
        best_nd, best_r = min(rows, key=lambda x: x[1])
        print(f"\nbest = ndata={best_nd} at {base/best_r:.2f}x vs unsharded")
