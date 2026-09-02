"""Measure the 2-D mesh (chains × data-shards) on the matrix SRM.

Two things, per the plan:
  (1) No lockstep/explosion — independent chains run at a stable per-iter rate
      (vs the vectorized run that exploded to ~256 s/it at N=400).
  (2) Per-chain data-parallel speedup at 5-way → the Amdahl serial fraction,
      which extrapolates what 120 cores (30-way) would give.

Three configs at N=400 (short 150+150 to see the deep-tree regime cheaply):
  - 1 chain, UNSHARDED (1 device)        → T1, rate1
  - 1 chain, 5-way data-sharded          → T5, rate5   (speedup = rate1/rate5)
  - 4 chains × 5 data-shards (full 2-D)  → T_2d         (≈ one sharded chain)

Run:  CORES=20 python3 bench_mesh2d.py
"""
import os, sys, time
CORES = int(os.environ.get("CORES", "20"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from mesh2d_nuts import run_2d_mesh
jax.config.update("jax_enable_x64", True)

N      = int(os.environ.get("N", "400"))
WARM   = int(os.environ.get("WARM", "150"))
SAMP   = int(os.environ.get("SAMP", "150"))
NDATA  = int(os.environ.get("NDATA", "5"))     # data shards per chain
SEED   = 42
N_GA, N_GM = 1, 3
m = bf("cpu", cores=CORES, rand_seed=False, print_devices_found=True)
print(f"N={N} cores={CORES} warm={WARM} samp={SAMP} data_shards={NDATA}\n")


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

d = simulate()
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


def run(nchain, ndata, specs, label):
    devs = np.array(jax.devices()[:nchain * ndata]).reshape(nchain, ndata)
    mesh = Mesh(devs, ("chain", "data"))
    t0 = time.perf_counter()
    out = run_2d_mesh(srm, KW, mesh, specs, nchain, jax.random.PRNGKey(1),
                      WARM, SAMP, max_tree_depth=10, target_accept_prob=0.8)
    jax.block_until_ready([v for v in out.values()])
    dt = time.perf_counter() - t0
    rate = dt / (WARM + SAMP)
    print(f"{label:<34} time={dt:8.1f}s  rate={rate:6.3f} s/it")
    return dt, rate, out


if __name__ == "__main__":
    print(">>> (a) 1 chain, UNSHARDED (1 device)"); sys.stdout.flush()
    t1, r1, _ = run(1, 1, REP, "1 chain unsharded")
    print(f"\n>>> (b) 1 chain, {NDATA}-way data-sharded"); sys.stdout.flush()
    t5, r5, _ = run(1, NDATA, ROW, f"1 chain {NDATA}-way sharded")
    print(f"\n>>> (c) 4 chains x {NDATA} data-shards (full 2-D mesh)"); sys.stdout.flush()
    t2d, r2d, _ = run(4, NDATA, ROW, f"4 chains x {NDATA} (2-D mesh)")

    sp = r1 / r5 if r5 else float("nan")
    # Amdahl: sp = 1/(s + (1-s)/NDATA)  → solve s
    s_frac = (1.0/sp - 1.0/NDATA) / (1.0 - 1.0/NDATA) if sp > 0 else float("nan")
    sp30 = 1.0/(s_frac + (1-s_frac)/30) if s_frac == s_frac else float("nan")
    print("\n=== MEASUREMENTS ===")
    print(f"(1) no explosion: 2-D mesh rate {r2d:.3f} s/it (vectorized exploded to ~256 s/it)")
    print(f"(2) per-chain {NDATA}-way speedup = {sp:.2f}x  → serial fraction s≈{s_frac:.2f}")
    print(f"    Amdahl extrapolation to 30-way (120 cores): ~{sp30:.1f}x per chain")
