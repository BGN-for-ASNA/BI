"""Direct head-to-head the README was missing: the FULL 4-chain mesh.

Question (user): with parallel + sharding you should get 20/4 = 5 cores per
chain, data sharded across each chain's 5 dedicated cores. Does that beat giving
each chain a single core?

  config A  4 x 1  : 4 independent chains, 1 core each (4 cores used, 16 idle)
                     — the numpyro chain_method='parallel' analog
  config B  4 x 5  : 4 independent chains, each data-sharded across 5 dedicated
                     cores (20 cores busy) — the full 2-D mesh you describe

Both run the SAME 4-chain workload; the only difference is whether each chain's
extra cores are put to work via sharding. Wall-time with compile removed
(build jitted fn once, compile via first call, time a cached second call).

Run:  PYTHONPATH=/home/sebastian_sosa/BF CORES=20 N=400 python3 bench_4chain.py
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

N      = int(os.environ.get("N", "400"))
TREE   = int(os.environ.get("TREE", "5"))
ITERS  = int(os.environ.get("ITERS", "20"))
NCHAIN = int(os.environ.get("NCHAIN", "4"))
NDATA  = int(os.environ.get("NDATA", "5"))
SEED   = 42
N_GA, N_GM = 1, 3
m = bf("cpu", cores=CORES, rand_seed=False, print_devices_found=True)
print(f"N={N} cores={CORES} tree_cap={TREE} iters={ITERS} chains={NCHAIN}\n")
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


def build_and_time(nchain, ndata, specs, label):
    """nchain independent chains, each ndata-way sharded. Compile removed:
    first call compiles, timed second call is pure wall-time for the full
    nchain workload run concurrently."""
    devs = np.array(jax.devices()[:nchain * ndata]).reshape(nchain, ndata)
    mesh = Mesh(devs, ("chain", "data"))
    kernel = NUTS(srm, max_tree_depth=TREE, target_accept_prob=0.8)
    keys = jax.random.split(jax.random.PRNGKey(1), nchain)

    def body(rk, mk):
        return single_chain_loop(kernel, rk[0], 1, ITERS, mk)

    in_specs = (P("chain"), {k: P() for k in KW})
    fn = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=P("chain"),
                       axis_names={"chain"}, check_vma=False)
    keys_c = jax.device_put(keys, NamedSharding(mesh, P("chain")))
    mk = {k: jax.device_put(v, NamedSharding(mesh, specs.get(k, P())))
          for k, v in KW.items()}
    jfn = jax.jit(fn)

    out = jfn(keys_c, mk); jax.block_until_ready([v for v in out.values()])
    t0 = time.perf_counter()
    out = jfn(keys_c, mk); jax.block_until_ready([v for v in out.values()])
    t_exec = time.perf_counter() - t0
    rate = t_exec / ITERS
    print(f"{label:<34} wall_exec={t_exec:7.2f}s  rate={rate:6.3f} s/it"); sys.stdout.flush()
    return rate


if __name__ == "__main__":
    print(f">>> A: {NCHAIN} chains x 1 core (16 cores idle)")
    rA = build_and_time(NCHAIN, 1, REP, f"A: {NCHAIN}x1 (1 core/chain)")
    print(f"\n>>> B: {NCHAIN} chains x {NDATA} sharded cores (all {NCHAIN*NDATA} busy)")
    rB = build_and_time(NCHAIN, NDATA, ROW, f"B: {NCHAIN}x{NDATA} (sharded)")
    print("\n=== 4-CHAIN WALL-TIME HEAD-TO-HEAD ===")
    print(f"A  {NCHAIN}x1 unsharded : {rA:.3f} s/it")
    print(f"B  {NCHAIN}x{NDATA} sharded   : {rB:.3f} s/it")
    print(f"\nsharding the dedicated cores => {rA/rB:.2f}x "
          f"({'FASTER' if rA/rB>1 else 'SLOWER'} than 1 core/chain)")
