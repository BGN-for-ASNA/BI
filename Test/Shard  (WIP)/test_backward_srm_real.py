"""Confirm the backward-collective safeguard fires on the REAL matrix SRM."""
import os, sys
CORES = int(os.environ.get("CORES", "8"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
sys.path.insert(0, "/home/sebastian_sosa/BF")

import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from BayesForge.Diagnostic.sharding_safeguards import run_shard_check
jax.config.update("jax_enable_x64", True)

NDEV = int(os.environ.get("NDEV", "4")); N = int(os.environ.get("N", "60"))
mesh = Mesh(np.array(jax.devices()[:NDEV]), ("data",))
m = bf("cpu", cores=CORES, rand_seed=False, print_devices_found=False)
N_GA, N_GM = 1, 3
np.random.seed(0); mt = bf("cpu", rand_seed=False, print_devices_found=False)
wf = np.random.normal(0,1,(N,4)); wt = np.random.normal(0,1,(N,4))
dy = np.random.binomial(1,0.3,(N,N,3))
for k in range(3): np.fill_diagonal(dy[:,:,k],0)
dedg = jnp.stack([mt.net.mat_to_edgl(dy[:,:,k]) for k in range(3)],axis=2)
Mer = np.random.choice([0,1,2],N,p=[.25,.5625,.1875]); Any=np.zeros(N,int)
nm = np.array([np.sum(Mer==i) for i in range(3)])
net = np.random.binomial(1,0.3,(N,N)); np.fill_diagonal(net,0)
Y = NeteffectMatrix.edgelist_to_matrix_outcome(mt.net.mat_to_edgl(net), N)
dmat = NeteffectMatrix.edgelist_to_matrix_predictors(dedg, N)
Anyf = jnp.array(Any,jnp.int32); Merf = jnp.array(Mer,jnp.int32)
na = jnp.array([N]); nmj = jnp.array(nm); mask = (1-jnp.eye(N))
EID, DIR = NeteffectMatrix.dyad_index_maps(N)
recv = jnp.array(wt)

def srm(Y, dmat, sender, Any_s, Mer_s, msk, eid, dir_):
    Ba = NeteffectMatrix.block_model(Anyf, N_GA, na, name="intercept", group_row=Any_s)
    Bm = NeteffectMatrix.block_model(Merf, N_GM, nmj, name="Merica", group_row=Mer_s)
    SR = m.net2.sender_receiver(sender, recv)
    Dd = NeteffectMatrix.dyadic_effect(dmat, eid=eid, dir_idx=dir_)
    m.dist.bernoulli(logits=(Ba+Bm+SR+Dd)*msk, obs=Y, name="Y")

def put(a, sp): return jax.device_put(jnp.asarray(a), NamedSharding(mesh, sp))
KW = dict(Y=jnp.asarray(Y), dmat=jnp.asarray(dmat), sender=jnp.array(wf),
          Any_s=Anyf, Mer_s=Merf, msk=mask, eid=EID, dir_=DIR)
ROW = {k: P("data") for k in KW}
shard = {k: put(v, ROW[k]) for k,v in KW.items()}
rep   = {k: put(v, P())    for k,v in KW.items()}

print(f"REAL matrix SRM: N={N} N_dyads={N*(N-1)//2} n_shards={NDEV}")
r = run_shard_check(srm, shard, rep, jax.random.PRNGKey(0), name="real_srm")
b = r["backward"]
print(f"\nperf_unsafe={r['perf_unsafe']} coll_elems={b['coll_elems']} "
      f"ratio={b['ratio']:.3f} dominant={b['dominant']}")
print("RESULT:", "PASS ✓ (SRM correctly flagged)" if r["perf_unsafe"] else "FAIL ✗")
sys.exit(0 if r["perf_unsafe"] else 1)
