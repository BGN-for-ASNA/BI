"""How much wall-time does the shard-check (and its new backward piece) add?
It runs ONCE at fit setup, never in the MCMC loop. Time it on the real SRM."""
import os, sys, time
CORES = int(os.environ.get("CORES", "8"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
sys.path.insert(0, "/home/sebastian_sosa/BF")
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from BayesForge import bf
from BayesForge.Network.model_effects2 import NeteffectMatrix
from BayesForge.Diagnostic import sharding_safeguards as sg
jax.config.update("jax_enable_x64", True)

NDEV = int(os.environ.get("NDEV", "4"))
m = bf("cpu", cores=CORES, rand_seed=False, print_devices_found=False)
mesh = Mesh(np.array(jax.devices()[:NDEV]), ("data",))
N_GA, N_GM = 1, 3

def build(N):
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
    Anyf=jnp.array(Any,jnp.int32); Merf=jnp.array(Mer,jnp.int32)
    na=jnp.array([N]); nmj=jnp.array(nm); mask=(1-jnp.eye(N)); recv=jnp.array(wt)
    EID,DIR = NeteffectMatrix.dyad_index_maps(N)
    def srm(Y,dmat,sender,Any_s,Mer_s,msk,eid,dir_):
        Ba=NeteffectMatrix.block_model(Anyf,N_GA,na,name="intercept",group_row=Any_s)
        Bm=NeteffectMatrix.block_model(Merf,N_GM,nmj,name="Merica",group_row=Mer_s)
        SR=m.net2.sender_receiver(sender,recv)
        Dd=NeteffectMatrix.dyadic_effect(dmat,eid=eid,dir_idx=dir_)
        m.dist.bernoulli(logits=(Ba+Bm+SR+Dd)*msk,obs=Y,name="Y")
    def put(a,sp): return jax.device_put(jnp.asarray(a),NamedSharding(mesh,sp))
    KW=dict(Y=jnp.asarray(Y),dmat=jnp.asarray(dmat),sender=jnp.array(wf),
            Any_s=Anyf,Mer_s=Merf,msk=mask,eid=EID,dir_=DIR)
    shard={k:put(v,P("data")) for k,v in KW.items()}
    rep={k:put(v,P()) for k,v in KW.items()}
    return srm, shard, rep

print(f"{'N':>5} {'full check':>12} {'backward only':>14} {'(of which)':>10}")
for N in [60, 160, 320]:
    srm, shard, rep = build(N)
    key = jax.random.PRNGKey(0)
    # isolate the new backward piece
    from numpyro.infer.util import initialize_model
    info = initialize_model(key, srm, model_kwargs=shard, dynamic_args=False)
    params = info[0].z
    t0=time.perf_counter(); sg.backward_collective_check(srm, params, shard); tb=time.perf_counter()-t0
    # full check (includes the backward piece + forward + runtime)
    t0=time.perf_counter(); sg.run_shard_check(srm, shard, rep, key, name=""); tf=time.perf_counter()-t0
    print(f"{N:>5} {tf:>11.2f}s {tb:>13.2f}s {100*tb/tf:>8.0f}%")
