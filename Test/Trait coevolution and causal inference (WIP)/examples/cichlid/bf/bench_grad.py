"""Forward vs reverse cost, piece by piece."""
import sys, time
import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_enable_x64", True)
HERE = "/home/sebastian_sosa/phylo/examples/cichlid"
sys.path.insert(0, HERE + "/bf")
from gdpm_core import ksolve, build_A, segment_quantities, traverse
from cichlid_bf import load_data

d = load_data()
J, N_seg = d["J"], d["N_seg"]
A = build_A(jnp.array([-0.5, -0.5, -0.5]), jnp.array([-2.0, 3.0, -2.0, 1.5]),
            d["off_rows"], d["off_cols"])
Q_inf = ksolve(A, jnp.diag(jnp.array([2.0, 2.0, 2.0])))
Ad, L, As = segment_quantities(A, Q_inf, d["ts"])
b = jnp.zeros(J); ea = jnp.zeros(J)
z = jnp.array(np.random.default_rng(0).normal(size=(N_seg - 1, J)))


def bench(name, f, *a):
    g = jax.jit(f); jax.block_until_ready(g(*a))
    t = time.perf_counter()
    for _ in range(10):
        r = g(*a)
    jax.block_until_ready(r)
    print(f"{name:44s} {(time.perf_counter()-t)/10*1e3:8.2f} ms")


seg = lambda A: sum(x.sum() for x in segment_quantities(A, Q_inf, d["ts"]))
bench("segment_quantities  fwd", seg, A)
bench("segment_quantities  grad", jax.grad(seg), A)

trav = lambda ea, z, b: traverse(ea, z, Ad, L, As, b, d["node_seq"],
                                 d["parent"], d["tip"]).sum()
bench("traverse            fwd", trav, ea, z, b)
bench("traverse            grad", jax.grad(trav, argnums=(0, 1, 2)), ea, z, b)

# traverse differentiated wrt the per-segment matrices too, as in the real model
trav2 = lambda Ad, L, As, ea, z, b: traverse(ea, z, Ad, L, As, b, d["node_seq"],
                                             d["parent"], d["tip"]).sum()
bench("traverse (+Ad,L,As) grad",
      jax.grad(trav2, argnums=(0, 1, 2, 3, 4, 5)), Ad, L, As, ea, z, b)
