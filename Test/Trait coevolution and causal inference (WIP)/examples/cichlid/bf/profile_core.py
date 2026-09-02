"""Where does the gradient time go: the expm vmap or the sequential traversal?"""
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
Q = jnp.diag(jnp.array([2.0, 2.0, 2.0]))
b = jnp.zeros(J); eta_anc = jnp.zeros(J)
z = jnp.array(np.random.default_rng(0).normal(size=(N_seg - 1, J)))


def bench(name, f, *a):
    g = jax.jit(f)
    r = g(*a); jax.block_until_ready(r)
    t = time.perf_counter()
    for _ in range(20):
        r = g(*a)
    jax.block_until_ready(r)
    print(f"{name:32s} {(time.perf_counter()-t)/20*1e3:8.2f} ms")


bench("ksolve", lambda A, Q: ksolve(A, Q), A, Q)
bench("segment_quantities (529 expm)",
      lambda A, Q, ts: segment_quantities(A, ksolve(A, Q), ts), A, Q, d["ts"])

Ad, L, As = segment_quantities(A, ksolve(A, Q), d["ts"])
bench("traverse (528-step scan)",
      lambda ea, z, b: traverse(ea, z, Ad, L, As, b, d["node_seq"], d["parent"], d["tip"]),
      eta_anc, z, b)


def full(A_off, z, b, ea):
    A = build_A(jnp.array([-0.5, -0.5, -0.5]), A_off, d["off_rows"], d["off_cols"])
    Ad, L, As = segment_quantities(A, ksolve(A, Q), d["ts"])
    eta = traverse(ea, z, Ad, L, As, b, d["node_seq"], d["parent"], d["tip"])
    return eta.sum()


bench("full forward", full, jnp.array([-2.0, 3.0, -2.0, 1.5]), z, b, eta_anc)
bench("full grad", jax.grad(full, argnums=(0, 1, 2, 3)),
      jnp.array([-2.0, 3.0, -2.0, 1.5]), z, b, eta_anc)

# how deep is the tree? that is the floor for a depth-parallel traversal
parent = np.array(d["parent"]); node_seq = np.array(d["node_seq"])
depth = np.zeros(N_seg, dtype=int)
for i in range(1, N_seg):
    depth[node_seq[i]] = depth[parent[i]] + 1
seg_depth = np.array([depth[node_seq[i]] for i in range(N_seg)])
print("\ntree depth (levels):", seg_depth.max() + 1)
w = np.bincount(seg_depth[1:])
print("max nodes per level:", w.max(), " levels:", len(w))
