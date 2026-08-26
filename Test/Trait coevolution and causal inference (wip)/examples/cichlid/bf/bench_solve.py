"""Isolate the A_solve = A^-1 (A_delta - I) computation."""
import sys, time
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
HERE = "/home/sebastian_sosa/phylo/examples/cichlid"
sys.path.insert(0, HERE + "/bf")
from gdpm_core import ksolve, build_A
from cichlid_bf import load_data

d = load_data()
J = d["J"]
A = build_A(jnp.array([-0.5, -0.5, -0.5]), jnp.array([-2.0, 3.0, -2.0, 1.5]),
            d["off_rows"], d["off_cols"])
ts = jnp.where(d["ts"] > 0, d["ts"], 0.0)
I = jnp.eye(J)
n = ts.shape[0]


def expm_batch(A, dts, order=8, squarings=10):
    B = A[None] * (dts[:, None, None] / (2.0 ** squarings))
    E = jnp.broadcast_to(I, B.shape); term = E
    for k in range(1, order + 1):
        term = term @ B / k
        E = E + term
    for _ in range(squarings):
        E = E @ E
    return E


M = expm_batch(A, ts) - I


def bench(name, f, *a):
    g = jax.jit(f); jax.block_until_ready(g(*a))
    t = time.perf_counter()
    for _ in range(20):
        r = g(*a)
    jax.block_until_ready(r)
    print(f"{name:40s} {(time.perf_counter()-t)/20*1e3:8.3f} ms")
    return g(*a)


r1 = bench("solve, stacked columns",
           lambda A, M: jnp.linalg.solve(A, M.transpose(1, 0, 2).reshape(J, n * J))
           .reshape(J, n, J).transpose(1, 0, 2), A, M)
r2 = bench("inv then batched matmul", lambda A, M: jnp.linalg.inv(A) @ M, A, M)
r3 = bench("solve with batched RHS (vmap)",
           lambda A, M: jax.vmap(lambda X: jnp.linalg.solve(A, X))(M), A, M)
r4 = bench("inv then einsum", lambda A, M: jnp.einsum("ij,njk->nik",
                                                      jnp.linalg.inv(A), M), A, M)
for nm, r in [("inv/matmul", r2), ("vmap solve", r3), ("einsum", r4)]:
    print(f"  {nm:12s} max diff vs stacked solve: {float(jnp.abs(r - r1).max()):.3e}")

bench("inv+matmul grad",
      jax.grad(lambda A, M: (jnp.linalg.inv(A) @ M).sum()), A, M)
