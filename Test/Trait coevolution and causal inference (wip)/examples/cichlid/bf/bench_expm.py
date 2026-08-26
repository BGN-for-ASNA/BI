"""Break segment_quantities down, and test a batched expm replacement.

jax.scipy.linalg.expm carries scaling-and-squaring control flow that vmap turns
into per-segment work. Every segment shares the same A and differs only in dt,
so a fixed-order scaling-and-squaring Taylor series -- no data-dependent
branching -- vectorises cleanly over segments.
"""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
HERE = "/home/sebastian_sosa/phylo/examples/cichlid"
sys.path.insert(0, HERE + "/bf")
from gdpm_core import ksolve, build_A
from cichlid_bf import load_data

d = load_data()
J = d["J"]
A = build_A(jnp.array([-0.5, -0.5, -0.5]), jnp.array([-2.0, 3.0, -2.0, 1.5]),
            d["off_rows"], d["off_cols"])
Q_inf = ksolve(A, jnp.diag(jnp.array([2.0, 2.0, 2.0])))
ts = jnp.where(d["ts"] > 0, d["ts"], 0.0)
I = jnp.eye(J)


def bench(name, f, *a):
    g = jax.jit(f); jax.block_until_ready(g(*a))
    t = time.perf_counter()
    for _ in range(20):
        r = g(*a)
    jax.block_until_ready(r)
    print(f"{name:38s} {(time.perf_counter()-t)/20*1e3:8.3f} ms")


bench("vmap expm only", lambda A, ts: jax.vmap(lambda dt: expm(A * dt))(ts), A, ts)
Ad_ref = jax.vmap(lambda dt: expm(A * dt))(ts)
bench("batched cholesky only",
      lambda Ad: jnp.linalg.cholesky(Q_inf - Ad @ Q_inf @ Ad.transpose(0, 2, 1)
                                     + 1e-10 * I), Ad_ref)


def expm_batch(A, dts, order=8, squarings=10):
    """e^{A dt} for many dt, sharing one A. Fixed cost, no branching."""
    B = A[None] * (dts[:, None, None] / (2.0 ** squarings))
    E = jnp.broadcast_to(I, B.shape)
    term = E
    for k in range(1, order + 1):          # Taylor, Horner-free but order is tiny
        term = term @ B / k
        E = E + term
    for _ in range(squarings):
        E = E @ E
    return E


bench("expm_batch (taylor 8, 10 squarings)", expm_batch, A, ts)
Ad_new = expm_batch(A, ts)
print("expm_batch max abs err vs jax expm :", float(jnp.abs(Ad_new - Ad_ref).max()))


def seg_new(A, Q_inf, dts):
    Ad = expm_batch(A, dts)
    VCV = Q_inf - Ad @ Q_inf @ Ad.transpose(0, 2, 1)
    VCV = 0.5 * (VCV + VCV.transpose(0, 2, 1))
    L = jnp.linalg.cholesky(VCV + 1e-10 * I)
    n = dts.shape[0]
    rhs = (Ad - I).transpose(1, 0, 2).reshape(J, n * J)
    As = jnp.linalg.solve(A, rhs).reshape(J, n, J).transpose(1, 0, 2)
    As = 0.5 * (As + As.transpose(0, 2, 1))
    return Ad, L, As


bench("seg_new (full replacement)", seg_new, A, Q_inf, ts)
bench("seg_new grad",
      jax.grad(lambda A, Q_inf, dts: sum(x.sum() for x in seg_new(A, Q_inf, dts))),
      A, Q_inf, ts)

from gdpm_core import segment_quantities
Ad_o, L_o, As_o = segment_quantities(A, Q_inf, d["ts"])
Ad_n, L_n, As_n = seg_new(A, Q_inf, ts)
print("A_delta diff:", float(jnp.abs(Ad_n - Ad_o).max()),
      " L diff:", float(jnp.abs(L_n - L_o).max()),
      " A_solve diff:", float(jnp.abs(As_n - As_o).max()))
