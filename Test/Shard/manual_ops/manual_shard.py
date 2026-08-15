"""Manual, per-operation sharding declarations for use INSIDE a model body.

Instead of sharding the input data globally and letting GSPMD propagate (the
auto-shard path), these helpers let a model author declare sharding for ONE
specific operation while keeping everything else replicated. Two mechanisms:

1. GSPMD layout constraints — ``shard0`` / ``replicate``
   Wrap an intermediate with ``jax.lax.with_sharding_constraint`` to pin its
   layout. XLA then partitions the downstream op and **keeps the result correct
   automatically**, inserting collectives as needed. SAFE (sharding-invariant)
   but you don't control *where* communication happens, and re-laying out a
   replicated array costs a scatter (+ a gather to bring the result back).

2. Explicit ``shard_map`` — ``map_reduce_sum`` / ``outer_sum_srm``
   Run a sub-function per shard with EXPLICIT collectives (``psum``). Full
   control and no hidden communication, but YOU are responsible for correctness:
   a missing ``psum`` silently drops data (see the value check in the tests).

The point of mechanism 2 is surgical correctness: parallelize exactly the
operation you have certified as map+reduce-safe, and nothing else.
"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax import shard_map


def make_mesh(n=None, axis="device"):
    """1-D device mesh over all (or n) devices."""
    n = n or len(jax.devices())
    return Mesh(mesh_utils.create_device_mesh((n,)), axis_names=(axis,))


# ---------------------------------------------------------------------------
# Mechanism 1 — GSPMD layout constraints (safe; XLA inserts collectives)
# ---------------------------------------------------------------------------
def shard0(x, mesh, axis="device"):
    """Pin *x* sharded along axis 0 → the op that consumes it runs partitioned.

    Use to parallelize one heavy per-row operation. Correctness is guaranteed by
    GSPMD; cost is whatever collective XLA must insert (a scatter if *x* arrived
    replicated, plus a gather/all-reduce to reassemble the result).
    """
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(axis)))


def replicate(x, mesh):
    """Pin *x* replicated on every device (e.g. a small O(N) coupled vector)."""
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))


# ---------------------------------------------------------------------------
# Mechanism 2 — explicit shard_map (you write the collectives)
# ---------------------------------------------------------------------------
def map_reduce_sum(local_fn, mesh, n_sharded, axis="device"):
    """Shard the first *n_sharded* args on axis 0, apply *local_fn* per shard,
    and ``psum`` the result across shards. The explicit, no-hidden-comms form of
    a map+reduce. Remaining args are passed replicated.
    """
    def wrapped(*xs):
        sharded, replicated = xs[:n_sharded], xs[n_sharded:]
        in_specs = tuple(P(axis) for _ in sharded) + tuple(P() for _ in replicated)

        def body(*xs_shard):
            local = local_fn(*xs_shard)            # per-shard local contribution
            return jax.lax.psum(local, axis)       # explicit cross-shard reduce
        return shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=P())(*xs)
    return wrapped


def outer_sum_srm(s, r, mesh, axis="device"):
    """SRM pattern: *s* sharded on axis 0, *r* replicated → local ``(N/d, N)``
    outer sum on each device. No transpose, no all-to-all.

    Returns the full ``(N, N)`` matrix, sharded along axis 0.
    """
    def body(s_shard, r_full):
        return s_shard[:, None] + r_full[None, :]      # (N/d, N) local
    return shard_map(body, mesh=mesh,
                     in_specs=(P(axis), P()), out_specs=P(axis))(s, r)


def map_reduce_sum_BUGGY(local_fn, mesh, n_sharded, axis="device"):
    """DELIBERATELY WRONG: identical to :func:`map_reduce_sum` but with the
    ``psum`` omitted and ``out_specs=P(axis)``, then we keep only shard 0's
    partial. Demonstrates that manual sharding shifts correctness onto the
    author — the runtime value check is what catches this.
    """
    def wrapped(*xs):
        sharded, replicated = xs[:n_sharded], xs[n_sharded:]
        in_specs = tuple(P(axis) for _ in sharded) + tuple(P() for _ in replicated)

        def body(*xs_shard):
            return local_fn(*xs_shard).reshape(1)      # NO psum — partial only
        partials = shard_map(body, mesh=mesh, in_specs=in_specs,
                             out_specs=P(axis))(*xs)
        return partials[0]                              # BUG: drop other shards
    return wrapped
