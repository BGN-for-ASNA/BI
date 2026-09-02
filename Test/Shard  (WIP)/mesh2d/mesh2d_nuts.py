"""2-D mesh NUTS driver: parallel chains × data sharding, built from numpyro's
PUBLIC API (no numpyro modification).

Idea
----
numpyro's MCMC offers only `parallel` (pmap whole chains, no within-chain data
sharding) or `vectorized` (vmap chains → tree-depth LOCKSTEP). To get *both*
independent chains (no lockstep) AND data sharding, we drive the chains ourselves:

  mesh = Mesh(devices.reshape(n_chains, n_data), ('chain', 'data'))

  * a single-chain NUTS loop = kernel.init + lax.scan(kernel.sample)  (jittable)
  * shard_map over the 'chain' axis  → each chain is an independent SPMD instance
    running its OWN tree-depth while-loop (no lockstep)
  * data arrays sharded over the 'data' axis → each chain's leapfrog is data-parallel

The single-chain loop mirrors numpyro's own single-chain orchestration (init tells
the kernel the warmup length; sample adapts during warmup, samples after), so the
samples are genuine NUTS.
"""
import jax
import jax.numpy as jnp
from jax import lax
from numpyro.infer import NUTS


def single_chain_loop(kernel, rng_key, num_warmup, num_samples,
                      model_kwargs, init_params=None):
    """One NUTS chain as a jittable scan. Returns post-warmup samples (dict)."""
    state = kernel.init(rng_key, num_warmup, init_params=init_params,
                        model_args=(), model_kwargs=model_kwargs)

    def step(state, _):
        state = kernel.sample(state, (), model_kwargs)
        return state, state.z          # state.z = UNCONSTRAINED params dict

    _, zs = lax.scan(step, state, None, length=num_warmup + num_samples)
    zs = {k: v[num_warmup:] for k, v in zs.items()}        # post-warmup, unconstrained
    # transform to constrained space (+ deterministics), exactly as numpyro does
    postprocess = kernel.postprocess_fn((), model_kwargs)
    return jax.vmap(postprocess)(zs)


def run_single_chain(model, model_kwargs, rng_key, num_warmup, num_samples,
                     init_params=None, max_tree_depth=10, target_accept_prob=0.8):
    """Convenience: build the kernel and run one chain (jitted)."""
    kernel = NUTS(model, max_tree_depth=max_tree_depth,
                  target_accept_prob=target_accept_prob)
    f = jax.jit(lambda rk, mk: single_chain_loop(kernel, rk, num_warmup,
                                                 num_samples, mk, init_params))
    return f(rng_key, model_kwargs)


def run_2d_mesh(model, model_kwargs, mesh, data_specs, n_chains, rng_key,
                num_warmup, num_samples, max_tree_depth=10, target_accept_prob=0.8,
                chain_axis="chain", data_axis="data"):
    """Run *n_chains* INDEPENDENT chains (no lockstep) with each chain's data
    sharded over the *data* mesh axis.

    shard_map is made manual over the *chain* axis only (axis_names={chain_axis}),
    so each chain is a separate SPMD instance running its own tree-depth loop,
    while the *data* axis stays auto (GSPMD) and shards each leapfrog.

    Args:
        mesh: 2-D Mesh with axes (chain_axis, data_axis).
        data_specs: dict {name: PartitionSpec} for each model_kwarg (data sharded
            over data_axis; everything else replicated).
        rng_key: split into n_chains keys, mapped over the chain axis.
    Returns:
        dict of samples with a leading (n_chains, num_samples, ...) shape.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    kernel = NUTS(model, max_tree_depth=max_tree_depth,
                  target_accept_prob=target_accept_prob)
    keys = jax.random.split(rng_key, n_chains)

    def body(rk, mk):
        # rk arrives as a per-chain shard of shape (1, 2) → take the single key
        return single_chain_loop(kernel, rk[0], num_warmup, num_samples, mk)

    # in_specs may only name the MANUAL axis (chain). Data is replicated over the
    # chain axis (every chain sees all data); its data-axis sharding is carried
    # by the committed array sharding below and handled by GSPMD inside the body.
    in_specs = (P(chain_axis), {k: P() for k in model_kwargs})
    out_specs = P(chain_axis)
    fn = jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                       axis_names={chain_axis}, check_vma=False)

    # commit shardings: keys over chain axis; data over data axis (or replicated)
    keys = jax.device_put(keys, NamedSharding(mesh, P(chain_axis)))
    mk = {k: jax.device_put(v, NamedSharding(mesh, data_specs.get(k, P())))
          for k, v in model_kwargs.items()}
    return jax.jit(fn)(keys, mk)
