"""End-to-end cost of one log-density gradient for the translated model."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
from numpyro.infer.util import initialize_model

jax.config.update("jax_enable_x64", True)
HERE = "/home/sebastian_sosa/phylo/examples/cichlid"
sys.path.insert(0, HERE + "/bf")
from cichlid_bf import load_data, make_model

d = load_data()
model = make_model(d["N_seg"], d["N_tips"], d["J"])
obs = {k: d[k] for k in ("y", "node_seq", "parent", "ts", "tip", "tip_id",
                         "off_rows", "off_cols", "level_seg", "level_valid")}

init = initialize_model(jax.random.PRNGKey(0), model, model_kwargs=obs)
params = init.param_info.z
pot = init.potential_fn

print("parameters:", {k: v.shape for k, v in params.items()})
print("logp at init:", float(pot(params)))

g = jax.jit(jax.grad(pot))
jax.block_until_ready(g(params))
t = time.perf_counter()
for _ in range(50):
    r = g(params)
jax.block_until_ready(r)
ms = (time.perf_counter() - t) / 50 * 1e3
print(f"grad of potential: {ms:.2f} ms")
print(f"~{ms * 500 / 1000:.1f} s per NUTS iteration at 500 leapfrog steps")
