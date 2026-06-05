"""NumPyro NUTS GPLVM — reference backend (mirrors Pyro GPLVM structure).

Uses raw NumPyro (no BI wrapper) so posteriors are directly accessible.
Identical model to gplvm_bi.py — only difference is no BI wrapping.
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", False)


def run_pyro_gplvm(Y_np, X_prior_mean_np, num_samples=500, warmup_steps=300, seed=0):
    N, D = Y_np.shape
    L = X_prior_mean_np.shape[1]

    Y_jax   = jnp.array(Y_np, dtype=jnp.float32)
    Xpm_jax = jnp.array(X_prior_mean_np, dtype=jnp.float32)

    def model():
        log_ls    = numpyro.sample("log_lengthscale", dist.Normal(jnp.zeros(L), jnp.ones(L)))
        log_var   = numpyro.sample("log_variance",    dist.Normal(0.0, 1.0))
        log_noise = numpyro.sample("log_noise",       dist.Normal(-2.0, 0.5))

        ls    = jnp.exp(log_ls)
        var   = jnp.exp(log_var)
        noise = jnp.exp(log_noise)

        X_flat = numpyro.sample(
            "X_flat",
            dist.Normal(Xpm_jax.reshape(-1), jnp.ones(N * L)),
        )
        X = X_flat.reshape(N, L)

        diff    = X[:, None, :] - X[None, :, :]
        sq_dist = jnp.sum((diff / ls) ** 2, axis=-1)
        K = var * jnp.exp(-0.5 * sq_dist) + (noise + 1e-6) * jnp.eye(N)

        for d in range(D):
            numpyro.sample(
                f"y_{d}",
                dist.MultivariateNormal(jnp.zeros(N), K),
                obs=Y_jax[:, d],
            )

    nuts   = NUTS(model)
    mcmc   = MCMC(nuts, num_samples=num_samples, num_warmup=warmup_steps,
                  num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))

    raw = mcmc.get_samples()
    ls_raw = np.array(raw["log_lengthscale"])   # (n_samples, L)

    samples = {
        "log_lengthscale_0": ls_raw[:, 0],
        "log_lengthscale_1": ls_raw[:, 1],
        "log_variance":      np.array(raw["log_variance"]),
        "log_noise":         np.array(raw["log_noise"]),
    }
    return samples


if __name__ == "__main__":
    from data_gen import generate_gplvm_data
    Y, X_prior_mean, true_params, _ = generate_gplvm_data()
    samples = run_pyro_gplvm(Y, X_prior_mean)
    for k, v in samples.items():
        print(f"{k}: mean={v.mean():.4f}  true={true_params[k]:.4f}")
