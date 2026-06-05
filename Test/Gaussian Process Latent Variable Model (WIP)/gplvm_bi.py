"""BI (NumPyro NUTS) GPLVM — translated backend.

RBF kernel implemented in JAX (no native GP in BI).
GP likelihood injected via numpyro.factor; all latent params tracked by BI.
"""
import sys
sys.path.insert(0, "C:/Users/Sosa/Documents/BI")

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from BI import bi


def run_bi_gplvm(Y_np, X_prior_mean_np, num_samples=500, num_warmup=300, seed=0):
    N, D = Y_np.shape
    L = X_prior_mean_np.shape[1]

    Y_jax   = jnp.array(Y_np, dtype=jnp.float32)
    Xpm_jax = jnp.array(X_prior_mean_np, dtype=jnp.float32)

    data = {"Y": Y_jax, "N": N, "D": D, "L": L, "X_prior_mean": Xpm_jax}

    m = bi(platform="cpu")
    m.data_on_model = {"data": data}

    def bi_gplvm(data):
        N_d = data["N"]
        D_d = data["D"]
        L_d = data["L"]

        # --- hyperparameter priors (log-space → positive) ---
        log_ls  = m.dist.normal(jnp.zeros(L_d), jnp.ones(L_d),
                                shape=(L_d,), name="log_lengthscale")
        log_var = m.dist.normal(0.0, 1.0, name="log_variance")
        log_noise = m.dist.normal(-2.0, 0.5, name="log_noise")

        ls    = jnp.exp(log_ls)
        var   = jnp.exp(log_var)
        noise = jnp.exp(log_noise)

        # --- latent positions X ~ Normal(prior_mean, 1) ---
        X_flat = m.dist.normal(
            data["X_prior_mean"].reshape(-1),
            jnp.ones(N_d * L_d),
            shape=(N_d * L_d,),
            name="X_flat",
        )
        X = X_flat.reshape(N_d, L_d)

        # --- RBF kernel ---
        diff    = X[:, None, :] - X[None, :, :]          # N x N x L
        sq_dist = jnp.sum((diff / ls) ** 2, axis=-1)   # N x N
        K = var * jnp.exp(-0.5 * sq_dist) + (noise + 1e-6) * jnp.eye(N_d)

        # --- GP likelihood: each gene is MVN(0, K) ---
        for d in range(D_d):
            log_p = ndist.MultivariateNormal(
                jnp.zeros(N_d), K
            ).log_prob(data["Y"][:, d])
            numpyro.factor(f"gp_lik_{d}", log_p)

    m.fit(
        bi_gplvm,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=1,
        progress_bar=False,
        seed=seed,
    )

    # Extract hyperparameter posteriors (BI tracks m.dist.* by name)
    def _get(key):
        arr = np.array(m.posteriors[key]).flatten()
        return arr

    raw_ls = _get("log_lengthscale")          # shape: (num_samples * L,)
    n = num_samples
    samples = {
        "log_lengthscale_0": raw_ls[:n] if raw_ls.ndim == 1 else raw_ls[:, 0],
        "log_lengthscale_1": raw_ls[n:] if raw_ls.ndim == 1 else raw_ls[:, 1],
        "log_variance":      _get("log_variance"),
        "log_noise":         _get("log_noise"),
    }

    # If BI stored log_lengthscale as (n_samples, L) array, re-extract
    ls_raw = np.array(m.posteriors["log_lengthscale"])
    if ls_raw.ndim == 2 and ls_raw.shape[-1] == 2:
        samples["log_lengthscale_0"] = ls_raw[:, 0].flatten()
        samples["log_lengthscale_1"] = ls_raw[:, 1].flatten()
    elif ls_raw.ndim == 2 and ls_raw.shape[0] == 2:
        samples["log_lengthscale_0"] = ls_raw[0].flatten()
        samples["log_lengthscale_1"] = ls_raw[1].flatten()

    # Trim all to min length for alignment
    min_n = min(len(v) for v in samples.values())
    samples = {k: v[:min_n] for k, v in samples.items()}
    return samples


if __name__ == "__main__":
    from data_gen import generate_gplvm_data
    Y, X_prior_mean, true_params, _ = generate_gplvm_data()
    samples = run_bi_gplvm(Y, X_prior_mean)
    for k, v in samples.items():
        print(f"{k}: mean={v.mean():.4f}  true={true_params[k]:.4f}")
