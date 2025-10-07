import jax.numpy as jnp
from jax import jit, vmap
from collections import defaultdict
import pandas as pd
import numpy as np

# --- Helper Functions for Statistics ---

@jit # JIT is fine here, no data-dependent control flow
def calculate_r_hat(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Gelman-Rubin statistic (R-hat).
    Assumes chains is of shape (n_chains, n_draws).
    """
    n_chains, n_draws = chains.shape
    within_chain_var = jnp.var(chains, axis=1, ddof=1)
    W = jnp.mean(within_chain_var)
    chain_means = jnp.mean(chains, axis=1)
    B = n_draws * jnp.var(chain_means, ddof=1)
    var_hat = ((n_draws - 1) / n_draws) * W + (1 / n_draws) * B
    return jnp.sqrt(var_hat / W)

# @jit <<<--- REMOVED THIS DECORATOR TO FIX THE ERROR
def calculate_ess(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Effective Sample Size (ESS) using autocorrelation.
    Assumes chains is of shape (n_chains, n_draws).
    """
    n_chains, n_draws = chains.shape
    
    def autocorr(x):
        mean = jnp.mean(x)
        var = jnp.var(x)
        x_centered = x - mean
        n = len(x_centered)
        fft_val = jnp.fft.fft(x_centered, n=2*n)
        autocorr_fft = jnp.fft.ifft(fft_val * jnp.conj(fft_val))
        autocorr_fft = autocorr_fft[:n]
        return jnp.real(autocorr_fft) / (n * var)

    # vmap is fine, it vectorizes the function
    rho_t = vmap(autocorr)(chains)
    
    # This loop is now executed in standard Python, not JIT compiled
    ess_sum = 0.0
    for t in range(1, n_draws // 2):
        rho_sum_pair = rho_t[:, 2*t] + rho_t[:, 2*t+1]
        
        # Taking the mean across chains
        mean_rho_sum_pair = jnp.mean(jnp.where(rho_sum_pair > 0, rho_sum_pair, 0))
        
        # The 'if' statement now works on a concrete value
        if mean_rho_sum_pair <= 0:
            break
        ess_sum += mean_rho_sum_pair
    
    ess = (n_chains * n_draws) / (1 + 2 * ess_sum)
    return ess

# --- Main Summary Function (Unchanged from previous fix) ---

def jax_summary(posterior_dict: dict) -> pd.DataFrame:
    """
    Computes summary statistics for a dictionary of posterior samples.
    Handles multi-dimensional parameters and single-chain cases.
    """
    summary_data = defaultdict(list)
    
    for var_name, samples in posterior_dict.items():
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        n_chains, n_draws = samples.shape[0], samples.shape[1]
        param_dims = samples.shape[2:]
        
        if param_dims:
            n_params = int(np.prod(param_dims))
            samples = samples.reshape(n_chains, n_draws, n_params)
            indices = list(np.ndindex(param_dims))
        else:
            n_params = 1
            samples = samples.reshape(n_chains, n_draws, 1)
            indices = [""]

        for i in range(n_params):
            param_slice = samples[:, :, i]
            full_var_name = f"{var_name}"
            if str(indices[i]):
                full_var_name += f"[{','.join(map(str, indices[i]))}]"

            flat_samples = param_slice.flatten()
            mean_val = jnp.mean(flat_samples)
            sd_val = jnp.std(flat_samples)
            hdi_low = jnp.percentile(flat_samples, 5.5)
            hdi_high = jnp.percentile(flat_samples, 94.5)

            r_hat = calculate_r_hat(param_slice) if n_chains > 1 else jnp.nan
            ess_bulk = calculate_ess(param_slice)
            ess_tail = ess_bulk # Using bulk as a proxy for tail
            mcse_mean = sd_val / jnp.sqrt(ess_bulk)
            mcse_sd = sd_val / jnp.sqrt(2 * (ess_bulk - 1))

            summary_data["var"].append(full_var_name)
            summary_data["mean"].append(round(float(mean_val), 3))
            summary_data["sd"].append(round(float(sd_val), 3))
            summary_data["hdi_5.5%"].append(round(float(hdi_low), 3))
            summary_data["hdi_94.5%"].append(round(float(hdi_high), 3))
            summary_data["mcse_mean"].append(round(float(mcse_mean), 3))
            summary_data["mcse_sd"].append(round(float(mcse_sd), 3))
            summary_data["ess_bulk"].append(round(float(ess_bulk), 3))
            summary_data["ess_tail"].append(round(float(ess_tail), 3))
            summary_data["r_hat"].append(round(float(r_hat), 3))
    return pd.DataFrame(summary_data).set_index("var").sort_index()