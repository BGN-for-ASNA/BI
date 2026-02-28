import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

@jit
def calculate_r_hat_jax(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Gelman-Rubin statistic (R-hat).
    chains: Array of shape (N, ..., Samples)
    where N is the number of chains and Samples is the number of draws.
    Returns: Array of shape (...) matching the parameter dimensions.
    """
    n_chains = chains.shape[0]
    n_draws = chains.shape[-1]
    
    within_chain_var = jnp.var(chains, axis=-1, ddof=1) # shape (N, ...)
    W = jnp.mean(within_chain_var, axis=0) # shape (...)
    
    chain_means = jnp.mean(chains, axis=-1) # shape (N, ...)
    B = n_draws * jnp.var(chain_means, axis=0, ddof=1) # shape (...)
    
    var_hat = ((n_draws - 1) / n_draws) * W + (1 / n_draws) * B
    
    # Avoid division by zero
    r_hat = jnp.where(W > 0, jnp.sqrt(var_hat / W), jnp.nan)
    
    # Return NaNs if n_chains < 2
    # Since JAX prefers static shapes/conditions when possible, 
    # we just overwrite with NaN if there's only 1 chain.
    if chains.shape[0] < 2:
        r_hat = jnp.full_like(r_hat, jnp.nan)
        
    return r_hat


@jit
def calculate_ess_jax(chains: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the Effective Sample Size (ESS) using autocorrelation.
    chains: Array of shape (N, ..., Samples)
    Returns: Array of shape (...) matching the parameter dimensions.
    """
    n_chains = chains.shape[0]
    n_draws = chains.shape[-1]
    
    def autocorr(x):
        # x is 1D array of shape (Samples,)
        mean = jnp.mean(x)
        var = jnp.var(x)
        x_centered = x - mean
        n = len(x_centered)
        fft_val = jnp.fft.fft(x_centered, n=2*n)
        autocorr_fft = jnp.fft.ifft(fft_val * jnp.conj(fft_val))
        autocorr_fft = autocorr_fft[:n]
        # Return autocorrelation, avoiding division by zero
        return jnp.where(var > 0, jnp.real(autocorr_fft) / (n * var), jnp.zeros_like(jnp.real(autocorr_fft)))

    # We want to apply autocorr over the last dimension (Samples).
    # Since chains can have arbitrary intermediate dimensions, we flatten them all,
    # map over the flattened dimension, and reconstruct.
    # Note: axis 0 is N, so flat_chains must merge N and the intermediate dimensions.
    # We want to run autocorr along 'Samples' for every independent sequence.
    # We have N chains, each of length Samples. For ESS, autocorr is computed PER CHAIN.
    # So we want to treat (N, ...) as independent 1D Series of length Samples.
    # To vmap over all these, we reshape to (N * prod(...), Samples)
    flat_chains = chains.reshape(-1, n_draws)
    rho_t_flat = vmap(autocorr)(flat_chains)
    rho_t = rho_t_flat.reshape(chains.shape) # shape (N, ..., Samples)
    
    # Sum autocorrelation pairs
    max_t = n_draws // 2
    
    # Create vectors of pairs for all t=1 to max_t-1
    # rho_sum_pair shape will be (N, ..., max_t-1)
    rho_t_even = rho_t[..., 2:2*max_t:2]
    rho_t_odd = rho_t[..., 3:2*max_t:2]
    rho_sum_pair = rho_t_even + rho_t_odd
    
    # Mean over chains (axis 0)
    # Shape becomes (..., max_t-1)
    mean_rho_sum_pair = jnp.mean(jnp.where(rho_sum_pair > 0, rho_sum_pair, 0), axis=0)
    
    # Stop condition: we only sum pairs until the first <= 0 is encountered.
    # We create a mask for valid pairs
    is_positive = mean_rho_sum_pair > 0
    # cumulative product acts as a cumulative AND
    mask = jnp.cumprod(is_positive, axis=-1)
    
    ess_sum = jnp.sum(mean_rho_sum_pair * mask, axis=-1)
    
    ess = (n_chains * n_draws) / (1 + 2 * ess_sum)
    return ess


def compute_diagnostics(posterior_dict: dict, prob: float = 0.94) -> dict:
    """
    Computes mean, sd, hdi, rhat, and ess for a dictionary of posterior samples.
    Adaptable to dimensions where axis 0 is chains (N) and axis -1 is samples.
    
    Parameters:
    - posterior_dict: Dictionary mapping variable names to jax arrays of shape (N, ..., Samples)
    - prob: Probability mass to include in the HDI (default 0.94)
    
    Returns:
    - Nested dictionary mapping variable names to their computed diagnostics.
    """
    results = {}
    lower_p = 50 * (1 - prob)
    upper_p = 50 * (1 + prob)
    
    for var_name, chains in posterior_dict.items():
        # Ensure minimum dimensions (N, Samples)
        if chains.ndim < 2:
            chains = jnp.expand_dims(chains, axis=0)
            
        # Compute summary statistics over both chains (axis 0) and samples (axis -1)
        mean_val = jnp.mean(chains, axis=(0, -1))
        sd_val = jnp.std(chains, axis=(0, -1), ddof=1)
        hdi_low = jnp.percentile(chains, lower_p, axis=(0, -1))
        hdi_high = jnp.percentile(chains, upper_p, axis=(0, -1))
        
        # Compute R-hat and ESS (these return arrays over the parameter dimensions)
        rhat = calculate_r_hat_jax(chains)
        ess = calculate_ess_jax(chains)
        
        results[var_name] = {
            "mean": mean_val,
            "sd": sd_val,
            f"hdi_{lower_p:g}%": hdi_low,
            f"hdi_{upper_p:g}%": hdi_high,
            "rhat": rhat,
            "ess": ess
        }
        
    return results
