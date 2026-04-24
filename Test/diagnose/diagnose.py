# %%
"""
MCMC convergence diagnostics for BI models.

Checks treedepth, divergences, E-BFMI, rank-normalized split ESS, and R-hat.
Mirrors the diagnostic output of CmdStan's `diagnose` utility.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize samples across all chains. chains: (C, S)."""
    flat = chains.flatten()
    ranks = stats.rankdata(flat)
    z = stats.norm.ppf((ranks - 0.375) / (flat.size + 0.25))
    return z.reshape(chains.shape)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half -> (2*C, S//2)."""
    C, S = chains.shape
    half = S // 2
    return np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)


def _rhat(chains: np.ndarray) -> float:
    """Rank-normalized split R-hat. chains: (C, S)."""
    rn = _rank_normalize(chains)
    split = _split_chains(rn)
    m, n = split.shape
    chain_means = split.mean(axis=1)
    overall_mean = chain_means.mean()
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(split, axis=1, ddof=1))
    if W == 0:
        return np.nan
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def _ess_bulk(chains: np.ndarray) -> float:
    """Rank-normalized split bulk ESS. chains: (C, S)."""
    rn = _rank_normalize(chains)
    return _ess_raw(_split_chains(rn))


def _ess_tail(chains: np.ndarray, prob: float = 0.05) -> float:
    """Tail ESS at prob and 1-prob quantiles (min of both)."""
    q_lo = (chains < np.quantile(chains, prob)).astype(float)
    q_hi = (chains < np.quantile(chains, 1 - prob)).astype(float)
    return min(_ess_raw(_split_chains(q_lo)), _ess_raw(_split_chains(q_hi)))


def _ess_raw(split: np.ndarray) -> float:
    """ESS from split chains via autocorrelation. split: (M, N)."""
    M, N = split.shape
    total = M * N
    # per-chain autocorrelations via FFT
    acovs = []
    for chain in split:
        x = chain - chain.mean()
        n = len(x)
        f = np.fft.rfft(x, n=2 * n)
        acov = np.fft.irfft(f * np.conj(f))[:n] / n
        acovs.append(acov)
    acovs = np.array(acovs)  # (M, N)
    mean_acov = acovs.mean(axis=0)  # (N,)
    var_hat = mean_acov[0]
    if var_hat == 0:
        return np.nan
    # Geyer's initial positive sequence
    rho = mean_acov / var_hat
    ess_sum = 0.0
    for t in range(1, N // 2):
        pair = rho[2 * t - 1] + rho[2 * t]
        if pair <= 0:
            break
        ess_sum += pair
    tau = -1 + 2 * ess_sum
    return float(total / max(1.0 + tau, 1.0))


def _ebfmi(energy: np.ndarray) -> list:
    """E-BFMI per chain. energy: (C, S). Returns list of floats."""
    result = []
    for chain in energy:
        delta = np.diff(chain)
        denom = np.var(chain, ddof=1)
        result.append(float(np.var(delta, ddof=1) / denom) if denom > 0 else np.nan)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose(
    m,
    max_treedepth: int = 10,
    ebfmi_threshold: float = 0.3,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
) -> str:
    """
    Run MCMC convergence diagnostics on a fitted BI model.

    Parameters
    ----------
    m : bi
        Fitted BI object (after calling m.fit()).
    max_treedepth : int
        Maximum tree depth used during sampling (default 10, matching BI default).
    ebfmi_threshold : float
        E-BFMI values below this trigger a warning (default 0.3).
    rhat_threshold : float
        R-hat values above this trigger a warning (default 1.01).
    ess_threshold : float
        ESS values below this trigger a warning (default 400).

    Returns
    -------
    str
        Human-readable diagnostic report.
    """
    lines = []
    problems = []
    extra = m.sampler.get_extra_fields(group_by_chain=True)
    num_chains = m.num_chains
    posteriors_by_chain = m.sampler.get_samples(group_by_chain=True)

    # ------------------------------------------------------------------
    # 1. Treedepth
    # ------------------------------------------------------------------
    lines.append("Checking sampler transitions treedepth.")
    if "num_steps" in extra and extra["num_steps"] is not None:
        num_steps = np.array(extra["num_steps"])  # (C, S)
        if num_steps.ndim == 1:
            num_steps = num_steps.reshape(1, -1)
        tree_depth = np.ceil(np.log2(num_steps + 1)).astype(int)
        per_chain_max_treedepth = np.sum(tree_depth >= max_treedepth, axis=1).tolist()
        saturated = sum(per_chain_max_treedepth)
        chain_vals = " ".join(f"{v}" for v in per_chain_max_treedepth)
        lines.append(f"$num_max_treedepth\n[1] {chain_vals}")
        if saturated > 0:
            msg = (
                f"{saturated} of {tree_depth.size} transitions hit the "
                f"maximum treedepth limit of {max_treedepth}. "
                "Increase max_tree_depth to improve sampling efficiency."
            )
            lines.append(msg)
            problems.append(msg)
        else:
            lines.append("Treedepth satisfactory for all transitions.")
    else:
        lines.append("Treedepth: extra_fields not collected (unavailable).")
    lines.append("")

    # ------------------------------------------------------------------
    # 2. Divergences
    # ------------------------------------------------------------------
    lines.append("Checking sampler transitions for divergences.")
    if "diverging" in extra and extra["diverging"] is not None:
        diverging = np.array(extra["diverging"])
        if diverging.ndim == 1:
            diverging = diverging.reshape(1, -1)
        per_chain_div = np.sum(diverging, axis=1).tolist()
        n_div = int(sum(per_chain_div))
        chain_vals = " ".join(f"{int(v)}" for v in per_chain_div)
        lines.append(f"$num_divergent\n[1] {chain_vals}")
        if n_div > 0:
            msg = (
                f"{n_div} divergent transition(s) found after warmup. "
                "Try increasing target_accept_prob or reparameterizing the model."
            )
            lines.append(msg)
            problems.append(msg)
        else:
            lines.append("No divergent transitions found.")
    else:
        lines.append("Divergences: extra_fields not collected (unavailable).")
    lines.append("")

    # ------------------------------------------------------------------
    # 3. E-BFMI
    # ------------------------------------------------------------------
    lines.append("Checking E-BFMI - sampler transitions HMC potential energy.")
    if "energy" in extra and extra["energy"] is not None:
        energy = np.array(extra["energy"])
        if energy.ndim == 1:
            energy = energy.reshape(1, -1)
        ebfmi_vals = _ebfmi(energy)
        chain_vals = " ".join(f"{v:.6f}" for v in ebfmi_vals)
        lines.append(f"$ebfmi\n[1] {chain_vals}")
        low_chains = [
            i
            for i, v in enumerate(ebfmi_vals)
            if not np.isnan(v) and v < ebfmi_threshold
        ]
        if low_chains:
            detail = ", ".join(
                f"chain {i}: E-BFMI={ebfmi_vals[i]:.3f}" for i in low_chains
            )
            msg = f"E-BFMI below threshold ({ebfmi_threshold}): {detail}."
            lines.append(msg)
            problems.append(msg)
        else:
            lines.append("E-BFMI satisfactory.")
    else:
        lines.append("E-BFMI: extra_fields not collected (unavailable).")
    lines.append("")

    # ------------------------------------------------------------------
    # 4. ESS
    # ------------------------------------------------------------------
    lines.append("Checking rank-normalized split effective sample size.")
    ess_problems = []
    for param, samples in posteriors_by_chain.items():
        arr = np.array(samples)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # flatten any extra dims into param indices
        C, S = arr.shape[0], arr.shape[1]
        extra_dims = arr.shape[2:]
        flat = arr.reshape(C, S, -1) if extra_dims else arr.reshape(C, S, 1)
        for idx in range(flat.shape[2]):
            chains = flat[:, :, idx]
            ess = _ess_bulk(chains)
            label = f"{param}[{idx}]" if extra_dims else param
            if not np.isnan(ess) and ess < ess_threshold:
                ess_problems.append(f"{label}: ESS={ess:.1f}")
    if ess_problems:
        msg = "Low ESS for: " + "; ".join(ess_problems) + "."
        lines.append(msg)
        problems.append(msg)
    else:
        lines.append(
            "Rank-normalized split effective sample size satisfactory for all parameters."
        )
    lines.append("")

    # ------------------------------------------------------------------
    # 5. R-hat
    # ------------------------------------------------------------------
    lines.append("Checking rank-normalized split R-hat values.")
    rhat_problems = []
    for param, samples in posteriors_by_chain.items():
        arr = np.array(samples)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        C, S = arr.shape[0], arr.shape[1]
        extra_dims = arr.shape[2:]
        flat = arr.reshape(C, S, -1) if extra_dims else arr.reshape(C, S, 1)
        for idx in range(flat.shape[2]):
            chains = flat[:, :, idx]
            rh = _rhat(chains)
            label = f"{param}[{idx}]" if extra_dims else param
            if not np.isnan(rh) and rh > rhat_threshold:
                rhat_problems.append(f"{label}: R-hat={rh:.4f}")
    if rhat_problems:
        msg = "High R-hat for: " + "; ".join(rhat_problems) + "."
        lines.append(msg)
        problems.append(msg)
    else:
        lines.append(
            "Rank-normalized split R-hat values satisfactory for all parameters."
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if problems:
        lines.append(f"Processing complete, {len(problems)} problem(s) detected.")
    else:
        lines.append("Processing complete, no problems detected.")

    report = "\n".join(lines)
    print(report)
    return report


# %%
from BI import bi

# Setup device------------------------------------------------
m = bi(platform="cpu")

# Import Data & Data Manipulation ------------------------------------------------
# Import
from importlib.resources import files

data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=";")
m.df = m.df[m.df.age > 18]  # Subset data to adults
m.scale(["weight"])  # Normalize


# Define model ------------------------------------------------
def model(weight, height):
    a = m.dist.normal(178, 20, name="a")
    b = m.dist.log_normal(0, 1, name="b")
    s = m.dist.uniform(0, 50, name="s")
    m.dist.normal(a + b * weight, s, obs=height)


# Run mcmc ------------------------------------------------
m.fit(model, progress_bar=False)  # Optimize model parameters through MCMC sampling

# Summary ------------------------------------------------
m.summary()  # Get posterior distributions
# %%
diagnose(m)

# %%
