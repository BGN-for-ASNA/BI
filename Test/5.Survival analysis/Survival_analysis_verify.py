import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import gaussian_kde
import pymc as pm
import pytensor.tensor as pt
import jax.numpy as jnp

from BayesForge import bf

def main():
    print("Initializing BF model...")
    m = bf(platform="cpu")

    print("Loading mastectomy data...")
    data_path = m.load.mastectomy(only_path=True)
    m.data(data_path)

    print("Preprocessing data...")
    # Convert 'yes'/'no' to 1/0
    m.df.metastasized = (m.df.metastasized.values == "yes").astype(jnp.int64)
    
    # Import time with even interval length = 3
    m.models.survival.import_time_even(m.df.time.values, m.df.event.values, interval_length=3)
    m.models.survival.import_covF(m.df.metastasized.values, ["metastasized"])

    print("Fitting BF model...")
    # Note: we use progress_bar=False to reduce terminal spam
    m.fit(m.models.survival.model, num_samples=1000, num_warmup=1000, num_chains=2, progress_bar=False, seed=42)

    print("Summarizing BF model...")
    BF_summary = m.summary()
    print(BF_summary)

    # --- PyMC Implementation ---
    print("Fitting PyMC model...")
    death = np.array(m.models.survival.death)
    exposure = np.array(m.models.survival.exposure)
    metastasized = np.array(m.df.metastasized.values)
    n_intervals = m.models.survival.n_intervals

    with pm.Model() as pymc_model:
        # Must match BF's Baseline_rate prior (m.models.survival.baseline_rate_prior).
        lambda0 = pm.Gamma("lambda0", *m.models.survival.baseline_rate_prior, shape=n_intervals)
        # sigma must match BF's Hazard_rate prior (Normal(0, 10)) for the
        # comparison to be like-for-like.
        beta = pm.Normal("beta", 0, sigma=m.models.survival.hazard_rate_prior_scale)

        # Hazard rate: lambda = lambda0 * exp(beta * covariate)
        lambda_ = pm.Deterministic("lambda_", pt.outer(pt.exp(beta * metastasized), lambda0))
        mu = pm.Deterministic("mu", exposure * lambda_)
        
        # Offset to keep the Poisson rate strictly positive. BF uses
        # jnp.finfo(float64).tiny here; 1e-12 is the PyMC-tutorial value.
        obs = pm.Poisson("obs", mu + 1e-12, observed=death)

        idata_pymc = pm.sample(1000, tune=1000, target_accept=0.99, random_seed=42, progressbar=False, chains=2, cores=1)

    # --- Comparison Plot ---
    print("Plotting comparison...")
    BF_beta = np.array(m.posteriors["Hazard_rate_metastasized"]).flatten()
    pymc_beta = idata_pymc.posterior["beta"].values.flatten()

    # Shared grid so the two densities are directly comparable, padded so the
    # tails are not clipped at the sample extremes.
    lo = min(BF_beta.min(), pymc_beta.min())
    hi = max(BF_beta.max(), pymc_beta.max())
    pad = 0.1 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for arr, color, label in [(BF_beta, "C0", "BF"), (pymc_beta, "C1", "PyMC")]:
        if arr.std() < 1e-8:
            # Frozen chains: a KDE here is a meaningless spike, so show the
            # draws themselves instead of pretending there is a density.
            print(f"WARNING: {label} draws are near-constant (std={arr.std():.2e}); "
                  "plotting locations instead of a density.")
            for x in np.unique(arr):
                ax.axvline(x, color=color, linestyle=":", label=label)
            continue
        ax.plot(xs, gaussian_kde(arr)(xs), color=color, label=label)
    ax.set_title("Posterior Distribution Comparison: Beta (metastasized)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "survival_comparison_verify.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    # --- Scatter Plot for Consistency ---
    print("Generating scatter plot for parameter consistency...")
    BF_lambda0_means = np.mean(np.array(m.posteriors["Baseline_rate"]), axis=0)
    pymc_lambda0_means = idata_pymc.posterior["lambda0"].mean(dim=("chain", "draw")).values

    BF_beta_mean = np.mean(BF_beta)
    pymc_beta_mean = np.mean(pymc_beta)

    # Combine all parameter means
    BF_all_means = np.concatenate([BF_lambda0_means, [BF_beta_mean]])
    pymc_all_means = np.concatenate([pymc_lambda0_means, [pymc_beta_mean]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pymc_all_means, BF_all_means, alpha=0.6, color="darkgreen", label="Parameters (lambda0 & beta)")

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="y = x (Perfect Match)")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Posterior Means Comparison: PyMC vs BF")
    ax.set_xlabel("PyMC Posterior Means")
    ax.set_ylabel("BF Posterior Means")
    ax.legend()

    scatter_path = os.path.join(script_dir, "survival_scatter_verify.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")
    
    print("\n--- Final Posterior Comparison Summary ---")
    print(f"BF Beta Mean:   {BF_beta_mean:.4f}")
    print(f"PyMC Beta Mean: {pymc_beta_mean:.4f}")
    print(f"Difference:     {abs(BF_beta_mean - pymc_beta_mean):.4f}")

    # A mean-to-mean match is meaningless if either sampler never moved, so
    # report convergence alongside it.
    print("\n--- Convergence ---")
    bf_beta_row = BF_summary.loc["Hazard_rate_metastasized[0]"]
    print(f"BF   beta: r_hat={bf_beta_row['r_hat']:.3g} ess_bulk={bf_beta_row['ess_bulk']:.1f}")
    pymc_beta_row = az.summary(idata_pymc, var_names=["beta"]).iloc[0]
    print(f"PyMC beta: r_hat={pymc_beta_row['r_hat']:.3g} ess_bulk={pymc_beta_row['ess_bulk']:.1f}")
    if bf_beta_row["r_hat"] > 1.05 or bf_beta_row["ess_bulk"] < 100:
        print("WARNING: BF chains did not converge - the comparison above is not valid.")
    print("Done!")

if __name__ == "__main__":
    main()
