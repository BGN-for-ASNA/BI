import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import jax.numpy as jnp

from BI import bi

def main():
    print("Initializing BI model...")
    m = bi(platform="cpu")

    print("Loading mastectomy data...")
    data_path = m.load.mastectomy(only_path=True)
    m.data(data_path)

    print("Preprocessing data...")
    # Convert 'yes'/'no' to 1/0
    m.df.metastasized = (m.df.metastasized.values == "yes").astype(jnp.int64)
    
    # Import time with even interval length = 3
    m.models.survival.import_time_even(m.df.time.values, m.df.event.values, interval_length=3)
    m.models.survival.import_covF(m.df.metastasized.values, ["metastasized"])

    print("Fitting BI model...")
    # Note: we use progress_bar=False to reduce terminal spam
    m.fit(m.models.survival.model, num_samples=1000, num_warmup=1000, num_chains=2, progress_bar=False, seed=42)

    print("Summarizing BI model...")
    bi_summary = m.summary()
    print(bi_summary)

    # --- PyMC Implementation ---
    print("Fitting PyMC model...")
    death = np.array(m.models.survival.death)
    exposure = np.array(m.models.survival.exposure)
    metastasized = np.array(m.df.metastasized.values)
    n_intervals = m.models.survival.n_intervals

    with pm.Model() as pymc_model:
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape=n_intervals)
        beta = pm.Normal("beta", 0, sigma=100)

        # Hazard rate: lambda = lambda0 * exp(beta * covariate)
        lambda_ = pm.Deterministic("lambda_", pt.outer(pt.exp(beta * metastasized), lambda0))
        mu = pm.Deterministic("mu", exposure * lambda_)
        
        # Adding tiny offset identical to BI for fair comparison
        obs = pm.Poisson("obs", mu + 1e-12, observed=death)

        idata_pymc = pm.sample(1000, tune=1000, target_accept=0.99, random_seed=42, progressbar=False, chains=2, cores=1)

    # --- Comparison Plot ---
    print("Plotting comparison...")
    bi_beta = np.array(m.posteriors["Hazard_rate_metastasized"]).flatten()
    pymc_beta = idata_pymc.posterior["beta"].values.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    az.plot_dist(bi_beta, color="C0", label="BI", ax=ax)
    az.plot_dist(pymc_beta, color="C1", label="PyMC", ax=ax)
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
    bi_lambda0_means = np.mean(np.array(m.posteriors["Baseline_rate"]), axis=0)
    pymc_lambda0_means = idata_pymc.posterior["lambda0"].mean(dim=("chain", "draw")).values

    bi_beta_mean = np.mean(bi_beta)
    pymc_beta_mean = np.mean(pymc_beta)

    # Combine all parameter means
    bi_all_means = np.concatenate([bi_lambda0_means, [bi_beta_mean]])
    pymc_all_means = np.concatenate([pymc_lambda0_means, [pymc_beta_mean]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pymc_all_means, bi_all_means, alpha=0.6, color="darkgreen", label="Parameters (lambda0 & beta)")

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="y = x (Perfect Match)")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Posterior Means Comparison: PyMC vs BI")
    ax.set_xlabel("PyMC Posterior Means")
    ax.set_ylabel("BI Posterior Means")
    ax.legend()

    scatter_path = os.path.join(script_dir, "survival_scatter_verify.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")
    
    print("\n--- Final Posterior Comparison Summary ---")
    print(f"BI Beta Mean:   {bi_beta_mean:.4f}")
    print(f"PyMC Beta Mean: {pymc_beta_mean:.4f}")
    print(f"Difference:     {abs(bi_beta_mean - pymc_beta_mean):.4f}")
    print("Done!")

if __name__ == "__main__":
    main()
