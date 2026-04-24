# %%
# from : https://www.pymc.io/projects/examples/en/latest/survival_analysis/survival_analysis.html
from BI import bi
import jax.numpy as jnp
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    m = bi(platform="cpu")

    print("Loading data...")
    data_path = m.load.mastectomy(only_path=True)
    m.data(data_path)

    print("Preprocessing data...")
    m.df.metastasized = (m.df.metastasized.values == "yes").astype(jnp.int64)
    m.models.survival.import_time_even(
        m.df.time.values, m.df.event.values, interval_length=3
    )

    print("Fitting BI model...")
    m.models.survival.import_covF(m.df.metastasized.values, ["metastasized"])
    m.fit(m.models.survival.model, num_samples=500)

    print("Summarizing BI model...")
    m.summary()

    # --- PyMC Implementation ---
    print("Fitting PyMC model...")
    death = np.array(m.models.survival.death)
    exposure = np.array(m.models.survival.exposure)
    metastasized = np.array(m.df.metastasized.values)
    n_intervals = m.models.survival.n_intervals

    with pm.Model() as pymc_model:
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape=n_intervals)
        beta = pm.Normal("beta", 0, sigma=1000)

        # Hazard rate: lambda = lambda0 * exp(beta * covariate)
        lambda_ = pm.Deterministic(
            "lambda_", pt.outer(pt.exp(beta * metastasized), lambda0)
        )
        mu = pm.Deterministic("mu", exposure * lambda_)
        obs = pm.Poisson("obs", mu, observed=death)

        idata_pymc = pm.sample(500, tune=500, target_accept=0.99, random_seed=42)

    # --- Comparison Plot ---
    print("Plotting comparison...")
    # Extract BI samples
    bi_beta = m.posteriors["Hazard_rate_metastasized"].flatten()
    pymc_beta = idata_pymc.posterior["beta"].values.flatten()

    # Create comparison for beta
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    az.plot_dist(bi_beta, color="C0", label="BI", ax=ax)
    az.plot_dist(pymc_beta, color="C1", label="PyMC", ax=ax)
    ax.set_title("Posterior Distribution Comparison: Beta (metastasized)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    # Save the plot
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "survival_comparison.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    # --- Scatter Plot for Consistency ---
    print("Generating scatter plot for parameter consistency...")
    bi_lambda0_means = np.mean(m.posteriors["Baseline_rate"], axis=0)
    pymc_lambda0_means = (
        idata_pymc.posterior["lambda0"].mean(dim=("chain", "draw")).values
    )

    bi_beta_mean = np.mean(m.posteriors["Hazard_rate_metastasized"])
    pymc_beta_mean = idata_pymc.posterior["beta"].mean().values

    # Combine all parameter means
    bi_all_means = np.concatenate([bi_lambda0_means, [bi_beta_mean]])
    pymc_all_means = np.concatenate([pymc_lambda0_means, [pymc_beta_mean]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pymc_all_means, bi_all_means, alpha=0.6, color="darkgreen")

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Posterior Means Comparison: PyMC vs BI")
    ax.set_xlabel("PyMC Posterior Means")
    ax.set_ylabel("BI Posterior Means")

    scatter_path = os.path.join(script_dir, "survival_scatter.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")

    # --- Logging Posterriors ---
    print("Logging posterior estimations...")
    try:
        bi_log = os.path.join(script_dir, "log_bi.txt")
        with open(bi_log, "w") as f:
            f.write(m.summary().to_string())
        print(f"BI posterior estimations logged to {bi_log}")
    except Exception as e:
        print(f"Failed to log BI: {e}")

    try:
        pymc_log = os.path.join(script_dir, "log_pymc.txt")
        print(f"Attempting to log PyMC summary to {pymc_log}...")
        # Try a simplified summary first
        pymc_summary = az.summary(idata_pymc, var_names=["beta", "lambda0"])
        with open(pymc_log, "w") as f:
            f.write(pymc_summary.to_string())
        print(f"PyMC posterior estimations logged to {pymc_log}")
    except Exception as e:
        print(f"Failed to log PyMC: {e}")
        # Fallback: manual mean calculation
        try:
            with open(pymc_log, "a") as f:
                f.write(f"\nManual fallback due to error: {e}\n")
                f.write(f"Beta mean: {idata_pymc.posterior['beta'].mean().values}\n")
        except:
            pass


if __name__ == "__main__":
    main()
