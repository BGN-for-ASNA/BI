"""
GPLVM Validation Pipeline — Pyro vs BI (NumPyro) backends.

Outputs:
  translation_density_overlay.png  — posterior density comparison per parameter
  translation_kl_progress.png      — KL divergence bar chart
  log.txt                          — table: param | pyro_mean | bi_mean | diff | kl_div
"""
import os
import sys
sys.path.insert(0, "C:/Users/Sosa/Documents/BI")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import rel_entr

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT_DIR)

from data_gen import generate_gplvm_data
from gplvm_pyro import run_pyro_gplvm
from gplvm_bi import run_bi_gplvm

PARAMS = ["log_lengthscale_0", "log_lengthscale_1", "log_variance", "log_noise"]
PARAM_LABELS = {
    "log_lengthscale_0": r"$\log\,\ell_1$",
    "log_lengthscale_1": r"$\log\,\ell_2$",
    "log_variance":      r"$\log\,\sigma^2$",
    "log_noise":         r"$\log\,\sigma^2_n$",
}


def kl_divergence_kde(samples_p, samples_q, n_points=500):
    """KL(P || Q) estimated via KDE on shared grid."""
    lo = min(samples_p.min(), samples_q.min()) - 0.5
    hi = max(samples_p.max(), samples_q.max()) + 0.5
    grid = np.linspace(lo, hi, n_points)

    kde_p = gaussian_kde(samples_p)(grid)
    kde_q = gaussian_kde(samples_q)(grid)

    kde_p = np.clip(kde_p, 1e-10, None)
    kde_q = np.clip(kde_q, 1e-10, None)
    kde_p /= kde_p.sum()
    kde_q /= kde_q.sum()

    return float(np.sum(rel_entr(kde_p, kde_q)))


def jsd(samples_p, samples_q, n_points=500):
    """Jensen-Shannon divergence between two sample sets via KDE."""
    lo = min(samples_p.min(), samples_q.min()) - 0.5
    hi = max(samples_p.max(), samples_q.max()) + 0.5
    grid = np.linspace(lo, hi, n_points)

    p = gaussian_kde(samples_p)(grid)
    q = gaussian_kde(samples_q)(grid)
    p = np.clip(p, 1e-10, None); p /= p.sum()
    q = np.clip(q, 1e-10, None); q /= q.sum()
    m = 0.5 * (p + q)

    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


def plot_density_overlay(pyro_samples, bi_samples, true_params, out_path):
    n_params = len(PARAMS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, param in enumerate(PARAMS):
        ax = axes[i]
        p_s = pyro_samples[param]
        b_s = bi_samples[param]
        true_v = true_params[param]

        lo = min(p_s.min(), b_s.min()) - 0.3
        hi = max(p_s.max(), b_s.max()) + 0.3
        grid = np.linspace(lo, hi, 400)

        ax.plot(grid, gaussian_kde(p_s)(grid), color="#E05C5C", lw=2, label="Pyro (NUTS)")
        ax.plot(grid, gaussian_kde(b_s)(grid), color="#4C8BF5", lw=2, label="BI (NumPyro)", ls="--")
        ax.axvline(true_v, color="k", lw=1.5, ls=":", label="True")

        kl = jsd(p_s, b_s)
        ax.set_title(f"{PARAM_LABELS[param]}  (JSD={kl:.4f})", fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("GPLVM: Pyro vs BI Posterior Density Overlay", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_kl_progress(kl_values, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50" if v < 0.05 else "#FF9800" if v < 0.10 else "#F44336"
              for v in kl_values.values()]
    bars = ax.bar(
        [PARAM_LABELS[p] for p in kl_values],
        list(kl_values.values()),
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.05, color="green", ls="--", lw=1.5, label="JSD=0.05 threshold")
    ax.axhline(0.10, color="orange", ls="--", lw=1.5, label="JSD=0.10 threshold")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("GPLVM Translation Validation: JSD per Parameter", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(kl_values.values()) * 1.3, 0.12))
    for bar, val in zip(bars, kl_values.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def write_log(pyro_samples, bi_samples, kl_values, out_path):
    col_w = [28, 14, 14, 14, 14]
    header = ["Parameter", "Pyro Mean", "BI Mean", "Diff", "JSD"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    lines = [sep]
    lines.append("|" + "|".join(h.center(w) for h, w in zip(header, col_w)) + "|")
    lines.append(sep)

    for param in PARAMS:
        p_mean = float(pyro_samples[param].mean())
        b_mean = float(bi_samples[param].mean())
        diff = p_mean - b_mean
        kl = kl_values[param]
        row = [param, f"{p_mean:.6f}", f"{b_mean:.6f}", f"{diff:+.6f}", f"{kl:.6f}"]
        lines.append("|" + "|".join(v.center(w) for v, w in zip(row, col_w)) + "|")

    lines.append(sep)
    lines.append("")
    lines.append("JSD < 0.05 → validated  |  0.05-0.10 → marginal  |  > 0.10 → divergent")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved: {out_path}")
    print(text)


def main():
    print("=" * 60)
    print("GPLVM Validation Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic data...")
    Y, X_prior_mean, true_params, X_true = generate_gplvm_data(N=30, D=4, L=2, seed=42)
    print(f"  Y shape: {Y.shape}  X_prior_mean shape: {X_prior_mean.shape}")
    print(f"  True params: { {k: f'{v:.3f}' for k, v in true_params.items()} }")

    JSD_TARGET = 0.01
    schedule = [
        (2000,  500),
        (4000, 1000),
        (8000, 2000),
    ]

    pyro_samples = bi_samples = kl_values = None
    for iteration, (n_samples, n_warmup) in enumerate(schedule, 1):
        print(f"\n[Iter {iteration}] n_samples={n_samples}  warmup={n_warmup}")

        print("  Running NumPyro NUTS (reference)...")
        pyro_samples = run_pyro_gplvm(
            Y, X_prior_mean, num_samples=n_samples, warmup_steps=n_warmup, seed=iteration
        )
        print("  Running BI NUTS...")
        bi_samples = run_bi_gplvm(
            Y, X_prior_mean, num_samples=n_samples, num_warmup=n_warmup, seed=iteration + 100
        )

        kl_values = {p: jsd(pyro_samples[p], bi_samples[p]) for p in PARAMS}
        worst = max(kl_values.values())
        print(f"  JSD: { {k: f'{v:.4f}' for k,v in kl_values.items()} }")
        print(f"  Worst JSD={worst:.4f}  {'→ DONE' if worst < JSD_TARGET else '→ iterating'}")
        if worst < JSD_TARGET:
            break

    print("\n[Final] Generating outputs...")
    kl_values = {p: jsd(pyro_samples[p], bi_samples[p]) for p in PARAMS}

    plot_density_overlay(
        pyro_samples, bi_samples, true_params,
        os.path.join(OUT_DIR, "translation_density_overlay.png"),
    )
    plot_kl_progress(
        kl_values,
        os.path.join(OUT_DIR, "translation_kl_progress.png"),
    )
    write_log(
        pyro_samples, bi_samples, kl_values,
        os.path.join(OUT_DIR, "log.txt"),
    )

    print("\nDone.")
    validated = all(v < 0.05 for v in kl_values.values())
    print(f"Translation status: {'VALIDATED' if validated else 'NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
