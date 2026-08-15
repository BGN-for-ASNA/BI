"""
Compare parameter estimates between BF (NumPyro GGMM) and reference (R phylolm).
Generates:
  - parameter_comparison.png
  - parameter_comparison.txt
"""

import os
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_CSV = os.path.join(OUT_DIR, "reference_estimates.csv")
BF_CSV  = os.path.join(OUT_DIR, "BF_estimates.csv")


def run_reference():
    r_script = os.path.join(OUT_DIR, "reference_phylolm.R")
    print("Running R reference (phylolm)...")
    result = subprocess.run(["Rscript", r_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("reference_phylolm.R failed")
    print(result.stdout)


def run_BF():
    print("Running BF GGMM (NumPyro)...")
    # Import and run inline
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(OUT_DIR))))
    import jax
    from ggmm_BF import run_ggmm, posterior_means

    samples = run_ggmm(num_warmup=500, num_samples=1000, num_chains=2)
    means = posterior_means(samples)

    param_map = {
        "b1":                means.get("b1"),
        "b2":                means.get("b2"),
        "xbar_metabolism":   means.get("xbar_0"),
        "xbar_range":        means.get("xbar_1"),
        "xbar_size":         means.get("xbar_2"),
        "theta_metabolism":  np.exp(means.get("ln_theta_0", np.nan)),
        "theta_range":       np.exp(means.get("ln_theta_1", np.nan)),
        "theta_size":        np.exp(means.get("ln_theta_2", np.nan)),
        "sigma2_metabolism": np.exp(means.get("ln_sigma2_0", np.nan)),
        "sigma2_range":      np.exp(means.get("ln_sigma2_1", np.nan)),
        "sigma2_size":       np.exp(means.get("ln_sigma2_2", np.nan)),
    }
    BF_df = pd.DataFrame([{"parameter": k, "estimate": v} for k, v in param_map.items()])
    BF_df.to_csv(BF_CSV, index=False)
    return BF_df


def make_outputs(ref_df, BF_df):
    merged = ref_df.merge(BF_df, on="parameter", suffixes=("_ref", "_bi"))
    merged["difference"] = merged["estimate_BF"] - merged["estimate_ref"]
    merged = merged.rename(columns={"estimate_ref": "ref_mean", "estimate_BF": "BF_mean"})

    # ── TXT table ──
    txt_path = os.path.join(OUT_DIR, "parameter_comparison.txt")
    col_w = [22, 14, 14, 12]
    header = (
        f"{'Parameter':<{col_w[0]}}{'Reference (R)':{col_w[1]}}"
        f"{'BF (NumPyro)':{col_w[2]}}{'Difference':{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    lines = [header, sep]
    for _, row in merged.iterrows():
        lines.append(
            f"{row['parameter']:<{col_w[0]}}"
            f"{row['ref_mean']:>{col_w[1]}.4f}"
            f"{row['BF_mean']:>{col_w[2]}.4f}"
            f"{row['difference']:>{col_w[3]}.4f}"
        )
    txt_content = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(txt_content + "\n")
    print("\n" + txt_content)
    print(f"\nSaved: {txt_path}")

    # ── PNG comparison ──
    params     = merged["parameter"].tolist()
    ref_vals   = merged["ref_mean"].values
    BF_vals    = merged["BF_mean"].values
    diffs      = merged["difference"].values
    n_params   = len(params)
    x          = np.arange(n_params)
    width      = 0.35

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5], hspace=0.45)

    # Panel 1: side-by-side bars
    ax1 = fig.add_subplot(gs[0])
    bars_ref = ax1.bar(x - width / 2, ref_vals, width, label="Reference (R phylolm)",
                       color="#2196F3", alpha=0.85, edgecolor="white")
    bars_BF  = ax1.bar(x + width / 2, BF_vals,  width, label="BF (NumPyro GGMM)",
                       color="#FF5722", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Estimate")
    ax1.set_title("GGMM Parameter Comparison: Reference (R) vs BF (NumPyro)")
    ax1.legend(framealpha=0.9)
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    # Panel 2: difference bars
    ax2 = fig.add_subplot(gs[1])
    colors = ["#4CAF50" if abs(d) < 0.05 * max(abs(ref_vals) + 1e-8) else "#F44336"
              for d in diffs]
    ax2.bar(x, diffs, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(params, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("BF − Reference")
    ax2.set_title("Difference (green ≈ close, red ≈ large discrepancy)")
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="-")

    png_path = os.path.join(OUT_DIR, "parameter_comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    return merged


if __name__ == "__main__":
    # 1. Reference
    if not os.path.exists(REF_CSV):
        run_reference()
    ref_df = pd.read_csv(REF_CSV)

    # 2. BF
    if "--skip-BF" in sys.argv and os.path.exists(BF_CSV):
        BF_df = pd.read_csv(BF_CSV)
        print("Loaded existing BF_estimates.csv")
    else:
        BF_df = run_BF()

    # 3. Compare
    merged = make_outputs(ref_df, BF_df)
    print("\nDone.")
