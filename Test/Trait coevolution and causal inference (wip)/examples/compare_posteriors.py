"""Phase 4-5 of the BI model_translation workflow: score a BI translation
against its Stan reference posterior, parameter by parameter.

Produces, per example:
  results/translation_comparison.csv   ref_mean, bf_mean, KL, JSD, flag
  results/translation_density_overlay.png
  results/translation_kl_progress.png
  results/translation_jsd_log.csv      appended each run, drives the progress plot

Note on provenance: the BI server's model_translation_validation skill ships the
canonical plotting code in the BI_agents repo, which is not present on this
machine (BI_mcp/mcp_server/server.py expects it at ../BI_agents/skills/). The
plots below follow the specification given in the workflow text -- per-parameter
KDE overlay, and per-parameter JSD over kept iterations -- but are not that
file's code.

JSD is reported as the Jensen-Shannon *divergence* in base 2, which lies in
[0, 1]; scipy's `jensenshannon` returns its square root (a distance), which is
also tabulated so the threshold cannot be misread.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

THRESHOLD = 0.05


def stan_name(base, idx):
    """Render a numpy index in Stan's 1-based bracket convention."""
    return base if not idx else f"{base}[{','.join(str(i + 1) for i in idx)}]"


def flatten_bi(npz):
    """BI posteriors -> {stan-style name: 1d draws}.

    Arrays arrive as (chains, draws, *event) or (draws, *event); chains are
    pooled, matching how the Stan draws_df is read below.
    """
    out = {}
    for base, arr in npz.items():
        a = np.asarray(arr)
        if a.ndim >= 2 and a.shape[0] <= 16 and a.shape[1] > 50:
            a = a.reshape(-1, *a.shape[2:])          # pool chains
        n, event = a.shape[0], a.shape[1:]
        for idx in np.ndindex(*event) if event else [()]:
            out[stan_name(base, idx)] = a[(slice(None),) + idx].reshape(n)
    return out


def load_stan(path, bases):
    df = pd.read_csv(path)
    keep = {}
    for c in df.columns:
        base = c.split("[")[0]
        if base in bases and not c.startswith("."):
            keep[c] = df[c].to_numpy()
    return keep


def kl_and_jsd(p_draws, q_draws, grid_n=512):
    """KL(ref||bi) and Jensen-Shannon divergence from KDEs on a shared grid."""
    lo = min(p_draws.min(), q_draws.min())
    hi = max(p_draws.max(), q_draws.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.nan, np.nan, np.nan
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_n)
    try:
        p = gaussian_kde(p_draws)(grid)
        q = gaussian_kde(q_draws)(grid)
    except np.linalg.LinAlgError:      # a parameter held at a constant
        return np.nan, np.nan, np.nan
    eps = 1e-12
    p = np.clip(p, eps, None); p /= p.sum()
    q = np.clip(q, eps, None); q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    dist = float(jensenshannon(p, q, base=2))
    return kl, dist ** 2, dist


def flag(jsd):
    if np.isnan(jsd):
        return "n/a"
    return "PASS" if jsd < THRESHOLD else ("WARN" if jsd < 0.10 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, help="cichlid | primates")
    ap.add_argument("--root", default="/home/sebastian_sosa/phylo/examples")
    ap.add_argument("--draws", default="bf_draws.npz",
                    help="draws file inside <example>/results/")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    res = os.path.join(args.root, args.example, "results")
    bi = flatten_bi(np.load(os.path.join(res, args.draws)))
    bases = {k.split("[")[0] for k in bi}
    stan = load_stan(os.path.join(res, "stan_draws.csv"), bases)

    # Stan declares the tree-indexed parameters as array[N_tree], so eta_anc
    # arrives as eta_anc[1,j] against BI's eta_anc[j]. With a single tree the
    # leading index is always 1; drop it so these parameters get scored.
    renamed = {}
    for k, v in stan.items():
        if "[1," in k:
            alt = k.replace("[1,", "[", 1)
            if alt in bi and alt not in stan:
                renamed[alt] = v
                continue
        renamed[k] = v
    stan = renamed

    common = [k for k in stan if k in bi]
    missing = sorted(set(stan) - set(bi)) + sorted(set(bi) - set(stan))
    if missing:
        print("not matched:", missing)

    rows = []
    for name in sorted(common):
        s, b = stan[name], bi[name]
        # a parameter fixed to zero by effects_mat carries no information
        if np.std(s) < 1e-10 and np.std(b) < 1e-10:
            rows.append(dict(parameter=name, ref_mean=s.mean(), bf_mean=b.mean(),
                             ref_sd=s.std(), bf_sd=b.std(), KL=0.0, JSD=0.0,
                             JS_distance=0.0, flag="fixed"))
            continue
        kl, jsd, dist = kl_and_jsd(s, b)
        rows.append(dict(parameter=name, ref_mean=s.mean(), bf_mean=b.mean(),
                         ref_sd=s.std(), bf_sd=b.std(), KL=kl, JSD=jsd,
                         JS_distance=dist, flag=flag(jsd)))

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(res, f"translation_comparison{args.tag}.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(tab.to_string(index=False,
                            float_format=lambda x: f"{x: .4f}"))

    live = tab[tab.flag != "fixed"]
    worst = live.loc[live.JSD.idxmax()] if len(live) else None
    print(f"\nparameters compared : {len(tab)} ({len(live)} free)")
    if worst is not None:
        print(f"max JSD             : {worst.JSD:.4f}  ({worst.parameter})")
        print(f"verdict             : "
              f"{'ALL PASS (JSD < 0.05)' if worst.JSD < THRESHOLD else 'ITERATE (JSD >= 0.05)'}")

    # --- JSD history, appended per run, drives the progress plot -----------
    log_path = os.path.join(res, "translation_jsd_log.csv")
    entry = live[["parameter", "JSD"]].copy()
    entry["iteration"] = 1
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        entry["iteration"] = old["iteration"].max() + 1
        entry = pd.concat([old, entry], ignore_index=True)
    entry.to_csv(log_path, index=False)

    plot(tab, live, stan, bi, entry, res, args.example)


def plot(tab, live, stan, bi, log, res, example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REF, BIC = "#4C72B0", "#DD8452"

    # --- density overlay, one panel per free parameter --------------------
    names = list(live.parameter)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        s, b = stan[name], bi[name]
        lo = min(s.min(), b.min()); hi = max(s.max(), b.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - pad, hi + pad, 400)
        for draws, c, lab in ((s, REF, "Stan"), (b, BIC, "BF")):
            try:
                ax.fill_between(grid, gaussian_kde(draws)(grid), color=c,
                                alpha=0.45, lw=0, label=lab)
            except np.linalg.LinAlgError:
                pass
        j = float(live.loc[live.parameter == name, "JSD"].iloc[0])
        ax.set_title(f"{name}   JSD={j:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{example}: posterior density, Stan reference vs BF translation",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(res, "translation_density_overlay.png"), dpi=150)
    plt.close(fig)

    # --- per-parameter JSD over kept iterations ---------------------------
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(names)), 4.2))
    iters = sorted(log.iteration.unique())
    if len(iters) == 1:
        d = log.sort_values("JSD", ascending=False)
        cols = [{"PASS": "#55A868", "WARN": "#DD8452"}.get(
            flag(v), "#C44E52") for v in d.JSD]
        ax.bar(range(len(d)), d.JSD, color=cols)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.parameter, rotation=90, fontsize=7)
        ax.set_ylabel("Jensen-Shannon divergence")
    else:
        for name, g in log.groupby("parameter"):
            ax.plot(g.iteration, g.JSD, marker="o", ms=3, lw=1, label=name)
        ax.set_xlabel("iteration"); ax.set_ylabel("Jensen-Shannon divergence")
        ax.set_xticks(iters)
        if len(names) <= 12:
            ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.axhline(THRESHOLD, color="k", ls="--", lw=1,
               label=f"threshold {THRESHOLD}")
    ax.set_title(f"{example}: per-parameter JSD"
                 f"{' over kept iterations' if len(iters) > 1 else ''}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(res, "translation_kl_progress.png"), dpi=150)
    plt.close(fig)
    print("\nwrote translation_density_overlay.png and translation_kl_progress.png")


if __name__ == "__main__":
    main()
