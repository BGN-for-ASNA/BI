"""Posterior visualisation for every parameter of a fitted GDPM.

Two figures per example, written to <example>/results/:

  posterior_intervals.png   point + 50%/89% HDI per parameter, Stan vs BF,
                            faceted by parameter block
  posterior_densities.png   marginal posterior density per parameter, each
                            annotated with KL(Stan||BF) and the Jensen-Shannon
                            divergence between the two posteriors

Why faceted rather than one panel: the parameter blocks live on wildly
different scales (primate `shape` is ~180 while `lambda_free` is ~0.17).
Small multiples, each with its own axis, is the honest way to show that --
a second y-axis on a shared panel would not be.

Palette is the dataviz reference instance, slots 1-2, validated for both
modes with scripts/validate_palette.js (worst adjacent CVD dE 24.7 light /
26.8 dark against an >=8 target; all six checks PASS).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from compare_posteriors import flatten_bi, load_stan, kl_and_jsd

# --- design tokens (light mode) -------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STAN = "#2a78d6"     # categorical slot 1
BF = "#eb6834"       # categorical slot 2

BLOCK_TITLES = {
    # A and Q are deterministic functions of the sampled blocks below them, so
    # the same quantity appears twice; the titles say so rather than letting
    # the repeats read as independent estimates.
    "A": "A — selection matrix (assembled from A_diag and A_offdiag)",
    "A_diag": "A_diag — autoregressive terms (= diagonal of A)",
    "A_offdiag": "A_offdiag — cross-lagged terms (= free off-diagonals of A)",
    "Q": "Q — drift covariance (built from Q_sigma)",
    "Q_sigma": "Q_sigma — drift scales (= sqrt of Q diagonal)",
    "b": "b — continuous-time intercepts",
    "eta_anc": "eta_anc — ancestral states",
    "alpha": "alpha — trait intercepts",
    "shape": "shape — gamma dispersion",
    "lambda_free": "lambda_free — factor loadings",
    "cor_R": "cor_R — drift correlation",
}


def hdi(x, prob):
    """Highest-density interval of a 1d sample."""
    s = np.sort(x)
    n = len(s)
    k = max(1, int(np.floor(prob * n)))
    if k >= n:
        return s[0], s[-1]
    widths = s[k:] - s[:n - k]
    i = int(np.argmin(widths))
    return s[i], s[i + k]


def collect(example, root):
    res = os.path.join(root, example, "results")
    bi = flatten_bi(np.load(os.path.join(res, "bf_draws.npz")))
    stan = load_stan(os.path.join(res, "stan_draws.csv"),
                     {k.split("[")[0] for k in bi})
    # Stan writes tree-indexed parameters as name[1,j]; BF as name[j]
    fixed = {}
    for k, v in stan.items():
        alt = k.replace("[1,", "[", 1)
        fixed[alt if ("[1," in k and alt in bi) else k] = v
    return res, fixed, bi


def truth_lookup(example, root):
    """True A/Q/b/eta_anc values, for the synthetic example only."""
    import json
    p = os.path.join(root, example, "data", "true_params.json")
    if not os.path.exists(p):
        return {}
    T = json.load(open(p))
    out = {}
    A = np.array(T["A"]); Q = np.array(T["Q"])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            out[f"A[{i+1},{j+1}]"] = A[i, j]
            out[f"Q[{i+1},{j+1}]"] = Q[i, j]
        out[f"A_diag[{i+1}]"] = A[i, i]
        out[f"b[{i+1}]"] = np.array(T["b"])[i]
        out[f"eta_anc[{i+1}]"] = np.array(T["eta_anc"])[i]
        out[f"Q_sigma[{i+1}]"] = np.sqrt(Q[i, i])

    # A_offdiag holds the free off-diagonals of A in Stan's ticker order
    # (row-major, skipping the diagonal and the entries effects_mat zeroes),
    # so it has known true values too -- without this its panel drew no
    # diamonds while the identical A entries above did.
    sd_path = os.path.join(root, example, "data", "standata.json")
    if os.path.exists(sd_path):
        eff = np.array(json.load(open(sd_path))["effects_mat"])
        t = 0
        for i in range(eff.shape[0]):
            for j in range(eff.shape[1]):
                if i != j and eff[i, j] == 1:
                    t += 1
                    out[f"A_offdiag[{t}]"] = A[i, j]
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def intervals(res, stan, bi, truth, example):
    names = sorted(set(stan) & set(bi))
    blocks = {}
    for n in names:
        blocks.setdefault(n.split("[")[0], []).append(n)
    order = [b for b in BLOCK_TITLES if b in blocks] + \
            [b for b in blocks if b not in BLOCK_TITLES]

    heights = [max(1, len(blocks[b])) for b in order]
    fig, axes = plt.subplots(len(order), 1, figsize=(8.4, 0.34 * sum(heights) + 2.4),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, blk in zip(axes.ravel(), order):
        ps = sorted(blocks[blk])
        style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for row, name in enumerate(ps):
            for draws, col, off in ((stan[name], STAN, 0.17), (bi[name], BF, -0.17)):
                y = row + off
                if np.std(draws) < 1e-12:      # structurally fixed at zero
                    ax.plot([draws.mean()], [y], marker="|", ms=9, color=MUTED,
                            mew=2, zorder=3)
                    continue
                lo89, hi89 = hdi(draws, 0.89)
                lo50, hi50 = hdi(draws, 0.50)
                ax.plot([lo89, hi89], [y, y], color=col, lw=2, solid_capstyle="round",
                        zorder=2)
                ax.plot([lo50, hi50], [y, y], color=col, lw=4.5,
                        solid_capstyle="round", zorder=3)
                ax.plot([draws.mean()], [y], marker="o", ms=5, color=col,
                        mec=SURFACE, mew=1.4, zorder=4)   # 2px surface ring
            if name in truth:
                ax.plot([truth[name]], [row], marker="D", ms=5.5,
                        color=TEXT_PRIMARY, mec=SURFACE, mew=1.2, zorder=5)
        ax.set_yticks(range(len(ps)))
        ax.set_yticklabels(ps, fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylim(-0.7, len(ps) - 0.3)
        ax.invert_yaxis()
        # keep interval ends off the panel edge, where they read as clipped
        ax.margins(x=0.07)
        ax.set_title(BLOCK_TITLES.get(blk, blk), fontsize=9.5, color=TEXT_PRIMARY,
                     loc="left", pad=5)

    handles = [Line2D([], [], color=STAN, lw=4.5, solid_capstyle="round",
                      label="Stan reference"),
               Line2D([], [], color=BF, lw=4.5, solid_capstyle="round",
                      label="BF translation"),
               # zeros come from effects_mat or estimate_correlated_drift=FALSE,
               # and cor_R's diagonal is fixed at 1, so do not name a value
               Line2D([], [], color=MUTED, lw=0, marker="|", ms=9, mew=2,
                      label="not estimated (held fixed)")]
    if truth:
        handles.append(Line2D([], [], color=TEXT_PRIMARY, lw=0, marker="D", ms=5.5,
                              label="true value"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle(f"{example}: posterior mean with 50% and 89% HDI",
                 fontsize=12.5, color=TEXT_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    out = os.path.join(res, "posterior_intervals.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def densities(res, stan, bi, truth, example):
    names = [n for n in sorted(set(stan) & set(bi))
             if np.std(stan[n]) > 1e-12 or np.std(bi[n]) > 1e-12]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.35 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.ravel(), names):
        style(ax)
        s, b = stan[name], bi[name]
        lo = min(s.min(), b.min()); hi = max(s.max(), b.max())
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - pad, hi + pad, 400)
        for draws, col in ((s, STAN), (b, BF)):
            try:
                ax.fill_between(grid, gaussian_kde(draws)(grid), color=col,
                                alpha=0.42, lw=0)
                ax.plot(grid, gaussian_kde(draws)(grid), color=col, lw=2)
            except np.linalg.LinAlgError:
                pass
        if name in truth:
            ax.axvline(truth[name], color=TEXT_PRIMARY, lw=1.4, ls=(0, (3, 2)))
        ax.set_title(name, fontsize=9, color=TEXT_PRIMARY)
        ax.set_yticks([])
        # divergence between the two posteriors, in text tokens rather than
        # series colour so it reads as annotation and not as a third series
        kl, jsd, _ = kl_and_jsd(s, b)
        ax.text(0.5, -0.30, f"KL {kl:.4f}   JSD {jsd:.4f}", fontsize=7.5,
                color=MUTED, ha="center", transform=ax.transAxes)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    handles = [Line2D([], [], color=STAN, lw=3, label="Stan reference"),
               Line2D([], [], color=BF, lw=3, label="BF translation")]
    if truth:
        handles.append(Line2D([], [], color=TEXT_PRIMARY, lw=1.4, ls=(0, (3, 2)),
                              label="true value"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(f"{example}: marginal posterior of every free parameter",
                 fontsize=12.5, color=TEXT_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.975])
    out = os.path.join(res, "posterior_densities.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True)
    ap.add_argument("--root", default="/home/sebastian_sosa/phylo/examples")
    a = ap.parse_args()

    res, stan, bi = collect(a.example, a.root)
    truth = truth_lookup(a.example, a.root)
    print("parameters plotted:", len(set(stan) & set(bi)),
          "| truth overlay:", "yes" if truth else "no (empirical example)")
    print(intervals(res, stan, bi, truth, a.example))
    print(densities(res, stan, bi, truth, a.example))
