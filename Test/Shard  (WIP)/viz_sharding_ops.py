"""Generate one clear figure per sharding case → individual PNGs.

Convention (N = 6 observations, n_devices = 2):
    device 0 = rows 0,1,2   (blue)
    device 1 = rows 3,4,5   (orange)
    grey   = replicated on both devices
    RED    = cross-device communication (allgather / halo / all-to-all) or breakage
    GREEN  = local / safe

Each figure is saved large enough to read on its own. Rerun to regenerate.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

DEV0 = "#9ecae1"
DEV1 = "#fdae6b"
REP  = "#d9d9d9"
BAD  = "#d62728"
GOOD = "#2ca02c"

D6 = [DEV0, DEV0, DEV0, DEV1, DEV1, DEV1]   # 6-row device colouring


def new_ax(title, color, w=7.2, h=3.4, xlim=(0, 12), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=color)
    return fig, ax


def cell(ax, x, y, w, h, color, text="", edge="#333", lw=1.1, fs=9, tcolor="#000"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tcolor)


def hrow(ax, x, y, labels, colors, w=1.0, h=0.9, fs=9, gap=0.1):
    for i, (lab, c) in enumerate(zip(labels, colors)):
        cell(ax, x + i * (w + gap), y, w, h, c, lab, fs=fs)


def vcol(ax, x, y_top, labels, colors, w=0.7, h=0.6, fs=9):
    for i, (lab, c) in enumerate(zip(labels, colors)):
        cell(ax, x, y_top - i * h, w, h, c, lab, fs=fs)


def grid(ax, x, y_top, nrow, ncol, rowcolors, s=0.5, fs=0, edge="#888", lw=0.5):
    for i in range(nrow):
        for j in range(ncol):
            cell(ax, x + j * s, y_top - i * s, s * 0.92, s * 0.92, rowcolors[i],
                 "", edge=edge, lw=lw)


def arrow(ax, p0, p1, color=BAD, lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


def boundary(ax, x, y0, y1, label="device boundary", color="#555", ls="--"):
    ax.plot([x, x], [y0, y1], color=color, lw=1.6, ls=ls)
    if label:
        ax.text(x, y1 + 0.15, label, color=color, fontsize=8, ha="center")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ===========================================================================
# SAFE PATTERNS
# ===========================================================================
def fig_map():
    fig, ax = new_ax("MAP  ✓  elementwise · X @ β · a[group]  (per-row)", GOOD)
    vcol(ax, 1.0, 5.2, [f"x{i}" for i in range(6)], D6)
    ax.text(1.35, 5.95, "X (sharded)", fontsize=9, ha="center")
    cell(ax, 3.3, 3.3, 0.7, 0.7, REP, "β", fs=11)
    ax.text(3.65, 4.2, "β (replicated)", fontsize=8, ha="center")
    vcol(ax, 5.6, 5.2, [f"μ{i}" for i in range(6)], D6)
    ax.text(5.95, 5.95, "μ (sharded)", fontsize=9, ha="center")
    arrow(ax, (2.4, 2.6), (5.4, 2.6), color=GOOD)
    ax.text(8.6, 3.0, "output row i depends\nonly on input row i\n→ no communication",
            fontsize=10, color=GOOD, ha="center")
    save(fig, "op_map.png")


def fig_reduce():
    fig, ax = new_ax("REDUCE  ✓  sum · mean · logsumexp · max  (over axis 0)", GOOD)
    vcol(ax, 1.0, 5.2, [f"ℓ{i}" for i in range(6)], D6)
    ax.text(1.35, 5.95, "per-row log-lik", fontsize=9, ha="center")
    cell(ax, 3.6, 4.2, 1.3, 0.7, DEV0, "Σ dev0", fs=9)
    cell(ax, 3.6, 3.0, 1.3, 0.7, DEV1, "Σ dev1", fs=9)
    arrow(ax, (1.8, 4.6), (3.5, 4.55), color=GOOD)
    arrow(ax, (1.8, 2.8), (3.5, 3.35), color=GOOD)
    cell(ax, 6.6, 3.6, 1.5, 0.7, GOOD, "total Σ", fs=10, tcolor="white")
    arrow(ax, (5.0, 4.55), (6.6, 4.0), color=GOOD)
    arrow(ax, (5.0, 3.35), (6.6, 3.8), color=GOOD)
    ax.text(9.9, 3.9, "associative combine\n→ one cheap all-reduce\nof a SCALAR",
            fontsize=10, color=GOOD, ha="center")
    save(fig, "op_reduce.png")


# ===========================================================================
# FAMILY 1 — cross-position coupling
# ===========================================================================
def fig_lag():
    fig, ax = new_ax("Family 1 ✗  diff / lag   jnp.diff,  y[1:] − ρ·y[:-1]", BAD)
    hrow(ax, 0.6, 3.6, [f"y{i}" for i in range(6)], D6, w=1.1, h=0.9)
    bx = 0.6 + 3 * (1.1 + 0.1) - 0.05
    boundary(ax, bx, 3.2, 4.8)
    ax.text(2.0, 2.9, "(y1,y0) ✓", color=GOOD, fontsize=9, ha="center")
    ax.text(3.4, 2.9, "(y2,y1) ✓", color=GOOD, fontsize=9, ha="center")
    arrow(ax, (bx + 1.0, 3.3), (bx - 0.4, 3.3), color=BAD)
    ax.text(bx + 1.1, 2.9, "(y3,y2): y2 is on device 0", color=BAD, fontsize=9,
            ha="left")
    ax.text(6.0, 1.7, "boundary term → halo exchange → deadlock (vectorized)",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_lag.png")


def fig_cumsum():
    fig, ax = new_ax("Family 1 ✗  cumsum / cumprod  (prefix scan)", BAD)
    hrow(ax, 0.6, 3.6, [f"y{i}" for i in range(6)], D6, w=1.1, h=0.9)
    bx = 0.6 + 3 * (1.1 + 0.1) - 0.05
    boundary(ax, bx, 3.2, 4.8)
    ax.text(2.3, 2.9, "c0..c2 use only dev0 ✓", color=GOOD, fontsize=9, ha="center")
    arrow(ax, (bx - 0.4, 2.6), (bx + 1.0, 2.6), color=BAD)
    ax.text(bx + 1.1, 2.4, "c3 = c2 + y3 needs the\nrunning total from device 0",
            color=BAD, fontsize=9, ha="left")
    ax.text(6.0, 1.3, "prefix dependency carries across the boundary",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_cumsum.png")


def fig_scan():
    fig, ax = new_ax("lax.scan   over-axis ✗   vs   within-row ✓", BAD,
                     h=4.0, ylim=(0, 7))
    # top: scan over the sharded axis
    hrow(ax, 0.6, 5.2, [f"y{i}" for i in range(6)], D6, w=1.0, h=0.8)
    bx = 0.6 + 3 * (1.0 + 0.1) - 0.05
    boundary(ax, bx, 4.9, 6.1, label="")
    arrow(ax, (bx - 0.4, 5.6), (bx + 0.9, 5.6), color=BAD)
    ax.text(7.4, 5.5, "scan(f, init, y): carry\ncrosses boundary  ✗ Family 1",
            color=BAD, fontsize=9, ha="left")
    # bottom: scan within a row, vmapped over axis 0
    hrow(ax, 0.6, 2.4, [f"x{i}" for i in range(6)], D6, w=1.0, h=0.8)
    for i in range(6):
        x0 = 0.6 + i * 1.1 + 0.5
        ax.text(x0, 1.9, "↻", color=GOOD, fontsize=12, ha="center")
    ax.text(7.4, 2.2, "vmap(scan over 100 inner\nsteps): local per row  ✓ map",
            color=GOOD, fontsize=9, ha="left")
    save(fig, "op_scan.png")


def fig_roll_sort():
    fig, ax = new_ax("Family 1 ✗  sort / argsort / roll / FFT  (reorder axis 0)", BAD)
    hrow(ax, 0.6, 4.2, [f"y{i}" for i in range(6)], D6, w=1.0, h=0.8)
    ax.text(0.4, 5.2, "input", fontsize=9)
    # sorted/rolled order: an element from dev1 must move into dev0's block
    order = [5, 0, 3, 1, 4, 2]
    ocols = [D6[k] for k in order]
    hrow(ax, 0.6, 2.2, [f"y{k}" for k in order], ocols, w=1.0, h=0.8)
    ax.text(0.4, 3.2, "after", fontsize=9)
    arrow(ax, (0.6 + 0 * 1.1 + 0.5, 4.1), (0.6 + 1 * 1.1 + 0.5, 3.05), color=BAD)
    arrow(ax, (0.6 + 5 * 1.1 + 0.5, 4.1), (0.6 + 0 * 1.1 + 0.5, 3.05), color=BAD)
    ax.text(8.2, 3.2, "elements move between\ndevices → all-to-all",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_roll_sort.png")


def fig_conv():
    fig, ax = new_ax("Family 1 ✗  convolution  (needs neighbours → halo)", BAD)
    hrow(ax, 0.6, 3.6, [f"y{i}" for i in range(6)], D6, w=1.1, h=0.9)
    bx = 0.6 + 3 * (1.1 + 0.1) - 0.05
    boundary(ax, bx, 3.2, 4.8)
    # window over y2,y3,y4 straddles boundary
    ax.add_patch(plt.Rectangle((0.6 + 2 * 1.2 - 0.05, 3.5), 3 * 1.2, 1.0,
                               fill=False, edgecolor=BAD, lw=2))
    ax.text(6.4, 4.9, "kernel window {y2,y3,y4}", color=BAD, fontsize=9)
    arrow(ax, (bx + 1.0, 3.3), (bx - 0.4, 3.3), color=BAD)
    ax.text(6.0, 2.4, "window at the boundary needs y2 from device 0 → halo exchange",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_conv.png")


# ===========================================================================
# FAMILY 2 — all-to-all / pairwise
# ===========================================================================
def fig_transpose():
    fig, ax = new_ax("Family 2 ✗  transpose · reciprocity   Y + Y.T", BAD)
    grid(ax, 0.8, 5.4, 6, 6, D6, s=0.62)
    ax.text(0.8 + 3 * 0.62, 5.7, "Y (row-sharded)", fontsize=9, ha="center")
    arrow(ax, (2.4, 1.6), (1.2, 2.6), color=BAD)
    ax.text(5.6, 3.0, "Y.T turns rows into columns:\n"
                      "every device needs columns\nfrom BOTH devices → all-to-all",
            color=BAD, fontsize=10, ha="left")
    save(fig, "op_transpose.png")


def fig_outer():
    fig, ax = new_ax("Family 2 ✗  outer product · Gram · distance · kernel · attention",
                     BAD)
    vcol(ax, 0.8, 5.2, [f"x{i}" for i in range(6)], D6, w=0.6, h=0.62, fs=8)
    ax.text(1.1, 5.95, "x", fontsize=9, ha="center")
    grid(ax, 2.2, 5.2, 6, 6, D6, s=0.62)
    ax.text(2.2 + 3 * 0.62, 5.7, "M[i,j] = f(xᵢ, xⱼ)", fontsize=9, ha="center")
    arrow(ax, (3.0, 2.0), (1.4, 2.0), color=BAD)
    ax.text(7.0, 3.2, "row i (on dev0) needs xⱼ\nfor j on dev1 → cross-device",
            color=BAD, fontsize=10, ha="left")
    save(fig, "op_outer.png")


# ===========================================================================
# FAMILY 3 — contraction over the sharded axis
# ===========================================================================
def fig_contraction():
    fig, ax = new_ax("Family 3 ✗  contraction over the sharded axis   W @ X", BAD)
    # X sharded along its axis-0 (the INNER/contracted dim)
    vcol(ax, 1.0, 5.2, [f"x{i}" for i in range(6)], D6, w=0.7, h=0.62)
    ax.text(1.35, 5.95, "X  (axis 0 = inner dim, sharded)", fontsize=8.5, ha="left")
    cell(ax, 4.0, 4.0, 2.0, 0.9, REP, "W (M×6)", fs=9)
    ax.text(5.0, 5.05, "W replicated", fontsize=8.5, ha="center")
    arrow(ax, (1.9, 3.0), (4.0, 3.0), color=BAD)
    ax.text(8.2, 3.4, "the contracted dim is split:\nW(M×6) · X(3) → shape clash,\n"
                      "needs full X → gather",
            color=BAD, fontsize=10, ha="center")
    ax.text(6.0, 1.2, "fix: write as X @ w (sharded on the LEFT) or as a sum-reduction",
            color=GOOD, fontsize=9.5, ha="center")
    save(fig, "op_contraction.png")


# ===========================================================================
# FAMILY 4 — whole-array linear algebra
# ===========================================================================
def fig_linalg():
    fig, ax = new_ax("Family 4 ✗  cholesky · inv · solve · qr · svd · eig · det", BAD)
    grid(ax, 0.8, 4.9, 6, 6, D6, s=0.62)
    ax.text(0.8 + 3 * 0.62, 5.55, "Σ (N×N, row-sharded)", fontsize=9, ha="center")
    # show that a factorization couples a row with all columns
    for j in range(6):
        cell(ax, 0.8 + j * 0.62, 4.9, 0.62 * 0.92, 0.62 * 0.92, DEV0, "",
             edge=BAD, lw=1.6)
    arrow(ax, (2.4, 1.6), (3.6, 2.6), color=BAD)
    ax.text(5.4, 3.2, "factorization couples each row\nwith ALL columns; a row block\n"
                      "(3×6) is not even square\n→ crash / heavy comms",
            color=BAD, fontsize=10, ha="left")
    ax.text(6.0, 0.7, "only fixable with block-diagonal / low-rank / Kronecker structure",
            color="#555", fontsize=9, ha="center")
    save(fig, "op_linalg.png")


# ===========================================================================
# FAMILY 5 — index / segment ops crossing partitions
# ===========================================================================
def fig_segment():
    fig, ax = new_ax("Family 5 ✗  segment_sum · scatter · bincount", BAD)
    hrow(ax, 1.8, 3.8, [f"v{i}" for i in range(6)], D6, w=1.1, h=0.9)
    seg = ["g0", "g0", "g1", "g1", "g2", "g2"]
    hrow(ax, 1.8, 2.6, seg, ["#eee"] * 6, w=1.1, h=0.7, fs=8)
    ax.text(1.6, 4.25, "values", fontsize=9, ha="right")
    ax.text(1.6, 2.95, "segment id", fontsize=9, ha="right")
    bx = 1.8 + 3 * (1.1 + 0.1) - 0.05
    boundary(ax, bx, 2.4, 4.9)
    ax.text(bx, 1.9, "segment g1 spans BOTH devices (rows 2 & 3)", color=BAD,
            fontsize=9.5, ha="center")
    ax.text(6.0, 1.1, "per-group reduce needs a cross-device combine",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_segment.png")


def fig_perm_gather():
    fig, ax = new_ax("Family 5 ✗  permutation gather  X[perm] · unique", BAD)
    hrow(ax, 0.6, 4.2, [f"x{i}" for i in range(6)], D6, w=1.0, h=0.8)
    ax.text(0.4, 5.2, "X", fontsize=9)
    perm = [4, 2, 5, 0, 3, 1]
    pcols = [D6[k] for k in perm]
    hrow(ax, 0.6, 2.2, [f"x{k}" for k in perm], pcols, w=1.0, h=0.8)
    ax.text(0.4, 3.2, "X[perm]", fontsize=9)
    arrow(ax, (0.6 + 4 * 1.1 + 0.5, 4.1), (0.6 + 0 * 1.1 + 0.5, 3.05), color=BAD)
    arrow(ax, (0.6 + 0 * 1.1 + 0.5, 4.1), (0.6 + 3 * 1.1 + 0.5, 3.05), color=BAD)
    ax.text(8.2, 3.2, "arbitrary indices pull rows\nacross devices → all-to-all",
            color=BAD, fontsize=10, ha="center")
    save(fig, "op_perm_gather.png")


# ===========================================================================
# FAMILY 6 — wrong leading axis
# ===========================================================================
def fig_wrong_axis():
    fig, ax = new_ax("Family 6 ✗  wrong leading axis: feature-/time-major (D, N)", BAD)
    rc = [DEV0, DEV0, DEV1, DEV1]
    grid(ax, 0.8, 5.0, 4, 6, rc, s=0.7)
    ax.text(0.8 + 3 * 0.7, 5.3, "X_time (D=4 features × N=6 obs)", fontsize=9,
            ha="center")
    ax.text(0.6, 1.7, "axis 0 = FEATURES, not observations:\n"
                      "each device holds 2 of 4 features → wrong μ, NO crash",
            color=BAD, fontsize=10, ha="left")
    save(fig, "op_wrong_axis.png")


# ===========================================================================
# FIXES
# ===========================================================================
def fig_fix_lag():
    fig, ax = new_ax("FIX ✓  materialize lagged pairs (observed AR)", GOOD)
    tcols = [DEV0, DEV0, DEV0, DEV1, DEV1]
    hrow(ax, 1.2, 4.2, ["y1", "y2", "y3", "y4", "y5"], tcols, w=1.1, h=0.8, fs=8)
    hrow(ax, 1.2, 3.0, ["y0", "y1", "y2", "y3", "y4"], tcols, w=1.1, h=0.8, fs=8)
    ax.text(1.0, 4.55, "y_t", fontsize=9, ha="right")
    ax.text(1.0, 3.35, "y_lag", fontsize=9, ha="right")
    bx = 1.2 + 3 * (1.1 + 0.1) - 0.05
    boundary(ax, bx, 2.8, 5.1)
    ax.text(6.0, 1.9, "eps = y_t − ρ·y_lag is elementwise (map);\n"
                      "each transition carries its own neighbour → no comms",
            color=GOOD, fontsize=10, ha="center")
    save(fig, "op_fix_lag.png")


def fig_fix_srm():
    fig, ax = new_ax("FIX ✓  SRM outer sum  s[:,None] + r[None,:]  (no transpose)",
                     GOOD)
    vcol(ax, 1.0, 5.2, [f"s{i}" for i in range(6)], D6, w=0.6, h=0.62, fs=8)
    ax.text(1.3, 5.95, "s (sharded)", fontsize=9, ha="center")
    for j in range(6):
        cell(ax, 2.2 + j * 0.62, 5.0, 0.58, 0.58, REP, f"r{j}", fs=7)
    ax.text(2.2 + 3 * 0.62, 5.75, "r (replicated, O(N))", fontsize=9, ha="center")
    grid(ax, 2.2, 4.2, 6, 6, D6, s=0.62)
    ax.text(8.4, 3.0, "result (N/d × N) is local:\nsharded column + replicated row\n"
                      "→ no all-to-all, no transpose",
            color=GOOD, fontsize=10, ha="center")
    save(fig, "op_fix_srm.png")


if __name__ == "__main__":
    fig_map(); fig_reduce()
    fig_lag(); fig_cumsum(); fig_scan(); fig_roll_sort(); fig_conv()
    fig_transpose(); fig_outer()
    fig_contraction()
    fig_linalg()
    fig_segment(); fig_perm_gather()
    fig_wrong_axis()
    fig_fix_lag(); fig_fix_srm()
    print("all figures generated")
