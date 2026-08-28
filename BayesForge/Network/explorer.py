"""NetExplorer data-handling port — R ``NetExplorer`` package -> BayesForge style.

Upstream: https://github.com/SebastianSosa/Network-Explorer  (R/Functions.R)
Pinned commit: e4000d3cc2b6d48246f568c5c31b744e75674773

Same job as the R package's data layer: take a **data frame of node
characteristics** + a **square adjacency matrix**, turn them into the
``nodes`` / ``links`` JSON that the bundled d3.js front-end reads, and splice
that into ``patron1.txt`` + tooltip JS + ``patron2.txt`` to emit a standalone
``NetExplorer.html``.

Design choices for the port:
  * node table stays a ``pandas.DataFrame`` (the R ``df``);
  * every matrix operation (matrix -> edge list, min-max scaling, layer
    intra/inter flags) runs on ``jax.numpy`` arrays (the R ``m``);
  * colour ramps use matplotlib's ``LinearSegmentedColormap`` in place of
    ``grDevices::colorRampPalette``.

R function  ->  method
  mat.to.edgl            -> NetExplorer.mat_to_edgl
  df.col.findId          -> NetExplorer._col_id
  colorize              -> NetExplorer._colorize
  shape                 -> NetExplorer._shape_codes
  vis.net.format.att    -> NetExplorer.format_att
  vis.net               -> NetExplorer.vis_net / NetExplorer.__call__

Reached from the BF object as ``m.net.viz`` (see BayesForge.Network.Net):
``m.net.viz(df, adj, col_id="id", col_size="strength", ...)`` renders the
network; ``m.net.viz.mat_to_edgl(adj)`` etc. for the individual helpers.
"""

from __future__ import annotations

import base64
import html as _html
import mimetypes
import os
import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex

from BayesForge.Network import layouts as _lay


def _categorical_hex(n: int) -> list[str]:
    """``n`` visually distinct hex colours for chord groups."""
    try:
        import matplotlib as _mpl

        cmap = _mpl.colormaps["tab10" if n <= 10 else "tab20" if n <= 20 else "hsv"]
        if n <= 20:
            return [to_hex(cmap(i)) for i in range(n)]
        return [to_hex(cmap(i / max(n, 1))) for i in range(n)]
    except Exception:
        cycle = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                 "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
        return [cycle[i % len(cycle)] for i in range(n)]

_ASSETS = Path(__file__).with_name("assets") / "netexplorer"


def _in_notebook() -> bool:
    """True inside an IPython/Jupyter kernel with a rich display frontend."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower() or "WSL" in platform.uname().release


def _open_in_browser(path: Path) -> bool:
    """Open a local file in the OS default browser. Handles WSL by handing the
    path to the Windows shell."""
    path = path.resolve()
    if _is_wsl():
        try:
            win = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", win],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass
    try:
        return webbrowser.open(path.as_uri())
    except Exception:
        return False


def _data_uri(p: Path) -> str:
    mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


class NetworkView:
    """Return value of :meth:`NetExplorer.vis_net`.

    * as the last expression in a Jupyter cell it renders the network inline
      (``_repr_html_`` -> ``<iframe srcdoc>`` with the self-contained HTML);
    * ``.open()`` (or ``vis_net(..., browser=True)``) opens it in the OS
      browser;
    * ``.path`` is the written ``.html`` file.
    """

    def __init__(self, path, html: str, height: int = 760):
        self.path = Path(path)
        self.html = html
        self.height = height

    def open(self) -> "NetworkView":
        _open_in_browser(self.path)
        return self

    def _repr_html_(self) -> str:
        srcdoc = _html.escape(self.html, quote=True)
        return (
            f'<iframe srcdoc="{srcdoc}" '
            f'style="width:100%;height:{self.height}px;border:1px solid #bbb;'
            f'border-radius:6px;" sandbox="allow-scripts"></iframe>'
        )

    def __repr__(self) -> str:
        return f"NetworkView(path={self.path!s}, {len(self.html)} bytes)"

# d3.symbols order in the bundled d3 build -> integer code the front-end wants.
_SHAPE_CODE = {
    "circle": 0,
    "cross": 1,
    "losange": 2,  # diamond
    "diamond": 2,
    "rectangle": 3,  # square
    "square": 3,
    "star": 4,
    "triangle": 5,
    "y": 6,  # wye
    "wye": 6,
}


class NetExplorer:
    """Build the NetExplorer visualisation HTML from a node DataFrame + a JAX
    adjacency matrix.

    Parameters
    ----------
    ids
        Optional node labels. If ``None`` they are taken from ``col_id`` at
        :meth:`vis_net` time, and failing that generated as ``n1..nN`` from the
        matrix size (mirrors the R behaviour of falling back to
        ``colnames(m)``).
    assets_dir
        Folder holding ``d3.min.js``, ``d3-tip.js``, ``style.css``,
        ``patron1.txt``, ``patron2.txt``, ``logo.jpg``, ``save.png``.
        Defaults to the ``assets/`` folder next to this file.
    """

    def __init__(self, ids: Sequence | None = None, assets_dir: str | os.PathLike | None = None):
        self.ids = None if ids is None else [str(x) for x in ids]
        self.assets_dir = Path(assets_dir) if assets_dir is not None else _ASSETS
        missing = [
            f
            for f in ("patron1.txt", "patron2.txt", "d3.min.js", "d3-tip.js", "style.css")
            if not (self.assets_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(f"assets missing in {self.assets_dir}: {missing}")

    # ------------------------------------------------------------------ #
    # matrix -> edge list   (R: mat.to.edgl)
    # ------------------------------------------------------------------ #
    def mat_to_edgl(self, M, sym: bool = False, erase_diag: bool = True) -> pd.DataFrame:
        """Square adjacency matrix -> ``from`` / ``to`` / ``weight`` DataFrame.

        Parameters
        ----------
        M
            ``(N, N)`` JAX array (or anything ``jnp.asarray`` accepts).
        sym
            If ``True`` keep the lower triangle only (undirected).
        erase_diag
            Drop self-loops.

        The directed path reproduces R's ``as.vector(M)`` column-major
        unrolling exactly: edge ``k`` is ``(row = k % N, col = k // N)``.
        """
        M = jnp.asarray(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be a square 2D array")
        N = int(M.shape[0])
        ids = self._resolve_ids(N)

        if sym:
            k = 0 if erase_diag else -1  # tril offset: -1 excludes diag, 0 keeps it
            rows, cols = jnp.tril_indices(N, k=-1 if erase_diag else 0)
            weight = np.asarray(M[rows, cols])
            rows, cols = np.asarray(rows), np.asarray(cols)
            frm = [ids[i] for i in cols]  # R: colnames(M)[tmp[,1]] (col index first)
            to = [ids[i] for i in rows]
        else:
            idx = jnp.arange(N * N)
            from_i = np.asarray(idx % N)
            to_i = np.asarray(idx // N)
            weight = np.asarray(M.T.reshape(-1))  # column-major == R as.vector(M)
            keep = np.ones(N * N, dtype=bool)
            if erase_diag:
                keep = from_i != to_i
            from_i, to_i, weight = from_i[keep], to_i[keep], weight[keep]
            frm = [ids[i] for i in from_i]
            to = [ids[i] for i in to_i]

        return pd.DataFrame({"from": frm, "to": to, "weight": np.asarray(weight, dtype=float)})

    # ------------------------------------------------------------------ #
    # helpers  (R: df.col.findId / colorize / shape)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _col_id(df: pd.DataFrame, label) -> str:
        """Resolve a column name or 0-based index to a column name."""
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
            if label < 0 or label >= df.shape[1]:
                raise IndexError(f"column index {label} out of bounds")
            return df.columns[int(label)]
        if label in df.columns:
            return label
        raise KeyError(f"'{label}' does not match any column of df: {list(df.columns)}")

    @staticmethod
    def _colorize(df: pd.DataFrame, col: str, colors: Sequence[str], new_col: str) -> pd.DataFrame:
        """Append a gradient hex column keyed on ``col``.

        Like R ``colorize``: one ramp stop per *distinct* value of ``col``,
        assigned by ascending rank (smallest value -> ``colors[0]``). Unlike R,
        the frame is **not** re-sorted — rows stay aligned to the adjacency
        matrix; the ramp is built from a value->hex map so order is irrelevant.
        """
        d = df.copy()
        try:
            levels = sorted(pd.unique(d[col]).tolist())
        except TypeError:
            levels = sorted(pd.unique(d[col]).tolist(), key=str)
        cmap = LinearSegmentedColormap.from_list("netexp", list(colors))
        if len(levels) == 1:
            ramp = {levels[0]: to_hex(cmap(0.0))}
        else:
            ramp = {lv: to_hex(cmap(i / (len(levels) - 1))) for i, lv in enumerate(levels)}
        d[new_col] = d[col].map(ramp)
        return d

    @staticmethod
    def _shape_codes(vec, chars: Sequence[str]) -> np.ndarray:
        """Categorical vector -> d3 symbol codes via a per-category shape name."""
        vec = pd.Series(vec)
        cats = sorted(pd.unique(vec).tolist(), key=str)
        if len(cats) > 7:
            raise ValueError("at most 7 node shapes are available")
        if len(cats) != len(chars):
            raise ValueError(
                f"`shapes` has {len(chars)} entries but the column has {len(cats)} categories"
            )
        name_by_cat = dict(zip(cats, chars))
        try:
            return np.array([_SHAPE_CODE[name_by_cat[v].lower()] for v in vec], dtype=int)
        except KeyError as e:
            raise ValueError(f"unsupported shape name {e!s}; pick from {sorted(_SHAPE_CODE)}")

    # ------------------------------------------------------------------ #
    # node attribute formatting  (R: vis.net.format.att)
    # ------------------------------------------------------------------ #
    def format_att(
        self,
        df: pd.DataFrame,
        col_id=None,
        col_size=None,
        color: Sequence[str] | None = None,
        col_color=None,
        shapes: Sequence[str] | None = None,
        col_shape=None,
        strokeCol: Sequence[str] | None = None,
        col_strokeCol=None,
        col_stroke=None,
        node_opacity=None,
    ):
        """Normalise an arbitrary node table into the fixed column set the
        front-end reads: ``id, size, color, strokeCol, strokeW, shape,
        opacity`` plus the ``*Value`` columns used for tooltips.

        Returns ``(df2, ori)`` where ``ori`` is the list of the seven source
        column names ``[id, size, color, strokeCol, stroke, shape, opacity]``
        (``None`` where not supplied), used later for tooltip labels.
        """
        d = df.copy()
        if col_id is None and any(x is not None for x in (color, strokeCol, col_shape, node_opacity)):
            raise ValueError("col_id cannot be None when other styling arguments are set")

        # --- opacity -------------------------------------------------- #
        if node_opacity is not None:
            c = self._col_id(d, node_opacity)
            if not np.issubdtype(d[c].dtype, np.number):
                raise TypeError("node_opacity column must be numeric")
            ori_opacity = c
            v = jnp.asarray(d[c].to_numpy(dtype=float))
            v = (v - v.min()) / (v.max() - v.min())
            v = np.array(v)  # writable copy off the JAX buffer
            if len(v):
                v[np.argmin(v)] = 1e-3
            d["opacity"] = v
        else:
            ori_opacity = None
            d["opacity"] = 1.0

        # --- size --------------------------------------------------- #
        if col_size is not None:
            c = self._col_id(d, col_size)
            if not np.issubdtype(d[c].dtype, np.number):
                raise TypeError("col_size column must be numeric")
            ori_size = c
            d["size"] = d[c].astype(float)
        else:
            ori_size = None
            d["size"] = 1.0

        # --- id ---------------------------------------------------- #
        if col_id is not None:
            c = self._col_id(d, col_id)
            ori_id = c
            d["id"] = d[c].astype(str)
        else:
            ori_id = None
            d["id"] = [str(i + 1) for i in range(len(d))]

        # --- shape ------------------------------------------------- #
        if col_shape is not None:
            c = self._col_id(d, col_shape)
            ori_shape = c
            if shapes is not None:
                d["shape"] = self._shape_codes(d[c], shapes)
            else:
                d["shape"] = pd.factorize(d[c], sort=True)[0]
        else:
            ori_shape = None
            d["shape"] = 0

        # --- stroke width + stroke colour ------------------------- #
        if col_stroke is not None:
            c = self._col_id(d, col_stroke)
            if not np.issubdtype(d[c].dtype, np.number):
                raise TypeError("col_stroke column must be numeric")
            ori_stroke = c
            d["strokeW"] = d[c].astype(float)
            if col_strokeCol is not None:
                if strokeCol is None or len(strokeCol) != 2:
                    raise ValueError("`strokeCol` must be a 2-colour gradient")
                cc = self._col_id(d, col_strokeCol)
                ori_strokeCol = cc
                d = self._colorize(d, cc, strokeCol, new_col="strokeCol")
            else:
                ori_strokeCol = None
                d["strokeCol"] = "white"
        else:
            ori_stroke = None
            ori_strokeCol = None
            d["strokeW"] = 0.0
            d["strokeCol"] = np.nan

        # --- node colour ----------------------------------------- #
        if col_color is not None:
            if color is None or len(color) != 2:
                raise ValueError("`color` must be a 2-colour gradient")
            cc = self._col_id(d, col_color)
            ori_color = cc
            d = self._colorize(d, cc, color, new_col="color")
        else:
            ori_color = None
            d["color"] = "black"

        # --- tooltip value columns ------------------------------- #
        for vcol, src in (
            ("sizeValue", ori_size),
            ("colorValue", ori_color),
            ("strokeColValue", ori_strokeCol),
            ("shapeValue", ori_shape),
        ):
            if src is not None:
                d[vcol] = d[src]
        d["strokeWValue"] = d["strokeW"]
        if ori_opacity is not None:
            d["opacityValue"] = d[ori_opacity]

        ori = [ori_id, ori_size, ori_color, ori_strokeCol, ori_stroke, ori_shape, ori_opacity]
        return d, ori

    # ------------------------------------------------------------------ #
    # full pipeline  (R: vis.net)
    # ------------------------------------------------------------------ #
    def vis_net(
        self,
        df: pd.DataFrame | None = None,
        m=None,
        col_id=None,
        col_size=None,
        color: Sequence[str] = ("black", "white"),
        col_color=None,
        col_shape=None,
        shapes: Sequence[str] | None = None,
        strokeCol: Sequence[str] = ("white", "black"),
        col_strokeCol=None,
        col_stroke=None,
        layers=None,
        node_opacity=None,
        link_opacity: bool = False,
        background: str = "grey",
        layout: str = "force",
        layout_col=None,
        layout_x=None,
        layout_y=None,
        directed: bool = True,
        out_dir: str | os.PathLike = "out",
        filename: str = "NetExplorer.html",
        inline: bool = True,
        browser: bool | None = None,
        height: int = 760,
    ) -> "NetworkView":
        """Write ``NetExplorer.html`` and return a :class:`NetworkView`.

        ``df`` is the node characteristics table, ``m`` a ``(N, N)`` JAX
        adjacency array. All styling arguments mirror the R ``vis.net`` names.

        Parameters
        ----------
        df
            Node characteristics. If ``None`` a placeholder frame of one
            all-ones column is built and ``col_id`` defaults to ``0`` — so
            ``m.net.viz(m=adj)`` works with no node table. (Node ids then fall
            back to ``n1..nN`` since that column is not unique.)
        inline
            Embed d3 / CSS / images into the HTML so the single file is
            portable (needed for Jupyter inline rendering). When ``False`` the
            assets are copied next to the file instead.
        browser
            Open the file in the OS default browser (WSL-aware). Default:
            ``True`` outside a notebook, ``False`` inside one (the cell renders
            it inline anyway — call ``.open()`` on the result to also pop a
            browser).
        layout
            Initial layout. ``force`` / ``circle`` / ``linear`` / ``multilayer``
            run live in the browser (unchanged). ``clustered`` / ``spectral`` /
            ``mds`` / ``radial`` / ``arc`` / ``layered`` / ``geo`` are computed
            here (see :mod:`BayesForge.Network.layouts`) and the page pins nodes
            to the result; ``chord`` draws a d3 ribbon chord of between-group
            flows. The dropdown offers the four live layouts plus whichever one
            you picked here.
        layout_col
            Column driving the chosen layout: the nodal metric for ``radial``
            (default: degree) and ``arc``, the grouping for ``clustered`` and
            ``chord`` (required for ``chord``).
        layout_x, layout_y
            Columns of fixed coordinates for ``layout="geo"`` (e.g. lon / lat).
        directed
            For ``layout="layered"``: treat the matrix as directed and use
            topological generations when it is acyclic.
        height
            Inline iframe height in px for the notebook view.
        """
        if browser is None:
            browser = not _in_notebook()
        if m is None:
            raise ValueError("vis_net needs an adjacency matrix `m`")
        m = jnp.asarray(m)
        N = int(m.shape[0])

        if df is None:
            df = pd.DataFrame(np.ones((N, 1)))
            if col_id is None:
                col_id = 0

        d, ori = self.format_att(
            df,
            col_id=col_id,
            col_size=col_size,
            color=color,
            col_color=col_color,
            shapes=shapes,
            col_shape=col_shape,
            strokeCol=strokeCol,
            col_strokeCol=col_strokeCol,
            col_stroke=col_stroke,
            node_opacity=node_opacity,
        )
        # node labels: prefer col_id, then ctor ids, then an n1..nN fallback.
        # Set self.ids so mat_to_edgl below labels edges with the same names.
        if ori[0] is not None:
            self.ids = d["id"].astype(str).tolist()
        else:
            self._resolve_ids(N)
            d["id"] = list(self.ids)
        # ids must be unique for the d3 force graph — a constant/placeholder id
        # column (e.g. df=None -> ones) collapses every node onto one point and
        # hides the links. Fall back to positional labels.
        if pd.Index(d["id"]).duplicated().any():
            import warnings as _warnings

            _warnings.warn(
                "node id column is not unique; using positional ids n1..nN",
                stacklevel=2,
            )
            d["id"] = [f"n{i + 1}" for i in range(len(d))]
            self.ids = list(d["id"])

        # layer index per node
        if layers is not None:
            lc = self._col_id(d, layers)
            d["layers"] = pd.factorize(d[lc], sort=True)[0] + 1
        else:
            d["layers"] = 1

        # edge list from the matrix (JAX), drop zero links
        edgl = self.mat_to_edgl(m)
        edgl = edgl[edgl["weight"] != 0].reset_index(drop=True)

        # link opacity via min-max on the JAX weight vector
        if link_opacity and len(edgl):
            w = jnp.asarray(edgl["weight"].to_numpy())
            lo = (w - w.min()) / (w.max() - w.min())
            lo = np.array(lo)  # writable copy off the JAX buffer
            lo[np.argmin(lo)] = 1e-4
            edgl["lOpacity"] = lo
        else:
            edgl["lOpacity"] = 1.0

        # carry the source node's colour onto each edge (R merges 'color')
        color_by_id = dict(zip(d["id"], d["color"]))
        edgl["colorL"] = edgl["from"].map(color_by_id)

        # intra / inter layer flags (JAX equality on the layer vectors)
        if layers is not None:
            layer_by_id = dict(zip(d["id"], d["layers"]))
            src_layer = jnp.asarray([layer_by_id[x] for x in edgl["from"]])
            tgt_layer = jnp.asarray([layer_by_id[x] for x in edgl["to"]])
            same = np.asarray(src_layer == tgt_layer)
            edgl["intralayer"] = np.where(same, 1.0, np.nan)
            edgl["interlayer"] = np.where(same, np.nan, 1.0)
        else:
            edgl["intralayer"] = np.nan
            edgl["interlayer"] = np.nan

        # layouts: precompute coordinates for every applicable non-live layout
        # so the GUI dropdown can switch between them client-side. Big graphs
        # (N > 800) only get the one that was asked for.
        layout = (layout or "force").lower()
        adj_np = np.asarray(m, dtype=float)
        grp_vals = (
            d[self._col_id(d, layout_col)].to_numpy() if layout_col is not None else None
        )
        geo_xy = None
        if layout_x is not None and layout_y is not None:
            geo_xy = (
                d[self._col_id(d, layout_x)].to_numpy(),
                d[self._col_id(d, layout_y)].to_numpy(),
            )

        wanted = _lay.PINNED if N <= 800 else (
            (layout,) if layout in _lay.PINNED else ()
        )
        all_pos: dict[str, tuple] = {}
        for name in wanted:
            try:
                if name == "geo":
                    if geo_xy is None:
                        continue
                    lx, ly = _lay.compute("geo", adj_np, x=geo_xy[0], y=geo_xy[1])
                elif name == "clustered":
                    lx, ly = _lay.compute("clustered", adj_np, groups=grp_vals)
                else:  # spectral / mds / radial / arc / layered — self-sufficient
                    lx, ly = _lay.compute(name, adj_np, directed=directed)
                all_pos[name] = (np.asarray(lx, float), np.asarray(ly, float))
            except Exception:
                continue
        if layout in _lay.PINNED and layout not in all_pos:
            raise ValueError(
                f"layout={layout!r} could not be computed"
                + (" (needs layout_x / layout_y)" if layout == "geo" else "")
            )

        chord_payload = None
        if grp_vals is not None:
            mat, labels = _lay.chord_matrix(grp_vals, adj_np)
            chord_payload = {
                "matrix": mat.tolist(),
                "labels": [str(x) for x in labels],
                "colors": _categorical_hex(len(labels)),
            }
        elif layout == "chord":
            raise ValueError("layout='chord' needs layout_col (the node grouping)")

        # assemble + write
        html = self._assemble_html(
            d, edgl, ori, layout=layout, chord=chord_payload, all_pos=all_pos
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if inline:
            html = self._inline_assets(html)
        else:
            for f in ("d3.min.js", "d3-tip.js", "style.css", "logo.jpg", "save.png"):
                src = self.assets_dir / f
                if src.exists():
                    shutil.copy(src, out_dir / f)
        target = out_dir / filename
        target.write_text(html, encoding="utf-8")

        view = NetworkView(target, html, height=height)
        if browser:
            _open_in_browser(target)
        # In a notebook the returned view renders itself once via _repr_html_
        # when it is the last expression in the cell. Do NOT also display() it
        # here — that draws the network twice. If the call is assigned
        # (`v = m.net.viz(...)`), evaluate `v` on its own line to show it.
        return view

    # `m.net.viz(df, adj, ...)` is the common path -> alias __call__ to vis_net
    __call__ = vis_net

    def _inline_assets(self, html: str) -> str:
        """Fold ``d3.min.js`` / ``d3-tip.js`` / ``style.css`` / images into the
        HTML so it is a single portable file."""
        d3 = (self.assets_dir / "d3.min.js").read_text(encoding="utf-8")
        tip = (self.assets_dir / "d3-tip.js").read_text(encoding="utf-8")
        css = (self.assets_dir / "style.css").read_text(encoding="utf-8")
        html = html.replace(
            '<link rel="stylesheet" type="text/css" href="style.css">',
            f"<style>\n{css}\n</style>",
        )
        html = html.replace("<script src='d3.min.js'></script>", f"<script>\n{d3}\n</script>")
        html = html.replace("<script src='d3-tip.js'></script>", f"<script>\n{tip}\n</script>")
        for img in ("logo.jpg", "save.png"):
            p = self.assets_dir / img
            if p.exists():
                html = html.replace(f'src="{img}"', f'src="{_data_uri(p)}"')
        return html

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _resolve_ids(self, N: int) -> list[str]:
        if self.ids is None:
            self.ids = [f"n{i + 1}" for i in range(N)]
        elif len(self.ids) != N:
            raise ValueError(f"ids has length {len(self.ids)} but matrix is {N}x{N}")
        return self.ids

    @staticmethod
    def _js_val(v) -> str:
        """Python scalar -> JS literal (numbers bare, NaN -> NaN, else quoted)."""
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            f = float(v)
            return "NaN" if np.isnan(f) else repr(int(f)) if f.is_integer() else repr(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NaN"
        return "'" + str(v).replace("'", "\\'") + "'"

    def _tooltip_js(self, ori) -> str:
        """The ``tooltip.html(...)`` argument list appended after patron1."""
        names = dict(zip(("id", "size", "color", "strokeCol", "stroke", "shape", "opacity"), ori))
        s = "\n'id : '+ d.id "
        if names["size"] is not None:
            s += f"+ '<p/>{names['size']} (size) : '+ d.sizeValue "
        if names["color"] is not None:
            s += f"+ '<p/>{names['color']} (color) : '+ d.colorValue "
        if names["strokeCol"] is not None:
            s += f"+ '<p/>{names['strokeCol']} (stroke color) : '+ d.strokeColValue "
        if names["stroke"] is not None:
            s += f"+ '<p/>{names['stroke']} (stroke width) : '+ d.strokeWValue "
        if names["shape"] is not None:
            s += f"+ '<p/>{names['shape']} (shape) : '+ d.shapeValue \n"
        s += (
            ").style( 'left' ,(d3.event.pageX) + 'px')\n"
            ".style( 'top' ,(d3.event.pageY) + 'px');\n})\n"
        )
        return s

    def _nodes_json(self, d: pd.DataFrame, ori, all_pos: dict | None = None) -> str:
        has = dict(zip(("id", "size", "color", "strokeCol", "stroke", "shape", "opacity"), ori))
        all_pos = all_pos or {}
        rows = []
        for i, (_, r) in enumerate(d.iterrows()):
            parts = [
                f"'id':{self._js_val(r['id'])}",
                f"'size':{self._js_val(r['size'])}",
                f"'color':{self._js_val(r['color'])}",
                f"'strokeCol':{self._js_val(r['strokeCol'])}",
                f"'strokeW':{self._js_val(r['strokeW'])}",
                f"'shape':{self._js_val(r['shape'])}",
                f"'opacity':{self._js_val(r['opacity'])}",
                f"'layers':{self._js_val(r['layers'])}",
            ]
            if has["size"] is not None:
                parts.append(f"'sizeValue':{self._js_val(r['sizeValue'])}")
            if has["color"] is not None:
                parts.append(f"'colorValue':{self._js_val(r['colorValue'])}")
            if has["strokeCol"] is not None:
                parts.append(f"'strokeColValue':{self._js_val(r['strokeColValue'])}")
            if has["stroke"] is not None:
                parts.append(f"'strokeWValue':{self._js_val(r['strokeWValue'])}")
            if has["shape"] is not None:
                parts.append(f"'shapeValue':{self._js_val(r['shapeValue'])}")
            if has["opacity"] is not None:
                parts.append(f"'opacityValue':{self._js_val(r['opacityValue'])}")
            if all_pos:
                pos = ",".join(
                    f"'{name}':[{float(lx[i]):.5f},{float(ly[i]):.5f}]"
                    for name, (lx, ly) in all_pos.items()
                    if np.isfinite(lx[i]) and np.isfinite(ly[i])
                )
                parts.append("'pos':{" + pos + "}")
            rows.append("{" + ",".join(parts) + "},")
        return "\n".join(rows)

    def _links_json(self, edgl: pd.DataFrame) -> str:
        rows = []
        for _, e in edgl.iterrows():
            rows.append(
                "{"
                f"'source':{self._js_val(e['from'])},"
                f"'target':{self._js_val(e['to'])},"
                f"'colorL':{self._js_val(e['colorL'])},"
                f"'lOpacity':{self._js_val(e['lOpacity'])},"
                f"'weigth':{self._js_val(e['weight'])},"
                f"'intralayer':{self._js_val(e['intralayer'])},"
                f"'interlayer':{self._js_val(e['interlayer'])}"
                "},"
            )
        return "\n".join(rows)

    def _assemble_html(
        self,
        d: pd.DataFrame,
        edgl: pd.DataFrame,
        ori,
        layout: str = "force",
        chord: dict | None = None,
        all_pos: dict | None = None,
    ) -> str:
        import json as _json

        p1 = (self.assets_dir / "patron1.txt").read_text(encoding="utf-8")
        p2 = (self.assets_dir / "patron2.txt").read_text(encoding="utf-8")
        p2 = self._inject_layouts(
            p2, layout, computed=list((all_pos or {}).keys()), chord=chord
        )
        chord_js = (
            f"json['chord'] = {_json.dumps(chord)};\n" if chord is not None else ""
        )
        return (
            p1
            + self._tooltip_js(ori)
            + p2
            + "\n           function getData() {\n   let json = { 'nodes':[\n"
            + self._nodes_json(d, ori, all_pos)
            + "\n],\n'links':[\n"
            + self._links_json(edgl)
            + "\n]}\n"
            + chord_js
            + "return json;\n}\n</script>\n"
        )

    _LIVE_LAYOUTS = ("force", "circle", "linear", "multilayer")

    @staticmethod
    def _lay_label(name: str) -> str:
        return "MDS" if name == "mds" else name.capitalize()

    def _inject_layouts(self, p2: str, layout: str, computed: list[str], chord: dict | None) -> str:
        """Extend the front-end: put every precomputed layout (and Chord, when a
        payload is present) in the dropdown; switch between them client-side."""
        want = ["Force", "Circle", "Linear", "Multilayer"]
        want += [self._lay_label(n) for n in computed if n not in self._LIVE_LAYOUTS]
        if chord is not None:
            want.append("Chord")
        initial = (
            layout.capitalize() if layout in self._LIVE_LAYOUTS
            else "Chord" if layout == "chord"
            else self._lay_label(layout)
        )
        if initial not in want:
            initial = "Force"

        p2 = p2.replace("var Layout = 'Force'", f"var Layout = '{initial}'", 1)
        p2 = p2.replace(
            'layouts = ["Force", "Circle", "Linear", "Multilayer"]',
            "layouts = " + repr(want).replace("'", '"'),
            1,
        )

        # pin nodes to whichever precomputed layout is selected + arc link paths
        pin_js = """
    {
      // map pos[layout] in [0,1] onto a centred square so aspect is preserved
      var _pk = Layout.toLowerCase(),
          _navW = 250, _pad = 40,
          _S = Math.min(width - _navW, height) - 2 * _pad,
          _ox = _navW + (width - _navW - _S) / 2,
          _oy = (height - _S) / 2;
      graph.nodes.forEach(function(d) {
        var p = d.pos && d.pos[_pk];
        if (p) { d.x = _ox + p[0] * _S; d.y = _oy + p[1] * _S; d.fx = d.x; d.fy = d.y; }
      });
    }
    if (Layout === 'Chord') { drawChordOnce(); }
    else { d3.select('#chordG').style('display', 'none');
           link.style('display', null); node.style('display', null); texts.style('display', null); }
"""
        p2 = p2.replace("    if(Layout == 'Circle'){", pin_js + "\n    if(Layout == 'Circle'){", 1)

        arc_link = """      // Curve (semicircular arcs under the Arc layout)
      link.attr('d', function(d) {
        if (Layout === 'Arc') {
          var r = Math.abs(d.target.x - d.source.x) / 2,
              y = Math.max(d.source.y, d.target.y),
              sweep = d.source.x < d.target.x ? 1 : 0;
          return 'M' + d.source.x + ',' + y + ' A' + r + ',' + r + ' 0 0,' + sweep + ' ' + d.target.x + ',' + y;
        }
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy)
        return 'M' + d.source.x + ',' + d.source.y + 'A' + dr + ',' + dr + ' 0 0,1 ' + d.target.x + ',' + d.target.y;
      });"""
        old_link = """      // Curve
      link.attr('d', function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy)
        return 'M' + d.source.x + ',' + d.source.y + 'A' + dr + ',' + dr + ' 0 0,1 ' + d.target.x + ',' + d.target.y;
      });"""
        p2 = p2.replace(old_link, arc_link, 1)

        # show the initial layout as the selected <option>
        p2 = p2.replace(
            '.attr("value", function (d) { return d; }) // corresponding value returned by the button',
            '.attr("value", function (d) { return d; });\n'
            '      d3.select("#dataviz_builtWithD3 select").property("value", Layout);',
            1,
        )

        # dropdown: clear pins when returning to a live layout; accept the extra
        old_handler = """          if(selectedLayout == 'Circle'){Layout = 'Circle';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Linear'){Layout = 'Linear';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Force'){Layout = 'Force2';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Multilayer'){Layout = 'Multilayer';simulation.alpha(0.5).restart();}"""
        new_handler = """          if(["Circle","Linear","Multilayer","Force"].indexOf(selectedLayout) !== -1){
            graph.nodes.forEach(function(d){ d.fx = null; d.fy = null; });
          }
          if(selectedLayout == 'Circle'){Layout = 'Circle';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Linear'){Layout = 'Linear';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Force'){Layout = 'Force2';simulation.alpha(0.5).restart();}
          if(selectedLayout == 'Multilayer'){Layout = 'Multilayer';simulation.alpha(0.5).restart();}
          if(["Circle","Linear","Multilayer","Force"].indexOf(selectedLayout) === -1){
            Layout = selectedLayout; simulation.alpha(0.5).restart();
          }"""
        p2 = p2.replace(old_handler, new_handler, 1)

        # chord helper (no-op unless a chord payload was emitted)
        chord_fn = """
    function drawChordOnce() {
      link.style('display','none'); node.style('display','none'); texts.style('display','none');
      if (window._chordDrawn || !graph.chord || typeof d3.chord !== 'function') return;
      window._chordDrawn = true;
      var cd = graph.chord, W = Math.min(width, height), R = W/2 - 70;
      var g = d3.select('svg').append('g').attr('id','chordG')
        .attr('transform','translate(' + (width/2) + ',' + (height/2) + ')');
      var chords = d3.chord().padAngle(0.045).sortSubgroups(d3.descending)(cd.matrix);
      var arcGen = d3.arc().innerRadius(R).outerRadius(R + 16);
      var ribGen = d3.ribbon().radius(R);
      var col = function(i){ return cd.colors[i % cd.colors.length]; };
      g.append('g').selectAll('path').data(chords.groups).enter().append('path')
        .attr('d', arcGen).style('fill', function(d){ return col(d.index); }).style('stroke','#333');
      g.append('g').style('opacity',0.72).selectAll('path').data(chords).enter().append('path')
        .attr('d', ribGen).style('fill', function(d){ return col(d.source.index); }).style('stroke','#333');
      g.append('g').selectAll('text').data(chords.groups).enter().append('text')
        .each(function(d){ d.a = (d.startAngle + d.endAngle) / 2; })
        .attr('dy','0.35em')
        .attr('transform', function(d){
          return 'rotate(' + (d.a * 180 / Math.PI - 90) + ') translate(' + (R + 22) + ')'
            + (d.a > Math.PI ? ' rotate(180)' : ''); })
        .attr('text-anchor', function(d){ return d.a > Math.PI ? 'end' : null; })
        .text(function(d){ return cd.labels[d.index]; })
        .style('font','12px sans-serif').style('fill','#111');
    }
"""
        p2 = p2.replace("var Layout = '", chord_fn + "\nvar Layout = '", 1)
        return p2
