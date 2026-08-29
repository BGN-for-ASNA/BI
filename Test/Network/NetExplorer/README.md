# NetExplorer — data-handling port (R → BayesForge)

Ports the **data layer** of the R package
[`NetExplorer`](https://github.com/SebastianSosa/Network-Explorer)
(`R/Functions.R`, pinned commit `e4000d3`) to a BayesForge-style Python class
that runs on **JAX arrays + pandas DataFrames** and emits the same standalone
`NetExplorer.html` for interactive d3.js network visualisation.

The implementation ships in the package as
**`BayesForge/Network/explorer.py`** (`class NetExplorer`, `class
NetworkView`) with the d3 front-end vendored under
`BayesForge/Network/assets/netexplorer/`. `d3.min.js`, `d3-tip.js`, `logo.jpg`,
`save.png` and the d3 logic in `patron1.txt` / `patron2.txt` are from upstream
`inst/www/`; `style.css` was replaced with a modern responsive one and
`patron1.txt` gained a ☰ toggle button. The Python side *builds the
`nodes`/`links` payload, precomputes the extra layouts, and splices everything
into the template.*

Reach it from a BF object as **`m.net.viz`** (and `m.net2.viz`):

```python
m.net.viz(df, adj, col_id="id", col_size="strength", ...)   # __call__ -> vis_net
m.net.viz.mat_to_edgl(adj)                                   # individual helpers
```

## Layout

```
BayesForge/Network/explorer.py          the port (class NetExplorer / NetworkView)
BayesForge/Network/assets/netexplorer/  vendored d3 front-end
BayesForge/Network/Net.py               wires self.viz = NetExplorer() into net / net2
Test/Network/NetExplorer/
  netexplorer.py        shim -> re-exports from BayesForge.Network.explorer
  test_netexplorer.py   end-to-end test + pytest cases
  out/  logs/           generated artefacts (gitignored)
```

## What maps to what

| R (`R/Functions.R`)      | Python (`explorer.py`)           | Notes |
|--------------------------|----------------------------------|-------|
| `mat.to.edgl(M, sym, erase.diag)` | `NetExplorer.mat_to_edgl` | `M` is a JAX array; directed path reproduces R's column-major `as.vector(M)` unrolling. Returns a `from/to/weight` DataFrame. |
| `df.col.findId(df, label)` | `NetExplorer._col_id` | accepts a name or a **0-based** index |
| `colorize(df, col.att, color, new.col.name)` | `NetExplorer._colorize` | `matplotlib.LinearSegmentedColormap` for `grDevices::colorRampPalette`; **does not** re-sort the frame (see deviations) |
| `shape(vec, char)` | `NetExplorer._shape_codes` | same d3 symbol codes (`circle`→0 … `wye`→6) |
| `vis.net.format.att(...)` | `NetExplorer.format_att` | returns `(df2, ori)`; `ori` = `[id, size, color, strokeCol, stroke, shape, opacity]` source names |
| `vis.net(df, m, ...)` | `NetExplorer.vis_net` / `__call__` | `df` optional (see deviations); writes `out/NetExplorer.html`, returns a `NetworkView` |

JAX carries every numeric step: matrix→edge list, min-max scaling for node /
link opacity, layer intra/inter equality flags.

## Feature layer

Computed in Python (`BayesForge/Network/vizfeatures.py`) and spliced into the
front-end by `explorer.py`. Network metrics use **`m.net.met` only** — no
networkx / scipy.

| `vis_net(...)` arg | effect |
|---|---|
| `metrics=True` *(default)* | per-node centralities (degree, strength, in/out, eigenvector, betweenness, clustering) + a global **Stats panel** (n, edges, density, mean degree, components, diameter, global clustering). Feeds the **Size by** / **Colour by** dropdowns. |
| `directed=True` | arrowheads on links. |
| `edge_color_col=` | colour links by a `df` column (via the source node). |
| `weight_posterior=` | `(draws, N, N)` array — posterior **mean** becomes the network, link opacity = P(weight > 0), 90% interval in the tooltip. |
| `palette="cb"` | Okabe-Ito categorical + viridis sequential (colour-blind safe). Categorical `col_color` is auto-detected (distinct swatches, not a gradient). |
| `theme="dark"` | initial dark theme (toggle in the page). |
| `canvas=True` | render on a `<canvas>` instead of SVG — auto above 1500 nodes. |
| `axis_x=`, `axis_y=` | pin nodes on an invisible scatter grid by any `df` covariate (numeric → min-max scaled, categorical → evenly banded, larger = higher). Just the initial pick — the page has **X axis** / **Y axis** dropdowns over every usable covariate; `(none)` on both releases the pins. e.g. `axis_x="strength", axis_y="sex"`. |

In the page, additionally: a **Legend** (colour / shape / size), a **Min edge
weight** slider, an **Arrow size** slider (directed graphs only), **X axis** /
**Y axis** covariate dropdowns (invisible scatter layout), **Find node** (centre
+ flash), **Fit** to view, click a node to brighten its neighbours' labels, and
**PNG / JSON / CSV** export buttons.

## Layouts

`vis_net(..., layout=...)`. The four the d3 front-end already runs live in the
browser are unchanged; the rest are computed in Python
(`BayesForge/Network/layouts.py`) and the page pins nodes to the result. The
dropdown offers the four live layouts plus whichever one you asked for.

| `layout=` | what | driver (`layout_col` unless noted) |
|---|---|---|
| `force` `circle` `linear` `multilayer` | live d3 (unchanged) | — |
| `clustered` | group-in-a-box: Louvain communities (or a given grouping) laid out per-box on a grid | grouping column (optional — Louvain if omitted) |
| `spectral` | Laplacian eigenmaps (2 smallest non-zero eigenvectors) — deterministic, no seed jitter | — |
| `mds` | classical MDS on shortest-path distances | — |
| `radial` | concentric rings, radius from a nodal metric | metric column (default: degree) |
| `arc` | nodes on a line ordered by a key, links as semicircular arcs | ordering column (default: degree) |
| `layered` | Sugiyama tiers: topological generations for a DAG, else BFS layers | `directed=` toggles DAG handling |
| `geo` | fixed coordinates from two columns | `layout_x=`, `layout_y=` |
| `chord` | d3 ribbon chord of between-group weight flows | grouping column (**required**) |

`networkx` / `scipy` are used when importable (community detection, Floyd–Warshall)
but not required — NumPy fallbacks cover everything except community detection,
which then needs an explicit grouping column.

```python
m.net.viz(df, adj, col_id="id", layout="clustered", layout_col="clan")
m.net.viz(df, adj, col_id="id", layout="chord",     layout_col="age_class")
m.net.viz(df, adj, col_id="id", layout="geo", layout_x="lon", layout_y="lat")
```

**In the page.** One `vis_net` call precomputes *every* applicable layout (for
graphs up to 800 nodes) and ships them in the node payload, so the **Type**
dropdown in the panel switches between all of them client-side — `layout=` only
picks which one is shown first. `chord` appears in the dropdown whenever a
`layout_col` grouping is given; `geo` only when `layout_x` / `layout_y` are.

## The page GUI

The front-end panel was modernised: a flat white control panel whose width
scales with the viewport (`clamp(210px, 21vw, 320px)`), a real styled
`<select>`, modern range sliders, no font-awesome. It **no longer needs browser
zoom** to fit — below ~820 px viewport width the panel becomes a slide-over
behind a ☰ toggle (top-left) and the graph takes the full width. `style.css`
and a two-line `patron1.txt` patch (the toggle button) carry this; the d3 logic
is untouched.

## Opening the visualisation

`vis_net` returns a **`NetworkView`**:

* **Jupyter** — put the call (or the returned view) as the **last expression**
  of a cell; `_repr_html_` renders the network inline in an `<iframe srcdoc>`
  (the HTML is self-contained, so no notebook file-server is needed). It is
  drawn **once** — `vis_net` does not also `display()` it. If you assign the
  result (`v = m.net.viz(...)`), put `v` on its own line to show it.
* **Browser** — outside a notebook, `vis_net` opens the file in the OS
  default browser automatically (the R `file.show` behaviour). WSL is
  handled by handing the path to `cmd.exe /c start`. Inside a notebook the
  browser is *not* opened by default — call `.open()` on the view for that.
* `view.path` is the written `.html`.

```python
# minimal: just an adjacency matrix, no node table
m.net.viz(m=adj)                       # df=None -> ones placeholder, ids n1..nN

# full styling
view = m.net.viz(
    nodes_df, adj, col_id="id", col_size="strength",
    color=("green", "yellow"), col_color="age",
    col_shape="sex", shapes=("circle", "triangle"),
    layers="kinship",
)          # notebook: renders inline   |   script: pops a browser tab
view.open()  # force a browser tab from inside a notebook too
```

`vis_net` keyword flags: `inline=True` (embed d3/CSS/images -> one portable
file; set `False` to copy assets next to the HTML instead), `browser=None`
(auto: on outside a notebook, off inside — pass a bool to force),
`height=760` (inline iframe height, px).

### Deviations from the R original (deliberate)

* Column indices are **0-based** (R is 1-based).
* `df` is **optional**. `df=None` builds a one-column all-ones placeholder and
  sets `col_id=0`; node ids then fall back to positional `n1..nN` (a constant
  id column would collapse every node onto one point and hide the links). Any
  non-unique id column triggers the same fallback with a `UserWarning`.
* `_colorize` does **not** re-sort the node frame. R's `colorize` returns
  `df[order(df[,col.att]),]`; the port keeps rows aligned to the adjacency
  matrix (R relies on `colnames(m)` for the edge labels, which we don't have —
  labels come from the node frame instead). The colour ramp is a
  value→hex map, so row order does not affect it.
* `format_att` only writes a `*Value` tooltip column when its source argument
  is supplied; the R version unconditionally dereferences `col.size` /
  `col.color` / `col.stroke` / `col.shape` and errors if any is `NULL`.
* `vis_net` returns a `NetworkView` (inline-renders in Jupyter, `.open()` for
  a browser) instead of calling `file.show`; the browser still opens
  automatically when run outside a notebook.
* `opacityValue` tooltip key (R's `opacityWValue` looked like a typo); R's
  link-weight key `weigth` is **kept** because the front-end reads `d.weigth`.

## Run

```bash
cd BF/Test/Network/NetExplorer
python3 test_netexplorer.py                # build + checks, then opens the result in a browser
python3 test_netexplorer.py --no-browser   # same, headless
pytest -q test_netexplorer.py              # test_build_and_payload, test_mat_to_edgl_roundtrip
```

Forces CPU JAX (`JAX_PLATFORMS=cpu`) — the box's GPU jaxlib has a CuDNN
version mismatch.

## Test fixture

`test_netexplorer.py` builds:

* **nodes DataFrame** (`make_nodes`) — 24 individuals ×
  `id, sex, age, kinship, strength, degree`.
* **network** (`make_network`) — a `(24, 24)` weighted, asymmetric JAX
  adjacency array with a planted 3-clan block structure (within-clan ties
  denser and ~2× heavier).

then calls

```python
NetExplorer(ids=nodes["id"]).vis_net(
    nodes, M,
    col_id="id", col_size="strength",
    color=("green", "yellow"), col_color="age",
    strokeCol=("red", "blue"), col_strokeCol="kinship",
    col_stroke="degree",
    col_shape="sex", shapes=("circle", "triangle"),
    layers="kinship", link_opacity=True, out_dir="out",
)
```

and checks the emitted HTML has 24 node objects, one link object per non-zero
directed tie (155 for the seeded fixture), a `getData()` returning
`{'nodes':[…],'links':[…]}`, tooltip labels for the mapped columns, shape
codes restricted to `{0, 5}` (circle/triangle), a non-degenerate colour
gradient, and the d3 assets copied alongside.

`out/render_check.png` — headless-Chromium screenshot confirming the d3
front-end draws the payload (force layout, F=circle / M=triangle, green→yellow
age gradient, red/blue kinship strokes).
