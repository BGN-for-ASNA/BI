# NetExplorer — data-handling port (R → BayesForge)

Ports the **data layer** of the R package
[`NetExplorer`](https://github.com/SebastianSosa/Network-Explorer)
(`R/Functions.R`, pinned commit `e4000d3`) to a BayesForge-style Python class
that runs on **JAX arrays + pandas DataFrames** and emits the same standalone
`NetExplorer.html` for interactive d3.js network visualisation.

The implementation ships in the package as
**`BayesForge/Network/explorer.py`** (`class NetExplorer`, `class
NetworkView`) with the d3 front-end vendored under
`BayesForge/Network/assets/netexplorer/` (`patron1.txt`, `patron2.txt`,
`d3.min.js`, `d3-tip.js`, `style.css`, `logo.jpg`, `save.png`, copied verbatim
from upstream `inst/www/`). Only the code that *builds the `nodes`/`links`
payload and splices it into the template* was rewritten.

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
| `vis.net(df, m, ...)` | `NetExplorer.vis_net` / `__call__` | writes `out/NetExplorer.html`, returns a `NetworkView` |

JAX carries every numeric step: matrix→edge list, min-max scaling for node /
link opacity, layer intra/inter equality flags.

## Opening the visualisation

`vis_net` returns a **`NetworkView`**:

* **Jupyter** — put the call (or the returned view) as the last line of a
  cell; `_repr_html_` renders the network inline in an `<iframe srcdoc>`
  (the HTML is self-contained, so no notebook file-server is needed). The
  call also `display()`s itself, so it shows even when not the last
  expression.
* **Browser** — outside a notebook, `vis_net` opens the file in the OS
  default browser automatically (the R `file.show` behaviour). WSL is
  handled by handing the path to `cmd.exe /c start`. Inside a notebook the
  browser is *not* opened by default — call `.open()` on the view for that.
* `view.path` is the written `.html`.

```python
from netexplorer import NetExplorer

view = NetExplorer(ids=nodes["id"]).vis_net(
    nodes, M, col_id="id", col_size="strength",
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
