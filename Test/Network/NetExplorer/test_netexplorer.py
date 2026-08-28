"""End-to-end test for the NetExplorer data-handling port.

Builds a node-characteristics ``pandas.DataFrame`` and a ``(N, N)`` JAX
adjacency array, runs :class:`NetExplorer.vis_net`, and checks the emitted
``NetExplorer.html`` carries a well-formed ``nodes`` / ``links`` payload plus
the copied d3 assets.

Run directly:      python3 test_netexplorer.py
Run under pytest:  pytest -q test_netexplorer.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # box GPU jax has a CuDNN mismatch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from netexplorer import NetExplorer  # noqa: E402

N = 24
SEED = 20260828
OUT = HERE / "out"
LOGS = HERE / "logs"


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
def make_nodes(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Node characteristics table (the R ``df``)."""
    rng = np.random.default_rng(seed)
    kin = rng.choice(["clanA", "clanB", "clanC"], size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.6, 0.4])
    age = rng.integers(1, 20, size=n)
    return pd.DataFrame(
        {
            "id": [f"ind{i + 1:02d}" for i in range(n)],
            "sex": sex,
            "age": age.astype(float),
            "kinship": kin,
            "strength": np.round(rng.gamma(2.0, 1.5, size=n), 3),
            "degree": rng.integers(1, n // 2, size=n).astype(float),
        }
    )


def make_network(nodes: pd.DataFrame, seed: int = SEED):
    """Weighted, asymmetric ``(N, N)`` JAX adjacency array with block structure
    (within-clan ties denser and heavier)."""
    n = len(nodes)
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    clan = jnp.asarray(pd.factorize(nodes["kinship"], sort=True)[0])
    same_clan = clan[:, None] == clan[None, :]
    p = jnp.where(same_clan, 0.55, 0.12)
    present = jax.random.bernoulli(k1, p, (n, n))
    w = jax.random.gamma(k2, 2.0, (n, n)) * jnp.where(same_clan, 2.0, 1.0)
    noise = 0.3 * jax.random.uniform(k3, (n, n))  # break symmetry
    M = present * (w + noise)
    M = M.at[jnp.diag_indices(n)].set(0.0)  # no self-loops
    return jnp.round(M, 3)


# --------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------- #
def build(out_dir: Path = OUT, log: logging.Logger | None = None, browser: bool = False) -> dict:
    log = log or logging.getLogger("netexplorer.test")
    nodes = make_nodes()
    M = make_network(nodes)

    log.info("nodes: %d rows x %d cols  |  columns=%s", *nodes.shape, list(nodes.columns))
    log.info("matrix: %s on %s", M.shape, list(M.devices()))
    nnz = int((np.asarray(M) != 0).sum())
    log.info("non-zero directed links (incl. would-be diagonal): %d", nnz)

    ne = NetExplorer(ids=nodes["id"].tolist())

    # exercise the edge-list converter on its own (JAX -> DataFrame)
    edgl = ne.mat_to_edgl(M)
    edgl_nz = edgl[edgl["weight"] != 0]
    log.info("mat_to_edgl -> %d rows, %d non-zero", len(edgl), len(edgl_nz))

    view = ne.vis_net(
        nodes,
        M,
        col_id="id",
        col_size="strength",
        color=("green", "yellow"),
        col_color="age",
        strokeCol=("red", "blue"),
        col_strokeCol="kinship",
        col_stroke="degree",
        col_shape="sex",
        shapes=("circle", "triangle"),
        layers="kinship",
        link_opacity=True,
        out_dir=out_dir,
        browser=browser,  # pytest keeps it False; __main__ passes True (R file.show behaviour)
    )
    html_path = view.path
    html = html_path.read_text(encoding="utf-8")
    log.info("wrote %s (%d bytes)  repr=%r", html_path, len(html), view)

    # non-inline variant: assets copied next to the file instead of embedded
    ext_dir = out_dir / "external"
    ne.vis_net(nodes, M, col_id="id", col_size="strength", col_color="age",
               color=("green", "yellow"), out_dir=ext_dir, inline=False, browser=False)

    node_objs = re.findall(r"\{'id':'ind\d+'.*?\}", html)
    link_objs = re.findall(r"\{'source':'ind\d+'.*?\}", html)
    log.info("payload: %d node objects, %d link objects", len(node_objs), len(link_objs))

    metrics = {
        "n_nodes": len(nodes),
        "n_node_objs": len(node_objs),
        "n_link_objs": len(link_objs),
        "n_edgl_nonzero": int(len(edgl_nz)),
        "html_bytes": len(html),
        "html_path": str(html_path),
        "self_contained": ("<script src='d3.min.js'>" not in html)
        and ("d3.select" in html)
        and ('src="data:image' in html),
        "ext_assets_copied": sorted(
            p.name for p in (out_dir / "external").iterdir() if p.suffix in {".js", ".css", ".jpg", ".png"}
        ),
        "has_getData": "function getData()" in html,
        "has_nodes_key": "'nodes':[" in html,
        "has_links_key": "'links':[" in html,
        "tooltip_has_age": "age (color)" in html,
        "tooltip_has_sex": "sex (shape)" in html,
        "shape_codes": sorted({int(x) for x in re.findall(r"'shape':(\d+)", html)}),
        "distinct_node_colors": sorted(set(re.findall(r"'color':'(#[0-9a-fA-F]{6})'", html))),
        "iframe_repr": view._repr_html_()[:60],
    }
    return metrics


def _check(metrics: dict) -> None:
    m = metrics
    assert m["n_node_objs"] == m["n_nodes"], m
    assert m["n_link_objs"] == m["n_edgl_nonzero"] > 0, m
    assert m["has_getData"] and m["has_nodes_key"] and m["has_links_key"], m
    assert m["tooltip_has_age"] and m["tooltip_has_sex"], m
    assert set(m["shape_codes"]).issubset({0, 5}), m  # circle / triangle only
    assert len(m["distinct_node_colors"]) > 1, m  # gradient actually varied
    assert m["self_contained"], m  # inline=True folds d3/css/images in
    assert {"d3.min.js", "d3-tip.js", "style.css"}.issubset(set(m["ext_assets_copied"])), m
    assert m["iframe_repr"].startswith("<iframe srcdoc="), m


# --------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------- #
def test_build_and_payload():
    _check(build())


def test_mat_to_edgl_roundtrip():
    """Directed edge list matches R's column-major unrolling."""
    ne = NetExplorer(ids=["a", "b", "c"])
    M = jnp.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    e = ne.mat_to_edgl(M).set_index(["from", "to"])["weight"].to_dict()
    assert e[("a", "b")] == 1.0 and e[("b", "a")] == 3.0
    assert e[("a", "c")] == 2.0 and e[("c", "a")] == 5.0
    assert e[("b", "c")] == 4.0 and e[("c", "b")] == 6.0
    assert ("a", "a") not in e


def test_df_none_defaults_and_unique_ids():
    """df=None -> ones placeholder + col_id=0; non-unique id column falls back
    to positional n1..nN so links stay visible."""
    n = 12
    rng = np.random.default_rng(0)
    M = jnp.asarray((rng.random((n, n)) < 0.25) * rng.integers(1, 4, (n, n))).astype(float)
    M = M.at[jnp.diag_indices(n)].set(0.0)
    out = OUT / "df_none"
    with pytest.warns(UserWarning, match="not unique"):
        view = NetExplorer().vis_net(m=M, out_dir=out, browser=False)
    html = view.path.read_text()
    ids = re.findall(r"'id':'([^']*)'", html)
    src = re.findall(r"'source':'([^']*)'", html)
    assert len(ids) == n and len(set(ids)) == n, ids
    assert ids[:3] == ["n1", "n2", "n3"], ids
    assert 0 < len(src) == int((np.asarray(M) != 0).sum()), (len(src),)


def test_layouts_emit_coords_or_chord():
    """Every non-live layout either pins nodes (lx/ly) or ships a chord payload,
    and the dropdown offers it."""
    n = 24
    rng = np.random.default_rng(3)
    grp = rng.choice(list("XYZ"), n)
    A = (rng.random((n, n)) < np.where(grp[:, None] == grp[None, :], 0.35, 0.06)).astype(float)
    A = A * rng.integers(1, 4, (n, n))
    np.fill_diagonal(A, 0)
    A = jnp.asarray(A.astype(float))
    df = pd.DataFrame(
        {"id": [f"i{k}" for k in range(n)], "grp": grp,
         "deg": np.asarray(A).sum(1), "x": rng.random(n), "y": rng.random(n)}
    )
    ne = NetExplorer(ids=df["id"].tolist())

    # one render: every applicable layout is precomputed and offered in the GUI
    html = ne.vis_net(
        df, A, col_id="id", layout="spectral",
        layout_col="grp", layout_x="x", layout_y="y",
        out_dir=OUT / "lay_all", browser=False,
    ).path.read_text()
    for want in ("Force", "Circle", "Linear", "Multilayer", "Clustered", "Spectral",
                 "MDS", "Radial", "Arc", "Layered", "Geo", "Chord"):
        assert f'"{want}"' in html, (want, "missing from dropdown")
    assert "var Layout = 'Spectral'" in html
    assert "json['chord']" in html and "drawChordOnce" in html
    m = re.search(r"'pos':\{([^}]*)\}", html).group(1)
    for k in ("clustered", "spectral", "mds", "radial", "arc", "layered", "geo"):
        assert f"'{k}':[" in m, (k, "missing from node pos map")
    assert 'id="menuToggle"' in html and "--panel-w" in html  # modernised GUI

    # each layout can be the initial one too
    for lay, kw in {
        "clustered": {"layout_col": "grp"}, "radial": {}, "arc": {},
        "layered": {}, "chord": {"layout_col": "grp"},
        "geo": {"layout_x": "x", "layout_y": "y"},
    }.items():
        h = ne.vis_net(df, A, col_id="id", layout=lay,
                       out_dir=OUT / f"lay_{lay}", browser=False, **kw).path.read_text()
        label = "MDS" if lay == "mds" else lay.capitalize()
        assert f"var Layout = '{label}'" in h, lay


# --------------------------------------------------------------------- #
# script entry point
# --------------------------------------------------------------------- #
def main() -> int:
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOGS / f"netexplorer_test_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("netexplorer.test")
    log.info("jax %s | numpy %s | pandas %s", jax.__version__, np.__version__, pd.__version__)

    no_browser = "--no-browser" in sys.argv
    metrics = build(log=log, browser=not no_browser)
    (OUT / f"results_{stamp}.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics:\n%s", json.dumps(metrics, indent=2))

    try:
        _check(metrics)
        test_mat_to_edgl_roundtrip()
    except AssertionError as e:
        log.error("CHECK FAILED: %s", e)
        return 1
    log.info("ALL CHECKS PASSED")
    log.info("%s browser for: %s", "skipped opening" if no_browser else "opened", metrics["html_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
