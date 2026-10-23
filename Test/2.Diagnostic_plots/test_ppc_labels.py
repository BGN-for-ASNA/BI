"""Titles and axis labels of every PPC plot in BayesForge/Diagnostic/ppc.py.

Every public PPC function takes ``title``, ``name``, ``x_label`` and
``y_label``:

  * ``name`` names the outcome in the default title, the axis titles and the
    trace names (``m.diag`` passes the observed site name, e.g. "height"), so
    nothing on the plot should still read "y"/"yrep" (the bayesplot jargon the
    labels replaced -- see ``_obs``/``_preds``/``_title`` in ppc.py);
  * ``title`` replaces the figure title verbatim;
  * ``x_label``/``y_label`` override the axis titles -- a string sets every
    x (y) axis, a list sets them one per subplot in layout order and a None
    entry keeps that axis's own default (``_axis_labels`` in ppc.py).

Each test below is parametrized over ``(function name, call lambda)`` so every
function gets its own test id, and checks, per function:

  1. default call with name="height": the title contains "height" and no axis
     title / trace name / subplot annotation still uses "y" or "yrep";
  2. title="My title" -> fig.layout.title.text == "My title";
  3. x_label/y_label as strings -> every xaxis*/yaxis* title is that string;
  4. multi-subplot figures: x_label/y_label as a list with one entry per axis,
     the second entry None -> each axis got its entry, the None one kept its
     default text;
  5. labels do not change the data: same traces, same x/y arrays as the
     default call.

Fixtures are the same arrays as test_ppc_vs_bayesplot.py reads from
fixtures/input.json (continuous y/yrep/log_lik, count y/yrep, x).
"""
import json
import os
import re
import warnings

import numpy as np
import pytest

from BayesForge.Diagnostic import ppc

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "fixtures", "input.json")) as _f:
    INP = json.load(_f)

Y = np.array(INP["continuous"]["y"], dtype=float)          # (30,)
YREP = np.array(INP["continuous"]["yrep"], dtype=float)    # (200, 30)
LL = np.array(INP["continuous"]["log_lik"], dtype=float)   # (200, 30)
X = np.array(INP["x"], dtype=float)                        # (30,) covariate
CY = np.array(INP["count"]["y"])                           # (30,) ints
CYREP = np.array(INP["count"]["yrep"])                     # (200, 30) ints
# X is loaded for parity with test_ppc_vs_bayesplot.py; the label checks below
# call each function with its own defaults, so x is never passed to it.

NAME = "height"
TITLE = "My title"
X_CUSTOM = "X custom"
Y_CUSTOM = "Y custom"


def _call_ppc_loo_intervals(**kwargs):
    """ppc_loo_intervals is a deprecated alias of ppc_intervals_sorted and
    warns on every call; the warning is not what these tests are about, so
    keep it out of the report (test_ppc_vs_bayesplot.py covers the alias)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ppc.ppc_loo_intervals(Y, YREP, **kwargs)


# (function name, call lambda) -- one entry per public PPC function; the
# lambdas take only the labelling kwargs (title/name/x_label/y_label).
PPC_PLOTS = [
    ("ppc_density", lambda **kw: ppc.ppc_density(Y, YREP, **kw)),
    ("ppc_hist", lambda **kw: ppc.ppc_hist(Y, YREP, **kw)),
    ("ppc_boxplot", lambda **kw: ppc.ppc_boxplot(Y, YREP, **kw)),
    ("ppc_stat", lambda **kw: ppc.ppc_stat(Y, YREP, **kw)),
    ("ppc_stat_2d", lambda **kw: ppc.ppc_stat_2d(Y, YREP, **kw)),
    ("ppc_intervals", lambda **kw: ppc.ppc_intervals(Y, YREP, **kw)),
    ("ppc_ribbon", lambda **kw: ppc.ppc_ribbon(Y, YREP, **kw)),
    ("ppc_error_scatter", lambda **kw: ppc.ppc_error_scatter(Y, YREP, **kw)),
    ("ppc_error_hist", lambda **kw: ppc.ppc_error_hist(Y, YREP, **kw)),
    ("ppc_scatter", lambda **kw: ppc.ppc_scatter(Y, YREP, **kw)),
    ("ppc_rootogram", lambda **kw: ppc.ppc_rootogram(CY, CYREP, **kw)),
    ("ppc_bars", lambda **kw: ppc.ppc_bars(CY, CYREP, **kw)),
    ("ppc_loo_pit", lambda **kw: ppc.ppc_loo_pit(Y, YREP, log_likelihood=LL, **kw)),
    ("ppc_intervals_sorted", lambda **kw: ppc.ppc_intervals_sorted(Y, YREP, **kw)),
    ("ppc_loo_intervals", _call_ppc_loo_intervals),
]
PPC_IDS = [name for name, _ in PPC_PLOTS]

# ppc_error_scatter draws the identity diagonal as a trace named "y = x": that
# is the equation of the line (bayesplot labels it the same way), not the
# outcome placeholder, so the bare-"y" check exempts exactly this label.
_IDENTITY_LINE_LABELS = {"y = x"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _axis_titles(fig, kind):
    """{axis key: title text} for every xaxis*/yaxis* in fig.layout, in the
    numeric layout order _axis_labels() itself uses."""
    layout = fig.layout.to_plotly_json()
    keys = sorted((k for k in layout if re.fullmatch(kind + r"\d*", k)),
                  key=lambda k: int(k[len(kind):] or 1))
    return {k: ((layout[k] or {}).get("title") or {}).get("text") for k in keys}


def _labelled_texts(fig):
    """Every string the figure shows as a title or label: the figure title, all
    axis titles, all trace names and all subplot annotations."""
    layout = fig.layout.to_plotly_json()
    texts = [(layout.get("title") or {}).get("text")]
    texts += list(_axis_titles(fig, "xaxis").values())
    texts += list(_axis_titles(fig, "yaxis").values())
    texts += [tr.name for tr in fig.data]
    texts += [a.get("text") for a in layout.get("annotations", [])]
    return [t for t in texts if t]


def _as_comparable(value):
    """Trace coordinates as an array: segment traces (ppc_intervals) hold
    None-separated tuples, everything else a numeric array."""
    if value is None:
        return None
    seq = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    if any(v is None for v in seq):
        return np.asarray(seq, dtype=object)
    return np.asarray(seq, dtype=float)


# ---------------------------------------------------------------------------
# 1. default labels name the outcome, no leftover "y"/"yrep"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_default_labels_name_the_outcome(case):
    fname, call = case
    fig = call(name=NAME)

    title = fig.layout.title.text or ""
    assert NAME in title, f"{fname}: default title {title!r} does not name the outcome"

    texts = _labelled_texts(fig)
    assert any(NAME in t for t in texts), \
        f"{fname}: no axis title, trace name or subplot title mentions {NAME!r}"

    leftovers = []
    for text in texts:
        if re.search(r"\byrep\b", text, re.IGNORECASE):
            leftovers.append((text, "says 'yrep'"))
        elif re.search(r"\by\b", text) and text.strip() not in _IDENTITY_LINE_LABELS:
            leftovers.append((text, "uses a bare 'y' for the outcome"))
    assert not leftovers, f"{fname}: labels still use the placeholder names: {leftovers}"


# ---------------------------------------------------------------------------
# 2. title= replaces the figure title verbatim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_title_overrides_default(case):
    fname, call = case
    fig = call(name=NAME, title=TITLE)
    assert fig.layout.title.text == TITLE, f"{fname}: title= was not used verbatim"


# ---------------------------------------------------------------------------
# 3. string x_label/y_label set every axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_string_labels_set_every_axis(case):
    fname, call = case
    fig = call(name=NAME, x_label=X_CUSTOM, y_label=Y_CUSTOM)

    x_axes, y_axes = _axis_titles(fig, "xaxis"), _axis_titles(fig, "yaxis")
    assert x_axes and y_axes, f"{fname}: no xaxis*/yaxis* key in fig.layout"
    wrong_x = {k: v for k, v in x_axes.items() if v != X_CUSTOM}
    wrong_y = {k: v for k, v in y_axes.items() if v != Y_CUSTOM}
    assert not wrong_x, f"{fname}: x axes without the custom title: {wrong_x}"
    assert not wrong_y, f"{fname}: y axes without the custom title: {wrong_y}"


# ---------------------------------------------------------------------------
# 4. list x_label/y_label: one per subplot, None keeps that axis's default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_list_labels_one_per_subplot(case):
    fname, call = case
    default = call(name=NAME)
    default_x, default_y = _axis_titles(default, "xaxis"), _axis_titles(default, "yaxis")
    if len(default_x) < 2:
        pytest.skip(f"{fname} draws a single subplot: nothing to set one-per-axis")

    x_keys, y_keys = list(default_x), list(default_y)
    x_labels = [f"x axis {k}" for k in x_keys]
    y_labels = [f"y axis {k}" for k in y_keys]
    x_labels[1] = y_labels[1] = None       # the second axis keeps its own title

    fig = call(name=NAME, x_label=x_labels, y_label=y_labels)
    got_x, got_y = _axis_titles(fig, "xaxis"), _axis_titles(fig, "yaxis")
    assert list(got_x) == x_keys, f"{fname}: the set of x axes changed"
    assert list(got_y) == y_keys, f"{fname}: the set of y axes changed"

    for kind, got, want, defaults in (("x", got_x, x_labels, default_x),
                                      ("y", got_y, y_labels, default_y)):
        for key, text, default_text in zip(got, want, defaults.values()):
            if text is None:
                assert got[key] == default_text, (
                    f"{fname}: {kind} axis {key!r} was passed None but lost its "
                    f"default title {default_text!r} (got {got[key]!r})")
            else:
                assert got[key] == text, (
                    f"{fname}: {kind} axis {key!r} should read {text!r}, "
                    f"got {got[key]!r}")


# ---------------------------------------------------------------------------
# 5. labelling does not touch the data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PPC_PLOTS, ids=PPC_IDS)
def test_labels_do_not_change_the_data(case):
    fname, call = case
    plain = call(name=NAME)
    labelled = call(name=NAME, title=TITLE, x_label=X_CUSTOM, y_label=Y_CUSTOM)

    assert len(labelled.data) == len(plain.data), (
        f"{fname}: {len(plain.data)} traces by default, "
        f"{len(labelled.data)} with labels")
    for i, (before, after) in enumerate(zip(plain.data, labelled.data)):
        assert before.type == after.type, (
            f"{fname}: trace {i} is {before.type} by default, {after.type} with labels")
        for attr in ("x", "y"):
            a, b = _as_comparable(getattr(before, attr)), _as_comparable(getattr(after, attr))
            assert (a is None) == (b is None), (
                f"{fname}: trace {i} attribute {attr!r} present in only one figure")
            if a is None:
                continue
            np.testing.assert_array_equal(
                a, b, err_msg=f"{fname}: trace {i} attribute {attr!r} changed "
                              "when the labels were set")
