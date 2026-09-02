"""Test suite for the probabilistic model-graph extraction (m.graph).

Runs under pytest or standalone (see test_causal_dag.py for the rationale).

Covers: latent / observed / deterministic / data node classification, edge
reconstruction from distribution parameters, the ``name=`` site-name override,
indexed dependencies (e.g. a_grp[group]), and the matplotlib renderer.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from BayesForge import bf
from BayesForge.PostModel.model_graph import (
    LATENT, OBSERVED, DETERMINISTIC, DATA,
)

m = bf("cpu", print_devices_found=False)


# Module-level model functions: inspect.getsource needs real, importable source
# (lambdas / REPL-defined functions are not inspectable).
def simple_linear(x, y):
    alpha = m.dist.normal(0, 1, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    mu = alpha + beta * x
    m.dist.normal(mu, sigma, obs=y)


def hierarchical(x, y, group):
    alpha = m.dist.normal(0, 1, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    tau = m.dist.exponential(1, name="tau")
    sigma = m.dist.exponential(1, name="sigma")
    a_grp = m.dist.normal(0, tau, name="a_grp", shape=(5,))
    mu = alpha + a_grp[group] + beta * x
    m.dist.normal(mu, sigma, obs=y)


def named_override(x, y):
    # the sample-site name differs from the python LHS variable
    b = m.dist.normal(0, 1, name="slope")
    mu = b * x
    m.dist.normal(mu, 1.0, obs=y, name="likelihood")


def test_node_classification():
    g = m.graph(simple_linear, draw=False)
    assert g.kinds["alpha"] == LATENT
    assert g.kinds["beta"] == LATENT
    assert g.kinds["sigma"] == LATENT
    assert g.kinds["mu"] == DETERMINISTIC
    assert g.kinds["y"] == OBSERVED
    assert g.kinds["x"] == DATA


def test_simple_linear_edges():
    g = m.graph(simple_linear, draw=False)
    e = set(g.edges)
    assert ("alpha", "mu") in e
    assert ("beta", "mu") in e
    assert ("x", "mu") in e
    assert ("mu", "y") in e
    assert ("sigma", "y") in e


def test_distribution_labels():
    g = m.graph(simple_linear, draw=False)
    assert g.dist_of["alpha"] == "normal"
    assert g.dist_of["sigma"] == "exponential"
    assert g.dist_of["y"] == "normal"
    assert g.dist_of["mu"] is None  # deterministic has no distribution


def test_hierarchical_scale_edge():
    g = m.graph(hierarchical, draw=False)
    e = set(g.edges)
    # non-centered-style scale parameter feeds the group effects
    assert ("tau", "a_grp") in e
    # indexed dependency a_grp[group] is detected, plus the index variable
    assert ("a_grp", "mu") in e
    assert ("group", "mu") in e
    assert g.kinds["a_grp"] == LATENT


def test_name_kwarg_overrides_node_name():
    g = m.graph(named_override, draw=False)
    # the site name 'slope' (not the LHS 'b') becomes the node identity
    assert "slope" in g.nodes
    assert "b" not in g.nodes
    assert "likelihood" in g.nodes
    assert g.kinds["likelihood"] == OBSERVED
    assert ("slope", "mu") in set(g.edges)
    assert ("mu", "likelihood") in set(g.edges)


def test_unused_inputs_are_pruned():
    def with_extra(x, y, unused):
        a = m.dist.normal(0, 1, name="a")
        mu = a + x
        m.dist.normal(mu, 1.0, obs=y)
    g = m.graph(with_extra, draw=False)
    assert "unused" not in g.nodes   # never referenced -> pruned
    assert "x" in g.nodes            # referenced -> kept


def test_draw_runs_headless():
    g = m.graph(simple_linear, draw=False)
    ax = g.draw(title="test")
    assert ax is not None
    plt.close("all")


def test_graph_method_returns_object_when_not_drawing():
    g = m.graph(hierarchical, draw=False)
    assert hasattr(g, "nodes") and hasattr(g, "edges")
    assert len(g.nodes) > 0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (model-graph suite)")
    raise SystemExit(1 if failed else 0)
