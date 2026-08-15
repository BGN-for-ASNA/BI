"""Test suite for the causal-DAG tooling (m.causal / m.dag).

Runs either under pytest (``pytest "Test/Causal & DAG"``) or standalone
(``python "Test/Causal & DAG/test_causal_dag.py"``), since pytest is not a
hard dependency of BayesForge.

Covers: parsing, cycle protection, ancestry, d-separation, the back-door
criterion / adjustment-set search, collider & mediator warnings, and the
do-operator. Each test asserts against textbook (dagitty) expectations.
"""

import matplotlib
matplotlib.use("Agg")  # headless: never pop a window during tests
import matplotlib.pyplot as plt

from BayesForge import bf

m = bf("cpu", print_devices_found=False)


# -- parsing & structure ------------------------------------------------------
def test_parse_arrow_both_directions():
    g = m.dag("a -> b, c <- b")          # mixes -> and <-
    assert set(g.edges) == {("a", "b"), ("b", "c")}
    assert g.nodes == {"a", "b", "c"}


def test_parse_separators():
    g = m.dag("a -> b; b -> c\n c -> d")  # comma / semicolon / newline
    assert set(g.edges) == {("a", "b"), ("b", "c"), ("c", "d")}


def test_cycle_is_rejected():
    raised = False
    try:
        m.dag("a -> b, b -> a")
    except ValueError:
        raised = True
    assert raised, "a 2-cycle must be rejected"


def test_self_loop_rejected():
    raised = False
    try:
        m.dag("a -> a")
    except ValueError:
        raised = True
    assert raised


# -- ancestry -----------------------------------------------------------------
def test_ancestors_descendants():
    g = m.dag("a -> b, b -> c, c -> d")
    assert g.ancestors("d") == {"a", "b", "c"}
    assert g.descendants("a") == {"b", "c", "d"}
    assert g.parents("c") == {"b"}
    assert g.children("b") == {"c"}


# -- d-separation -------------------------------------------------------------
def test_chain_dsep():
    g = m.dag("a -> b, b -> c")
    assert g.d_separated("a", "c") is False           # open chain
    assert g.d_separated("a", "c", given={"b"}) is True  # blocked by mediator


def test_fork_dsep():
    g = m.dag("b -> a, b -> c")                        # common cause
    assert g.d_separated("a", "c") is False
    assert g.d_separated("a", "c", given={"b"}) is True


def test_collider_dsep():
    g = m.dag("a -> c, b -> c")                        # collider at c
    assert g.d_separated("a", "b") is True             # closed by default
    assert g.d_separated("a", "b", given={"c"}) is False  # opened by conditioning


def test_collider_descendant_opens():
    g = m.dag("a -> c, b -> c, c -> d")
    # conditioning on a *descendant* of the collider also opens the path
    assert g.d_separated("a", "b", given={"d"}) is False


# -- back-door adjustment -----------------------------------------------------
def test_confounder_adjustment():
    g = m.dag("z -> x, z -> y, x -> y")
    assert g.adjustment_sets("x", "y") == [{"z"}]
    assert g.is_valid_adjustment_set("x", "y", {"z"}) is True
    assert g.is_valid_adjustment_set("x", "y", set()) is False


def test_no_confounding_needs_nothing():
    g = m.dag("x -> y")
    assert g.adjustment_sets("x", "y") == [set()]


def test_mediator_not_adjusted():
    g = m.dag("x -> mediator, mediator -> y, x -> y")
    # the back-door set is empty: you must NOT condition on the mediator
    assert g.adjustment_sets("x", "y") == [set()]


def test_mbias_empty_set_valid():
    # classic M-bias: m is a collider; conditioning on it induces bias
    g = m.dag("u1 -> x, u1 -> m, u2 -> m, u2 -> y, x -> y")
    sets = g.adjustment_sets("x", "y")
    assert set() in sets
    assert g.is_valid_adjustment_set("x", "y", {"m"}) is False


def test_two_confounders():
    g = m.dag("z1 -> x, z1 -> y, z2 -> x, z2 -> y, x -> y")
    sets = g.adjustment_sets("x", "y")
    # adjusting for both confounders is the (minimal) valid set
    assert {"z1", "z2"} in sets


# -- diagnostics (check) ------------------------------------------------------
def test_check_flags_collider():
    g = m.dag("x -> c, y -> c")
    warns = g.check("x", "y", given={"c"})
    assert any("collider" in w for w in warns)


def test_check_flags_mediator():
    g = m.dag("x -> mm, mm -> y, x -> y")
    warns = g.check("x", "y", given={"mm"})
    assert any("mediator" in w or "directed path" in w for w in warns)


def test_check_clean_set_is_silent():
    g = m.dag("z -> x, z -> y, x -> y")
    assert g.check("x", "y", given={"z"}) == []


# -- do-operator --------------------------------------------------------------
def test_do_cuts_incoming_edges():
    g = m.dag("z -> x, z -> y, x -> y")
    gx = g.do("x")
    assert ("z", "x") not in gx.edges          # incoming arrow removed
    assert ("x", "y") in gx.edges              # outgoing arrow kept
    assert ("z", "y") in gx.edges              # untouched elsewhere


# -- plotting (smoke test) ----------------------------------------------------
def test_plot_runs_headless():
    g = m.dag("z -> x, z -> y, x -> y")
    ax = g.plot(x="x", y="y", adjust={"z"}, title="test")
    assert ax is not None
    plt.close("all")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
    print(f"\n{passed} passed, {failed} failed (causal DAG suite)")
    raise SystemExit(1 if failed else 0)
