"""Visual demo of the causal-DAG and model-graph features.

Unlike the test suites (which only assert correctness, headless), this script
*renders* every plot. By default it saves PNGs next to this file; pass
``--show`` to open interactive windows instead.

    python "Test/Causal & DAG/demo.py"            # save PNGs
    python "Test/Causal & DAG/demo.py" --show     # interactive windows
"""

import os
import sys

SHOW = "--show" in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from BayesForge import bf

HERE = os.path.dirname(os.path.abspath(__file__))
m = bf("cpu", print_devices_found=False)


def _finish(name):
    if SHOW:
        plt.show()
    else:
        path = os.path.join(HERE, name)
        plt.savefig(path, dpi=90, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close("all")


# 1. Causal DAGs -------------------------------------------------------------
print("Causal DAGs:")

g = m.dag("z -> x, z -> y, x -> y")
print("  confounder -> adjustment_sets(x, y):", g.adjustment_sets("x", "y"))
g.plot(x="x", y="y", adjust={"z"}, title="Confounder: adjust for z")
_finish("demo_confounder.png")

g2 = m.dag("x -> mediator, mediator -> y, x -> y")
print("  mediator   -> adjustment_sets(x, y):", g2.adjustment_sets("x", "y"))
print("  mediator   -> check(given=mediator):", g2.check("x", "y", given={"mediator"}))
g2.plot(x="x", y="y", title="Mediator: do NOT adjust for it")
_finish("demo_mediator.png")

g3 = m.dag("u1 -> x, u1 -> m, u2 -> m, u2 -> y, x -> y")
print("  M-bias     -> adjustment_sets(x, y):", g3.adjustment_sets("x", "y"))
g3.plot(x="x", y="y", title="M-bias: conditioning on m opens a path")
_finish("demo_mbias.png")


# 2. Probabilistic model graph ----------------------------------------------
print("\nModel graph:")


def hierarchical_model(x, y, group):
    alpha = m.dist.normal(0, 1, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    tau = m.dist.exponential(1, name="tau")
    sigma = m.dist.exponential(1, name="sigma")
    a_grp = m.dist.normal(0, tau, name="a_grp", shape=(5,))
    mu = alpha + a_grp[group] + beta * x
    m.dist.normal(mu, sigma, obs=y)


graph = m.graph(hierarchical_model, draw=False)
print("  nodes:", graph.nodes)
print("  edges:", graph.edges)
graph.draw(title="Hierarchical linear model")
_finish("demo_model_graph.png")

print("\nDone." + ("" if SHOW else " Open the PNG files listed above."))
