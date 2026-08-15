"""Probabilistic-model graph extraction & plotting for BayesForge.

Parses the *source* of a BF/NumPyro model function (same AST approach as
``to_latex``) and reconstructs the directed graph of the underlying
probabilistic model:

    * stochastic (latent) sample sites        -- ellipse, white
    * observed sample sites (``obs=`` given)  -- ellipse, shaded
    * deterministic assignments               -- rounded box
    * data / model inputs (function arguments) -- plain box

Edges connect a node to every other node referenced in its distribution
parameters (or, for deterministic nodes, its right-hand side).

Usage from a BF instance::

    m.graph()              # graph of the last-fitted / current model
    m.graph(my_model)      # graph of an arbitrary model function
"""

import ast
import inspect
import textwrap


_WARNED = False


def _warn_experimental():
    global _WARNED
    if not _WARNED:
        print("[WARNING] Probabilistic model-graph plotting is an experimental "
              "feature. Use it with caution. [WARNING]")
        _WARNED = True


# Node kinds
LATENT = "latent"
OBSERVED = "observed"
DETERMINISTIC = "deterministic"
DATA = "data"


def _names_in(node):
    """Collect all bare ``Name`` identifiers referenced inside an AST node."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.append(sub.id)
    return out


def _is_dist_call(node):
    return (isinstance(node, ast.Call)
            and "dist" in ast.unparse(node.func))


def _dist_name(call):
    """The distribution name, e.g. ``m.dist.normal(...)`` -> 'normal'."""
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id


def _call_meta(call):
    """Return (name_kwarg, obs_value_name, dependency_names) for a dist call."""
    name_kw, obs_name = None, None
    dep_nodes = list(call.args)
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            name_kw = kw.value.value
        elif kw.arg == "obs":
            obs_name = (kw.value.id if isinstance(kw.value, ast.Name)
                        else ast.unparse(kw.value))
        elif kw.arg in ("shape", "sample"):
            continue  # structural, not a probabilistic dependency
        else:
            dep_nodes.append(kw.value)
    deps = []
    for n in dep_nodes:
        deps.extend(_names_in(n))
    return name_kw, obs_name, deps


class ModelGraph:
    """A parsed probabilistic-model graph with a matplotlib renderer."""

    def __init__(self, nodes, edges, kinds, dist_of):
        self.nodes = nodes          # ordered list of node names
        self.edges = edges          # list of (src, dst)
        self.kinds = kinds          # name -> kind
        self.dist_of = dist_of      # name -> distribution label (or None)

    # -- layout ---------------------------------------------------------------
    def _layers(self):
        parents = {n: set() for n in self.nodes}
        children = {n: set() for n in self.nodes}
        for a, b in self.edges:
            parents[b].add(a)
            children[a].add(b)
        # longest-path layering over a topological order
        indeg = {n: len(parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if indeg[n] == 0]
        order, layer = [], {n: 0 for n in self.nodes}
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in children[n]:
                layer[c] = max(layer[c], layer[n] + 1)
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        # any nodes left out by cycles (shouldn't happen for a valid model)
        for n in self.nodes:
            layer.setdefault(n, 0)
        return layer

    def draw(self, figsize=(8, 6), title=None):
        """Render the model graph with matplotlib."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        layer = self._layers()
        by_layer = {}
        for n in self.nodes:
            by_layer.setdefault(layer[n], []).append(n)

        pos = {}
        for l, nodes in by_layer.items():
            k = len(nodes)
            for i, node in enumerate(nodes):
                pos[node] = (l * 2.2, (k - 1) / 2 - i)

        style = {
            LATENT:        dict(fc="white",   ec="#333333", box="ellipse"),
            OBSERVED:      dict(fc="#9ecae1", ec="#3182bd", box="ellipse"),
            DETERMINISTIC: dict(fc="#fff2cc", ec="#d6b656", box="round"),
            DATA:          dict(fc="#eeeeee", ec="#999999", box="square"),
        }

        fig, ax = plt.subplots(figsize=figsize)
        for a, b in self.edges:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                shrinkA=20, shrinkB=20, color="#666666", lw=1.3, zorder=1))

        for node, (px, py) in pos.items():
            st = style[self.kinds[node]]
            boxstyle = {
                "ellipse": "circle,pad=0.45",
                "round": "round,pad=0.45",
                "square": "square,pad=0.4",
            }[st["box"]]
            label = node
            d = self.dist_of.get(node)
            if d:
                label = f"{node}\n~{d}"
            ax.text(px, py, label, ha="center", va="center", fontsize=9, zorder=2,
                    bbox=dict(boxstyle=boxstyle, fc=st["fc"], ec=st["ec"], lw=1.5))

        xs = [p[0] for p in pos.values()] or [0]
        ys = [p[1] for p in pos.values()] or [0]
        ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
        ax.set_ylim(min(ys) - 1.6, max(ys) + 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title or "Probabilistic model graph")

        from matplotlib.patches import Patch
        legend = [
            Patch(fc="white", ec="#333333", label="latent"),
            Patch(fc="#9ecae1", ec="#3182bd", label="observed"),
            Patch(fc="#fff2cc", ec="#d6b656", label="deterministic"),
            Patch(fc="#eeeeee", ec="#999999", label="data/input"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        plt.show()
        return ax

    def __repr__(self):
        e = ", ".join(f"{a}->{b}" for a, b in self.edges)
        return f"ModelGraph(nodes={len(self.nodes)}, edges=[{e}])"


def model_to_graph(model):
    """Parse a model function into a :class:`ModelGraph`.

    Args:
        model: A BF/NumPyro model function (its source must be inspectable).
    """
    _warn_experimental()
    src = textwrap.dedent(inspect.getsource(model))
    tree = ast.parse(src)
    func = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)), None)
    if func is None:
        raise ValueError("Could not find a function definition in the model source")

    inputs = [a.arg for a in func.args.args if a.arg not in ("self",)]

    nodes = []
    kinds = {}
    dist_of = {}
    deps = {}          # node -> list of referenced names
    var_to_node = {}   # python LHS variable -> node name (often identical)

    def add_node(name, kind, dist=None, depends=()):
        if name not in kinds:
            nodes.append(name)
        kinds[name] = kind
        dist_of[name] = dist
        deps[name] = list(depends)

    # function inputs become data nodes (kept only if later referenced)
    for inp in inputs:
        add_node(inp, DATA)

    for stmt in func.body:
        # `lhs = <something>`
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            lhs = stmt.targets[0].id
            if _is_dist_call(stmt.value):
                name_kw, obs_name, d = _call_meta(stmt.value)
                node = name_kw or lhs
                kind = OBSERVED if obs_name else LATENT
                add_node(node, kind, _dist_name(stmt.value), d)
                var_to_node[lhs] = node
                if obs_name:
                    var_to_node.setdefault(obs_name, node)
            else:
                add_node(lhs, DETERMINISTIC, None, _names_in(stmt.value))
                var_to_node[lhs] = lhs
        # bare `m.dist.xxx(..., obs=y)`
        elif isinstance(stmt, ast.Expr) and _is_dist_call(stmt.value):
            name_kw, obs_name, d = _call_meta(stmt.value)
            node = name_kw or obs_name or f"obs_{_dist_name(stmt.value)}"
            add_node(node, OBSERVED if obs_name else LATENT,
                     _dist_name(stmt.value), d)
            if obs_name:
                var_to_node.setdefault(obs_name, node)

    # resolve dependency names to actual node names and build edges
    edges = []
    known = set(nodes)
    for node in nodes:
        for ref in deps.get(node, []):
            target = var_to_node.get(ref, ref)
            if target in known and target != node:
                edges.append((target, node))
    edges = list(dict.fromkeys(edges))  # dedupe, preserve order

    # drop data/input nodes that are never referenced (keeps graph clean)
    referenced = {a for a, _ in edges} | {b for _, b in edges}
    pruned = [n for n in nodes
              if kinds[n] != DATA or n in referenced]
    edges = [(a, b) for a, b in edges if a in pruned and b in pruned]

    return ModelGraph(pruned, edges, kinds, dist_of)
