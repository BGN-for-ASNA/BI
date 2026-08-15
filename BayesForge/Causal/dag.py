"""Causal DAG tooling for BayesForge.

Pure-python (no networkx/graphviz dependency) implementation of the core
graphical-causal-inference primitives that *Statistical Rethinking* users
expect: d-separation, the back-door criterion, minimal adjustment-set search,
collider/mediator warnings and interventional (do-operator) graph mutilation.

Example
-------
>>> g = DAG("z -> x, z -> y, x -> y")          # confounder z
>>> g.adjustment_sets("x", "y")
[{'z'}]
>>> g.d_separated("x", "y", given={"z"})
False
>>> g.check("x", "y", given={"m"})             # m is a mediator x->m->y
['conditioning on {m} blocks a directed path x ~> y (mediator/over-control)']
"""

from itertools import combinations

_WARNED = False


def _warn_experimental():
    global _WARNED
    if not _WARNED:
        print("[WARNING] Causal DAG tooling is an experimental feature. "
              "Use it with caution. [WARNING]")
        _WARNED = True


def _parse_spec(spec):
    """Parse a DAG specification string into a list of (parent, child) edges.

    Accepts ``->`` / ``<-`` edges separated by commas, semicolons or newlines,
    e.g. ``"z -> x, z -> y; x -> y"`` or ``"y <- x"``.
    """
    edges = []
    for chunk in spec.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "<-" in chunk:
            rhs, lhs = chunk.split("<-")
            a, b = rhs.strip(), lhs.strip()
            edges.append((b, a))  # lhs is the parent
        elif "->" in chunk:
            a, b = chunk.split("->")
            edges.append((a.strip(), b.strip()))
        else:
            raise ValueError(f"Cannot parse edge: {chunk!r} (use 'a -> b' or 'b <- a')")
    return edges


class DAG:
    """A directed acyclic graph for causal reasoning.

    Args:
        spec: Either a specification string (``"z -> x, x -> y"``) or an
            iterable of ``(parent, child)`` tuples.
    """

    def __init__(self, spec=None):
        _warn_experimental()
        self._parents = {}   # node -> set of parents
        self._children = {}  # node -> set of children
        if spec is not None:
            edges = _parse_spec(spec) if isinstance(spec, str) else list(spec)
            for a, b in edges:
                self.add_edge(a, b)

    # -- construction ---------------------------------------------------------
    def _ensure(self, node):
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def add_edge(self, parent, child):
        """Add a directed edge ``parent -> child`` (raises if it creates a cycle)."""
        self._ensure(parent)
        self._ensure(child)
        self._children[parent].add(child)
        self._parents[child].add(parent)
        if self._creates_cycle():
            self._children[parent].discard(child)
            self._parents[child].discard(parent)
            raise ValueError(f"Edge {parent} -> {child} introduces a cycle (not a DAG)")
        return self

    def _creates_cycle(self):
        color = {}  # 0=visiting, 1=done

        def visit(n):
            color[n] = 0
            for c in self._children[n]:
                if color.get(c) == 0:
                    return True
                if color.get(c) is None and visit(c):
                    return True
            color[n] = 1
            return False

        return any(color.get(n) is None and visit(n) for n in self._parents)

    @property
    def nodes(self):
        return set(self._parents)

    @property
    def edges(self):
        return [(p, c) for p, cs in self._children.items() for c in cs]

    def parents(self, node):
        return set(self._parents.get(node, set()))

    def children(self, node):
        return set(self._children.get(node, set()))

    # -- ancestry -------------------------------------------------------------
    def ancestors(self, nodes):
        if isinstance(nodes, str):
            nodes = {nodes}
        seen, stack = set(), list(nodes)
        while stack:
            n = stack.pop()
            for p in self._parents.get(n, set()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def descendants(self, nodes):
        if isinstance(nodes, str):
            nodes = {nodes}
        seen, stack = set(), list(nodes)
        while stack:
            n = stack.pop()
            for c in self._children.get(n, set()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    # -- paths ----------------------------------------------------------------
    def _neighbors(self, node):
        return self._parents[node] | self._children[node]

    def _all_paths(self, x, y):
        """Enumerate simple undirected paths between x and y (as node lists)."""
        paths, stack = [], [(x, [x])]
        while stack:
            node, path = stack.pop()
            for nb in self._neighbors(node):
                if nb in path:
                    continue
                if nb == y:
                    paths.append(path + [nb])
                else:
                    stack.append((nb, path + [nb]))
        return paths

    def _edge_dir(self, a, b):
        """Return '->' if a is a parent of b, else '<-'."""
        return "->" if b in self._children[a] else "<-"

    def _is_open(self, path, given):
        """Is an undirected path d-connected (open) given the set ``given``?"""
        given = set(given)
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            left = self._edge_dir(a, b)    # direction of a-b at b
            right = self._edge_dir(b, c)   # direction of b-c at b
            is_collider = (left == "->" and right == "<-")
            if is_collider:
                # Collider blocks unless b or a descendant of b is conditioned on.
                if b not in given and not (self.descendants(b) & given):
                    return False
            else:
                # Chain or fork: blocked iff the middle node is conditioned on.
                if b in given:
                    return False
        return True

    def d_separated(self, x, y, given=None):
        """True iff x and y are d-separated (conditionally independent) given ``given``."""
        given = set(given or [])
        return not any(self._is_open(p, given) for p in self._all_paths(x, y))

    # -- back-door criterion --------------------------------------------------
    def backdoor_paths(self, x, y):
        """All paths from x to y that start with an arrow *into* x (confounding paths)."""
        return [p for p in self._all_paths(x, y)
                if len(p) > 1 and self._edge_dir(p[0], p[1]) == "<-"]

    def is_valid_adjustment_set(self, x, y, z):
        """Check the back-door criterion for adjustment set ``z`` w.r.t. effect x -> y."""
        z = set(z)
        if z & ({x, y} | self.descendants(x)):
            return False  # must not contain X, Y or any descendant of X
        return all(not self._is_open(p, z) for p in self.backdoor_paths(x, y))

    def adjustment_sets(self, x, y, max_size=None, minimal=True):
        """Find adjustment sets satisfying the back-door criterion for x -> y.

        Args:
            max_size: Cap on the size of candidate sets (default: all candidates).
            minimal: If True, return only minimal (non-redundant) valid sets.

        Returns:
            List of sets (smallest first). ``[set()]`` means no adjustment is
            needed; ``[]`` means no valid set exists (effect not identifiable by
            back-door adjustment).
        """
        candidates = self.nodes - {x, y} - self.descendants(x)
        candidates = sorted(candidates)
        hi = len(candidates) if max_size is None else min(max_size, len(candidates))
        valid = []
        for k in range(hi + 1):
            for combo in combinations(candidates, k):
                s = set(combo)
                if self.is_valid_adjustment_set(x, y, s):
                    if minimal and any(v <= s for v in valid):
                        continue  # superset of an already-found valid set
                    valid.append(s)
        return valid

    # -- intervention ---------------------------------------------------------
    def do(self, x):
        """Return the mutilated graph for ``do(x)`` (all arrows into x removed)."""
        if isinstance(x, str):
            x = {x}
        g = DAG()
        for node in self.nodes:
            g._ensure(node)
        for p, c in self.edges:
            if c in x:
                continue  # cut incoming arrows to intervened nodes
            g.add_edge(p, c)
        return g

    # -- diagnostics ----------------------------------------------------------
    def check(self, x, y, given=None):
        """Warn about common back-door mistakes when conditioning on ``given``.

        Flags: conditioning on a collider (opens a spurious path), on a mediator
        (blocks part of the causal effect), or on a descendant of X.
        Returns a list of human-readable warnings (empty if the set looks safe).
        """
        given = set(given or [])
        warnings = []
        descendants_x = self.descendants(x)

        bad = given & descendants_x
        if bad:
            warnings.append(
                f"conditioning on {bad} (descendant(s) of {x}) can bias the effect")

        # Mediator: a node on a directed path x ~> y.
        on_causal = self.descendants(x) & self.ancestors(y)
        med = given & on_causal
        if med:
            warnings.append(
                f"conditioning on {med} blocks a directed path {x} ~> {y} "
                f"(mediator/over-control)")

        # Collider: conditioning opens a path that was closed when unconditioned.
        for node in given:
            without = given - {node}
            opened = [p for p in self._all_paths(x, y)
                      if self._is_open(p, given) and not self._is_open(p, without)]
            if opened:
                warnings.append(
                    f"conditioning on '{node}' opens a spurious (collider) path "
                    f"between {x} and {y}")
        return warnings

    # -- visualisation --------------------------------------------------------
    def _layers(self):
        """Assign each node a layer = longest directed distance from a root."""
        layer = {}
        order = self._topo_order()
        for n in order:
            ps = self._parents[n]
            layer[n] = 0 if not ps else 1 + max(layer[p] for p in ps)
        return layer

    def _topo_order(self):
        indeg = {n: len(self._parents[n]) for n in self.nodes}
        queue = sorted([n for n, d in indeg.items() if d == 0])
        order = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in sorted(self._children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        return order

    def plot(self, x=None, y=None, adjust=None, figsize=(7, 5), title=None):
        """Draw the DAG with matplotlib.

        Args:
            x, y: Optionally highlight the exposure (blue) and outcome (green).
            adjust: Iterable of nodes to mark as adjusted-for (grey, boxed).
            figsize: Figure size.
            title: Optional title.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        adjust = set(adjust or [])
        layer = self._layers()
        by_layer = {}
        for n, l in layer.items():
            by_layer.setdefault(l, []).append(n)

        pos = {}
        for l, nodes in by_layer.items():
            nodes = sorted(nodes)
            n = len(nodes)
            for i, node in enumerate(nodes):
                ypos = (n - 1) / 2 - i  # center the column vertically
                pos[node] = (l * 2.0, ypos)

        fig, ax = plt.subplots(figsize=figsize)
        for p, c in self.edges:
            x0, y0 = pos[p]
            x1, y1 = pos[c]
            # curve edges that skip layers or stay on the same row so they
            # don't hide behind intermediate nodes (e.g. confounder triangles).
            span = abs(layer[c] - layer[p])
            same_row = (y0 == y1)
            rad = 0.0
            if span > 1 or (same_row and span == 0):
                rad = 0.3 * span if span > 1 else 0.4
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                shrinkA=16, shrinkB=16, color="#555555", lw=1.4, zorder=1,
                connectionstyle=f"arc3,rad={rad}")
            ax.add_patch(arrow)

        for node, (px, py) in pos.items():
            if node == x:
                fc, ec = "#9ecae1", "#3182bd"
            elif node == y:
                fc, ec = "#a1d99b", "#31a354"
            elif node in adjust:
                fc, ec = "#d9d9d9", "#969696"
            else:
                fc, ec = "white", "#333333"
            boxstyle = "square,pad=0.4" if node in adjust else "circle,pad=0.4"
            ax.text(px, py, node, ha="center", va="center", fontsize=11, zorder=2,
                    bbox=dict(boxstyle=boxstyle, fc=fc, ec=ec, lw=1.6))

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return ax

    def __repr__(self):
        e = ", ".join(f"{p} -> {c}" for p, c in self.edges)
        return f"DAG({e})"


class causal:
    """High-level entry point exposed as ``m.causal`` on a BF instance."""

    DAG = DAG

    def dag(self, spec=None):
        """Build a :class:`DAG` from a spec string or edge list."""
        return DAG(spec)
