"""Run the "Example Usage" snippet of every distribution docstring (m.dist.*).

These snippets are copied by users and are also used to generate the R help
pages, so each one must run as written.
"""
import inspect
import re

import matplotlib
import pytest

matplotlib.use("Agg")

from BayesForge.Distributions.np_dists import UnifiedDist as np_dists

HEADER = re.compile(r"^\s*#+\s*Example Usage:?\s*$", re.M)
# Docstring section headers are "####"-style; a single "#" is a code comment.
NEXT_SECTION = re.compile(r"^\s*(#{2,}\s*\w|References\b|\.\. )", re.M)


def example_of(func):
    doc = inspect.getdoc(func) or ""
    match = HEADER.search(doc)
    if not match:
        return None
    body = doc[match.end():]
    stop = NEXT_SECTION.search(body)
    body = body[: stop.start()] if stop else body
    return inspect.cleandoc(body) or None


EXAMPLES = {
    name: code
    for name, func in inspect.getmembers(np_dists, callable)
    if not name.startswith("_") and (code := example_of(func))
}


def test_examples_found():
    assert len(EXAMPLES) > 50


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_distribution_example_runs(name):
    exec(compile(EXAMPLES[name], f"<m.dist.{name} example>", "exec"), {})
