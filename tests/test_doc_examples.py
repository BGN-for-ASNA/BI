"""Execute the Python code blocks of the "Get started" documentation pages.

Every ```python block of a page is concatenated (blocks build on each other)
and run in a fresh interpreter, so a page fails the test as soon as a copy-
pasted example would fail for a reader. Blocks fenced as ```{.python .no-test}
or ```python containing ``# doc-test: skip`` are left out (e.g. GPU-only code).

Run a single page with:  pytest tests/test_doc_examples.py -k "seeds"
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[1] / "Documentation"
PAGES = sorted((DOCS / "start").glob("*.qmd"))
# Matches ```python and ``` {python} fences (not R/Julia/shell tabs).
FENCE = re.compile(r"^```\s*\{?python[^\n]*\}?\s*\n(.*?)^```\s*$", re.M | re.S)


def python_blocks(page):
    blocks = []
    for match in FENCE.finditer(page.read_text()):
        header = match.group(0).splitlines()[0]
        code = match.group(1)
        if "no-test" in header or "# doc-test: skip" in code:
            continue
        blocks.append(code)
    return blocks


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_python_examples_run(page, tmp_path):
    blocks = python_blocks(page)
    if not blocks:
        pytest.skip("no python blocks")
    script = tmp_path / "page.py"
    # Headless plotting; never open windows from a test run.
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(blocks))
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
