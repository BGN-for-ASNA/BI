"""2.5 -- execute the Python examples of every model page of the documentation.

Same rule as tests/test_doc_examples.py (Get started pages): all ```python
blocks of a page are concatenated and run in a fresh interpreter, so a page
fails as soon as a copy-pasted example would fail for a reader. Blocks fenced
as ```{.python .no-test} or containing ``# doc-test: skip`` are left out.

These pages fit real models (minutes per page), so the test has its own
marker, ``pages``: the pull-request CI skips it (-m "not pages") and the
weekly workflow .github/workflows/doc-pages.yml runs all pages. Locally,
pick pages with -k, e.g.
    pytest tests/test_model_pages.py -m pages -k "Poisson"
"""
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_doc_examples import python_blocks  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "Documentation"
NOT_MODEL_PAGES = {"index", "get started", "0.Introduction"}
PAGES = sorted(p for p in DOCS.glob("*.qmd") if p.stem not in NOT_MODEL_PAGES)


@pytest.mark.pages
@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_model_page_python_examples_run(page, tmp_path):
    blocks = python_blocks(page)
    if not blocks:
        pytest.skip("no python blocks")
    script = tmp_path / "page.py"
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(blocks))
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
