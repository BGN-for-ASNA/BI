"""Pytest configuration for the BayesForge test tree.

Puts the repository root on sys.path so tests can `import BayesForge` without an
installed package or a per-file sys.path hack, and forces JAX onto CPU: this
machine's CuDNN runtime (9.1.0) is older than the one jaxlib was built against
(9.8.0), so any GPU-backed op fails with
"FAILED_PRECONDITION: DNN library initialization failed".
"""
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
