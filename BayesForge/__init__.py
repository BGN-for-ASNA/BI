# JAX compatibility monkey-patch for older NumPyro versions
try:
    import jax.interpreters.pxla
    if not hasattr(jax.interpreters.pxla, "xla_pmap_p"):
        jax.interpreters.pxla.xla_pmap_p = getattr(jax.interpreters.pxla, "pmap_p", None)
except ImportError:
    pass

# ArviZ >=1.3 pulls NumPyro through lazy_loader; when arviz is imported before
# numpyro the `numpyro.infer.initialization` submodule ends up in sys.modules
# without being attached to its parent, so arviz_base.io_numpyro.infer_dims
# raises AttributeError. Re-attach it.
try:
    import sys as _sys
    import numpyro.infer as _npi
    import numpyro.infer.initialization  # noqa: F401
    if not hasattr(_npi, "initialization"):
        _npi.initialization = _sys.modules["numpyro.infer.initialization"]
except ImportError:
    pass

from .Main.main import bf, BF
from .SetDevice.set import setup_device
from .Resources.datasets import load
from .Parallel.simulate import grid, run_simulations
import importlib.metadata

BayesForge = BF

try:
    __version__ = importlib.metadata.version("BayesForge")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# BF_QUIET is set by run_simulations for its workers: with a pool of them the
# banner prints once per process and drowns the actual output.
import os as _os
if _os.environ.get("BF_QUIET", "0") != "1":
    print(f"bf v {__version__} package loaded")
__all__ = ["bf", "BF", "BayesForge", "setup_device", "load",
           "run_simulations", "grid"]
