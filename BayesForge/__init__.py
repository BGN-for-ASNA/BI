# JAX compatibility monkey-patch for older NumPyro versions
try:
    import jax.interpreters.pxla
    if not hasattr(jax.interpreters.pxla, "xla_pmap_p"):
        jax.interpreters.pxla.xla_pmap_p = getattr(jax.interpreters.pxla, "pmap_p", None)
except ImportError:
    pass

from .Main.main import bf, BF
from .SetDevice.set import setup_device
from .Resources.datasets import load
import importlib.metadata

BayesForge = BF

try:
    __version__ = importlib.metadata.version("BayesForge")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

print(f"bf v {__version__} package loaded")
__all__ = ["bf", "BF", "BayesForge", "setup_device", "load"]
