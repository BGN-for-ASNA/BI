from .Main.main import bi
from .SetDevice.set import setup_device
from .Resources.datasets import load
import importlib.metadata

try:
    __version__ = importlib.metadata.version("BayesInference")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

print(f"BI v {__version__} package loaded")
__all__ = ["bi", "setup_device", "load"]
