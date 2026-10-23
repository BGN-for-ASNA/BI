"""Shim — the implementation now lives in the package.

    from BayesForge.Network.explorer import NetExplorer, NetworkView

Kept so ``test_netexplorer.py`` (and any old ``import netexplorer``) still
resolve. Reach it in normal use as ``m.net.viz`` (see
``BayesForge/Network/Net.py``).
"""

from BayesForge.Network.explorer import (  # noqa: F401
    NetExplorer,
    NetworkView,
    _data_uri,
    _in_notebook,
    _is_wsl,
    _open_in_browser,
)

__all__ = ["NetExplorer", "NetworkView"]
