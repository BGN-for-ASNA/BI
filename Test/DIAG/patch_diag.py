"""Re-export from canonical location in BI package."""
from BI.Diagnostic.patch_diag import *  # noqa: F401, F403
from BI.Diagnostic.patch_diag import (  # explicit for IDE support
    patch_diag_class,
    bind_diag_to_model,
    _get_posteriors,
    _expand,
)
