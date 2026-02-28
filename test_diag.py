import jax.numpy as jnp
import numpy as np
import sys
# add local path to sys.path if needed
sys.path.append(r'c:\Users\Sosa\Documents\BI')

from BI.Diagnostic.fast_diagnostics import compute_diagnostics

# Dummy data: 4 chains, 3 dimensions, 1000 samples
# Shape: (4, 3, 1000)
key = jnp.array(np.random.normal(size=(4, 3, 1000)))
posterior = {"beta": key}

try:
    res = compute_diagnostics(posterior)
    print("Success!")
    print("Keys in result:", res["beta"].keys())
    print("Mean shape:", res["beta"]["mean"].shape)
    print("SD shape:", res["beta"]["sd"].shape)
    print("HDI lower shape:", res["beta"]["hdi_3%"].shape)
    print("HDI upper shape:", res["beta"]["hdi_97%"].shape)
    print("Rhat shape:", res["beta"]["rhat"].shape)
    print("ESS shape:", res["beta"]["ess"].shape)
    
    # Also test 4D data: (4, 2, 5, 1000)
    key2 = jnp.array(np.random.normal(size=(4, 2, 5, 1000)))
    res2 = compute_diagnostics({"gamma": key2})
    print("\nSuccess 4D!")
    print("ESS shape 4D:", res2["gamma"]["ess"].shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
