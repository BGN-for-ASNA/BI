import os
import sys
import numpy as np
import jax.numpy as jnp

# Setup R environment
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ["R_HOME"] = R_HOME
os.environ["PATH"] = os.path.join(R_HOME, "bin", "x64") + ";" + os.environ["PATH"]

data_file = "Test/Capture-Recapture/cr-in-stan/case-studies/data/fleayi-stan-data.rds"
import rpy2.robjects as ro
full_data = ro.r['readRDS'](data_file)
y_raw = np.array(full_data.rx2('y'))
print("y_raw shape:", y_raw.shape)

# Site 1, individuals 1..400
y_site1 = y_raw[0, :400, :, :]
print("y_site1 (subset) shape:", y_site1.shape)

# Check sum over states to see how many states an individual can be in
sum_states = np.sum(y_site1, axis=2)
print("Unique states seen in capture history (last dim):")
for i in range(y_site1.shape[2]):
    seen = np.any(y_site1[:, :, i] == 1)
    print(f"  State {i+1} seen: {seen}")

# Collapse to categorical
y_cat = np.zeros((400, 21), dtype=int)
for s in range(y_site1.shape[2]):
    mask = y_site1[:, :, s] == 1
    y_cat[mask] = s + 1

print("Unique values in y_cat:", np.unique(y_cat))
