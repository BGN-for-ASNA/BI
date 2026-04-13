import os
import numpyro
numpyro.set_host_device_count(4)
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from BI import bi

# Add parent directory to path for tree_data.py
sys.path.append('..')
sys.path.append('.')
from tree_data import get_tree_data

m = bi(platform='cpu')

# Load Real Data
# leaf_likelihoods shape: (N_taxa, L, 4)
leaf_likelihoods = jnp.load("..\\primate_data.npy")
N_taxa, L, _ = leaf_likelihoods.shape

