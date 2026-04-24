import os
import sys

# Force JAX to use CPU and avoid GPU initialization
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
jax.config.update('jax_platform_name', 'cpu')

sys.path.append("../BI")
from BI import bi
import numpyro
from tqdm.auto import tqdm as tqdm_auto

m = bi("cpu")

# Restore stdout to see progress
# sys.stdout = open(os.devnull, "w")


