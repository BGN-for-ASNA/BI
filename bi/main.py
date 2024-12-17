import setup
import warnings
import arviz as az
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax as jax
import numpy as np
import jax.random as random

from setup.device import setup
from data.manip import manip
from utils.array import Mgaussian as gaussian
from utils.array import factors 
from network.net import net
from mcmc.Link import link
from mcmc.Model import model
from mcmc.Samplers import samplers
from diagnostic.Diag import diag

from utils.unified_dists import tfpLight as dist


import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import *
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import inspect
import re

class bi(manip, dist, gaussian, factors, net, link, samplers, diag):
    def __init__(self, platform='cpu', cores=None, deallocate = False, dtype = jnp.float32):
        setup(platform, cores, deallocate) 
        jax.config.update("jax_enable_x64", True)
        self.trace = None
        self.dtype = dtype
        super().__init__()

    # Dist functions (sampling and model)--------------------------
    class dist(dist):
        pass
    
    # Links functions--------------------------
    class link(link):
        pass

    # Network functions--------------------------
    class net(net):
        pass

    # Model building functions--------------------------
    class model(model):
        pass

    # MCMC samplers --------------------------
    class samplers(samplers):
        pass

    # MCMC samplers --------------------------
    class samplers(samplers):
        pass

    # Diag samplers --------------------------
    class diag(diag):
        pass
    

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import random as r
