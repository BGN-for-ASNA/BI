from network.metrics import met
from network.util import manip
from network.model_effects import Neteffect 
import jax.numpy as jnp

class net(met, Neteffect,manip):
    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)
        # Additional initialization code if needed
