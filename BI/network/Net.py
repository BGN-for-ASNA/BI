from BI.Network.metrics import met
from BI.Network.util import array_manip 
from BI.Network.model_effects import Neteffect 
import jax.numpy as jnp

class net(met, Neteffect, array_manip):
    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)
        # Additional initialization code if needed
