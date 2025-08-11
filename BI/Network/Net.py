from BI.Network.metrics import met
from BI.Network.util import array_manip 
from BI.Network.model_effects import Neteffect 
import jax.numpy as jnp

class net(met, Neteffect, array_manip):
    """The net class serves as a high-level interface for managing and utilizing various network metrics and effects within the BI framework. 
    It encapsulates functionalities for computing clustering coefficients, eigenvector centrality, Dijkstra's algorithm for shortest paths, and other network metrics. 
    Additionally, it extends the array_manip class to provide methods for handling network effects, including sender-receiver effects, dyadic effects, and block models.
    This class is designed to simplify the process of working with complex network structures, allowing users to easily compute metrics and model network interactions through a consistent API.
    """
    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)
        # Additional initialization code if needed
