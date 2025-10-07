from BI.Models.surv import survival
from BI.NBDA.NBDA import NBDA
from BI.Models.GMM import *
from BI.Models.DPMM import *
from BI.Models.PCA import *


class models():
    """
    The models class serves as a high-level interface for managing and utilizing various Bayesian models within the BI framework. It encapsulates different model types such as DPMM, GMM, NBDA, and others, providing a unified structure for model initialization, fitting, and diagnostics. This class is designed to simplify the process of working with complex Bayesian models, allowing users to easily switch between different model types and access their functionalities through a consistent API.
    """
    def __init__(self,parent):
        """
        Initialize the models class. Currently empty but can be extended for initialization needs.
        """
        pass
        self.gmm = gmm
        self.dpmm = dpmm
        self.pca =  pca
        self.nbda = NBDA.model
        self.survival = survival(parent)

        self.available = {
            "gmm": self.gmm,
            "dpmm": self.dpmm,
            "nbda": self.nbda,
            "pca": self.pca,
            "survival": self.survival
        }


    