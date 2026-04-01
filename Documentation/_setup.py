import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.append("../BI")
from BI import bi


class LastOnlyTQDM(tqdm_auto):
    def display(self, msg=None, pos=None):
        # Only display once the bar reaches 100%
        if self.n != self.total:
            return
        super().display(msg, pos)

# Patch NumPyro's tqdm globally
numpyro.util.tqdm = LastOnlyTQDM



import sys
import os
sys.stdout = open(os.devnull, 'w')

