from BI import bi, jnp
import numpy as np

m = bi(platform='cpu')
N = 3
mat = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]])
edgl = m.net.mat_to_edgl(jnp.array(mat))
print(f"Matrix:\n{mat}")
print(f"Edgelist:\n{edgl}")
# This should print [1, 2, 3, 4, 5, 6] if row-major off-diagonal
