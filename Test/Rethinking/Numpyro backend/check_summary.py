from BI import bi
from jax import numpy as jnp

m = bi(platform='cpu', backend='numpyro')

def model_bi(y):
    a = m.dist.normal(0, 1, shape=(2,), name='a')
    m.dist.normal(a[0], 1, obs=y)

y = jnp.array([1.0, 1.1, 0.9])
m.fit(model_bi, y=y, num_samples=100)
print("Summary Index List:")
print(m.summary().index.tolist())
