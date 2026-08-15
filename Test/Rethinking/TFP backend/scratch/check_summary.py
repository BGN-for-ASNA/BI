from BayesForge import bf
import jax.numpy as jnp
import pandas as pd

m = bf(backend='tfp', platform='cpu')
m.df = pd.DataFrame({'y': [1.0, 2.0, 3.0]})
m.data_to_model(['y'])

def model(y):
    a = yield m.dist.normal(0, 1, name='a')
    yield m.dist.normal(a, 1, obs=y)

m.fit(model, num_samples=10, num_warmup=10, progress_bar=False)
summary = m.summary()
print("Summary index:")
print(summary.index.tolist())
