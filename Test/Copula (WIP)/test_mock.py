import jax.numpy as jnp
from jax import random
from scipy import stats
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions.copula import GaussianCopula

class BetaPoissonMarginal:
    @property
    def batch_shape(self):
        return (2,)
    @property
    def event_shape(self):
        return ()
    def icdf(self, u):
        u1 = np.asarray(u[..., 0])
        u2 = np.asarray(u[..., 1])
        x_b = stats.beta.ppf(u1, a=2.0, b=5.0)
        # Using Poisson mean=3.0
        x_p = stats.poisson.ppf(u2, mu=3.0)
        return jnp.stack([x_b, x_p], axis=-1)

key = random.PRNGKey(0)
Sigma = jnp.array([[1.0, 0.5], [0.5, 1.0]])
Sigma_chol = jnp.linalg.cholesky(Sigma)

cop = GaussianCopula(marginal_dist=BetaPoissonMarginal(), correlation_cholesky=Sigma_chol)
samples = cop.sample(key, sample_shape=(10,))
print("Samples from Builtin Copula with Beta+Poisson:", samples)
