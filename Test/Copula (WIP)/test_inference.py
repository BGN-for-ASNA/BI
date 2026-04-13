# %%
import jax.numpy as jnp
from jax import random
from scipy import stats
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from BI import bi

m = bi("cpu")
seed = 123


class BetaPoissonMarginal(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, alpha, beta, lam):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        super(BetaPoissonMarginal, self).__init__(batch_shape=(2,), event_shape=())

    def log_prob(self, value):
        v1 = value[..., 0]
        v2 = value[..., 1]
        lp1 = dist.Beta(self.alpha, self.beta).log_prob(v1)
        lp2 = dist.Poisson(self.lam).log_prob(v2)
        return jnp.stack([lp1, lp2], axis=-1)

    def cdf(self, value):
        v1 = value[..., 0]
        v2 = value[..., 1]
        cdf1 = dist.Beta(self.alpha, self.beta).cdf(v1)
        cdf2 = dist.Poisson(self.lam).cdf(v2)
        return jnp.stack([cdf1, cdf2], axis=-1)

    def icdf(self, u):
        u1 = np.asarray(u[..., 0])
        u2 = np.asarray(u[..., 1])
        x_b = stats.beta.ppf(u1, a=np.asarray(self.alpha), b=np.asarray(self.beta))
        x_p = stats.poisson.ppf(u2, mu=np.asarray(self.lam))
        return jnp.array(np.stack([x_b, x_p], axis=-1))


# Generate data
n = 500
Sigma = jnp.array([[1.0, 0.4], [0.4, 1.0]])
Sigma_chol = jnp.linalg.cholesky(Sigma)

samples_bi = m.dist.gaussian_copula(
    marginal_dist=BetaPoissonMarginal(2.0, 5.0, 3.0),
    correlation_cholesky=Sigma_chol,
    sample=True,
    shape=(n,),
    seed=seed,
)
x_b1 = samples_bi[:, 0]
x_b2 = samples_bi[:, 1]


def model(x_b1, x_b2):
    # Priors for marginal distributions
    alpha = m.dist.exponential(0.1, name="alpha", sample=True)
    beta = m.dist.exponential(0.1, name="beta", sample=True)
    lam = m.dist.exponential(0.1, name="lam", sample=True)

    # Prior for correlation
    rho = m.dist.lkj_cholesky(2, 2.0, name="rho", sample=True)

    obs_data = jnp.stack([x_b1, x_b2], axis=-1)

    m.dist.gaussian_copula(
        marginal_dist=BetaPoissonMarginal(alpha, beta, lam),
        correlation_cholesky=rho,
        obs=obs_data,
        name="obs",
    )


m.data_on_model = {"x_b1": x_b1, "x_b2": x_b2}
m.fit(model, num_samples=300, num_warmup=100, num_chains=1, progress_bar=False)

m.summary()


# %%
