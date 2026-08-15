import jax.numpy as jnp
from jax import random
from scipy import stats
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from BayesForge import BayesForge

# PATCH NORMAL ICDF TO AVOID TFP
import jax.scipy.special as jss
dist.Normal.icdf = lambda self, q: jss.ndtri(jnp.clip(q, 1e-6, 1.0 - 1e-6))

m = BF('cpu')
seed = 123

class BetaPoissonMarginal(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, alpha, beta, lam, validate_args=None):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        super().__init__(batch_shape=(2,), event_shape=(), validate_args=validate_args)

    def log_prob(self, value):
        v1 = value[..., 0]
        v2 = value[..., 1]
        lp1 = dist.Beta(self.alpha, self.beta).log_prob(v1)
        lp2 = dist.Poisson(self.lam).log_prob(v2)
        return jnp.stack([lp1, lp2], axis=-1)

    def cdf(self, value):
        import jax.scipy.special as jss
        v1 = value[..., 0]
        v2 = value[..., 1]
        cdf1 = jss.betainc(self.alpha, self.beta, v1)
        cdf2 = jss.gammaincc(jnp.floor(v2) + 1.0, self.lam)
        return jnp.stack([cdf1, cdf2], axis=-1)
        
    def icdf(self, u):
        u1 = np.asarray(u[..., 0])
        u2 = np.asarray(u[..., 1])
        x_b = stats.beta.ppf(u1, a=np.asarray(self.alpha), b=np.asarray(self.beta))
        x_p = stats.poisson.ppf(u2, mu=np.asarray(self.lam))
        return jnp.stack([jnp.array(x_b), jnp.array(x_p)], axis=-1)

# Generate data
n = 500
Sigma_chol = jnp.linalg.cholesky(jnp.array([[1.0, 0.4], [0.4, 1.0]]))

samples_BF = m.dist.gaussian_copula(
    marginal_dist=BetaPoissonMarginal(2.0, 5.0, 3.0),
    correlation_cholesky=Sigma_chol,
    sample=True,
    shape=(n,),
    seed=seed
)
x_b1 = samples_BF[:, 0]
x_b2 = samples_BF[:, 1]

def model(x_b1, x_b2):
    alpha = m.dist.exponential(0.1, name='alpha', sample=True)
    beta = m.dist.exponential(0.1, name='beta', sample=True)
    lam = m.dist.exponential(0.1, name='lam', sample=True)
    rho = m.dist.lkj_cholesky(2, 2.0, name='rho', sample=True)
    obs_data = jnp.stack([x_b1, x_b2], axis=-1)
    
    m.dist.gaussian_copula(
        marginal_dist=BetaPoissonMarginal(alpha, beta, lam),
        correlation_cholesky=rho,
        obs=obs_data,
        name='obs'
    )

m.data_on_model = {'x_b1': x_b1, 'x_b2': x_b2}
m.fit(model, num_samples=10, num_warmup=10, num_chains=1, progress_bar=False)
print("SUCCESS!")
