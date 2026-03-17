
from BI import bi
import jax.numpy as jnp
import jax
import numpy as np
from jax.nn import softmax

# Setup device
m = bi(platform="cpu")

# Simulated data
np.random.seed(42)
N = 200
X = np.random.normal(0, 1, size=N)

# True parameters
true_alpha = np.array([0.5, -0.5])
true_beta = np.array([1.0, 0.5])
true_kappa = 10.0

# Linear predictors
phi = np.zeros((N, 3))
phi[:, 0] = true_alpha[0] + true_beta[0] * X
phi[:, 1] = true_alpha[1] + true_beta[1] * X
phi[:, 2] = 0.0 # Reference

# Probabilities (theta)
theta = softmax(phi, axis=1)

# Observations (Y)
Y = np.zeros((N, 3))
for i in range(N):
    Y[i, :] = np.random.dirichlet(theta[i, :] * true_kappa)

print(f"True alpha: {true_alpha}")
print(f"True beta: {true_beta}")
print(f"True kappa: {true_kappa}")

# Model data
def model_dirichlet(X, Y):
    alpha = m.dist.normal(0, 5, shape=(2,), name="alpha")
    beta = m.dist.normal(0, 5, shape=(2,), name="beta")
    kappa = m.dist.exponential(1.0, name="kappa")
    
    s1 = alpha[0] + beta[0] * X
    s2 = alpha[1] + beta[1] * X
    s3 = jnp.zeros_like(s1)
    
    phi = jnp.stack([s1, s2, s3], axis=1)
    theta = jax.nn.softmax(phi, axis=1)
    
    m.dist.dirichlet(theta * kappa, obs=Y)

m.data_on_model = dict(X=X, Y=Y)
m.fit(model_dirichlet, num_samples=1000, num_warmup=500, progress_bar=False)

summ = m.summary()
print("\nPosterior Means:")
print(summ[['mean']])

# Check accuracy
# alpha[0] should be around 0.5, alpha[1] around -0.5
# beta[0] should be around 1.0, beta[1] around 0.5
# kappa should be around 10.0
