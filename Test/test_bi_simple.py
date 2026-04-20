from BI import bi
import jax.numpy as jnp
import numpyro

def simple_model(y):
    m = bi()
    mu = m.dist.normal(0, 1, name='mu')
    m.dist.normal(mu, 1, name='obs', obs=y)

def test():
    m = bi(platform='cpu')
    y = jnp.array([1.0, 1.2, 0.8])
    m.fit(model=simple_model, obs={'y': y}, num_warmup=100, num_samples=100, num_chains=1)
    print("Simple model fit complete.")
    print(m.summary())

if __name__ == "__main__":
    test()
