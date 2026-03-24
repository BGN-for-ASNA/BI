import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def test_tfp():
    print("Testing TFP on JAX...")
    try:
        dist = tfd.Normal(loc=0., scale=1.)
        samples = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(10,))
        print("Samples:", samples)
        print("TFP test successful!")
    except Exception as e:
        print("TFP test failed!")
        print(e)

if __name__ == "__main__":
    test_tfp()
