#%%
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions
from main import*
num_vars = 2        # Number of variables (`n` in formula).
var_dim = 1         # Dimensionality of each variable `x[i]`.
num_components = 2  # Number of components for each mixture (`K` in formula).
sigma = 0.1       # Fixed standard deviation of each component.

# Set seed.
tf.random.set_seed(77)
def sim(N = 1000, num_vars = 2, var_dim = 1, num_components = 2, sigma = 0.1 , mean_components = 0, std_components = 1):
    """Draw a mixture of distribution following the formula: $P(x) = sum{P(c==i) P(x|c = i)}$

    Args:
        N (int, optional): Number of samples. Defaults to 1000.
        num_vars (int, optional): Number of variables (`n` in formula). Defaults to 2.
        var_dim (int, optional): Dimensionality of each variable `x[i]`. Defaults to 1.
        num_components (int, optional): umber of components for each mixture (`K` in formula). Defaults to 2.
        sigma (float, optional): Fixed standard deviation of each component. Defaults to 0.1.
        mean_components (int, optional): Mean of the normal distribution for components. Defaults to 0.
        std_components (int, optional): Standard deviation of the normal distribution for components. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # categorical distribution
    categorical = tfd.Categorical(logits=tf.zeros([num_vars, num_components]))

    # Choose some random (component) modes.
    component_mean = tfd.Normal(0,1).sample([num_vars, num_components, var_dim])

    # component distribution for the mixture family
    components = tfd.MultivariateNormalDiag(loc=component_mean, scale_diag=[sigma])

    # create the mixture same family distribution
    distribution_family = tfd.MixtureSameFamily(mixture_distribution=categorical, components_distribution=components)

    # Combine the distributions
    mixture_distribution = tfd.Independent(distribution_family, reinterpreted_batch_ndims=1)

    # Extract a sample from the distribution
    samples = mixture_distribution.sample(N).numpy()

    # Plot the distributions
    g = sns.jointplot(x=samples[:, 0, 0], y=samples[:, 1, 0], kind="scatter", color='blue', marginal_kws=dict(bins=50))
    plt.show()

    return samples

sim()
# %%
