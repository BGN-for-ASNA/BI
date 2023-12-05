# %% 
import tensorflow_probability as tfp
tfd = tfp.distributions

# Generate a dict of all tensorflow distribution available --------------------------------------------------------
distributions = [attr for attr in dir(tfp.distributions) if callable(getattr(tfp.distributions, attr))]

distribution_dict = {}
for dist in distributions:
    distribution_dict[dist] = getattr(tfd, dist)

# Use dictionary to query the distribution and their arguments --------------------------------------------------------
def dist(distribution, **kwargs):
    """_summary_
        Apply the distribution function and their arguments
    Args:
        distribution (str): string that match tensorflow available distributions.

    Raises:
        ValueError: _description_

    Returns:
        _type_: Tensorflow probability distribution with the provided arguments.
    """
    if distribution in distribution_dict:
        distribution_class = distribution_dict[distribution]
        return distribution_class(**kwargs)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

# Return tensorflow probability distribution required arguments --------------------------------------------------------
def arguments (distribution = 'Normal', description = False):
    """_summary_
        Return tensorflow probability distribution required arguments
    Args:
        distribution (str, optional): Distribution ot query. Defaults to 'Normal'.
        description (bool, optional): Get arguments details. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: Dictionary of required arguments for the given distribution
    """
    import inspect
    if distribution in distribution_dict:
        distribution_class = distribution_dict[distribution]
        if description:
            signature = distribution_class.__init__.__doc__
            signature = signature.split('\n')
        else:
            signature = inspect.signature(distribution_class.__init__)
            signature = list(signature.parameters.keys())
            del signature[0]
        return signature
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
