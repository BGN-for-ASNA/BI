import os 
import re
def deallocate():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    
def setup(platform='cpu', cores=None, deallocate = False):
    """
    Configures JAX for distributed computation.

    Args:
        platform (str): The platform to use for computation. Default is 'cpu'.
        cores (int): Number of CPU cores to use. If None, it defaults to the total number of available CPU cores.

    Returns:
        None
    """
    if cores is None:
        cores = os.cpu_count()
    if deallocate:
        deallocate()

    # Set the XLA flags before importing jax
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(cores)] + xla_flags)

    # Now import jax
    import jax as jax

    # Explicitly update the configuration after import
    jax.config.update("jax_platform_name", platform)



    print('jax.local_device_count', jax.local_device_count(backend=None))
