import os 
import re
def set_deallocate():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    
def setup_device(platform='cpu', cores=None, deallocate = False, print_devices_found = True):
    """## Configures JAX for distributed computation.

    This function sets up the JAX computing environment by specifying the hardware 
    platform and managing CPU core allocation. It also handles deallocation of existing
    devices and configures XLA flags appropriately.

    ### Args:
    
    - *platform:* str, optional
        The hardware platform to use for computation. Options include:
        - 'cpu': Use CPU(s) for computation
        - 'gpu': Use GPU(s) for computation
        - 'tpu': Use TPU(s) for computation
        Defaults to 'cpu'.

    - *cores:* int, optional
        Number of CPU cores to allocate for computation. If None, all available CPU 
        cores will be used. Only applicable when platform is 'cpu'.
        
    - *deallocate:* bool, optional
        Whether to deallocate any existing devices before setting up new configuration.
        Defaults to False.

    ### Notes
    This function must be called before any JAX imports or usage. It configures the 
    XLA_FLAGS environment variable to specify the number of CPU cores to use. The 
    XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT flag is particularly important for properly 
    distributing computation across multiple CPUs.

     ### Examples

    Basic usage:
    >>> setup_device(platform='cpu')

    Specifying CPU cores:
    >>> setup_device(platform='cpu', cores=4)

     ### Returns
     *None*
    """
    if cores is None:
        cores = os.cpu_count()
    if deallocate:
        set_deallocate()

    # Set the XLA flags before importing jax
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(cores)] + xla_flags)

    # Now import jax
    import jax as jax

    # Explicitly update the configuration after import
    jax.config.update("jax_platform_name", platform)


    if print_devices_found:
        print('jax.local_device_count', jax.local_device_count(backend=None))
