from Darray import *
# strength
def sum(x):
    return jax.numpy.sum(x)

def outstrength(x):
    return jax.numpy.sum(x, axis=1)

def instrength(x):
    return jax.numpy.sum(x, axis=0)

def vec_outstrength(matrix):
    """
    Compute row sums across a batch of matrices using JAX's vmap.

    Args:
        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).

    Returns:
        jax.interpreters.xla.DeviceArray: Array containing row sums for each matrix in the batch with shape (batch_size, num_rows).
    """
    # Use vmap to apply outstrength to each matrix in the batch
    return jax.vmap(sum, in_axes = 0)(matrix)

def vec_instrength(matrix):
    """
    Compute column sums across a batch of matrices using JAX's vmap.

    Args:
        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).

    Returns:
        jax.interpreters.xla.DeviceArray: Array containing column sums for each matrix in the batch with shape (batch_size, num_cols).
    """
    # Use vmap to apply instrength to each matrix in the batch
    return jax.vmap(sum, in_axes = 1)(matrix)

def para_outstrength(x):
    return jnp.sum(split_array_to_cores(x), axis = 1)

def para_instrength(x):
    return jnp.sum(split_array_to_cores(x), axis = 0)

# degree
# strength
def outdegree(x):
    mask = x != 0
    return jax.numpy.sum(mask, axis=1)

def indegree(x):
    mask = x != 0
    return jax.numpy.sum(mask, axis=0)

def vec_outdegree(matrix):
    """
    Compute row sums across a batch of matrices using JAX's vmap.

    Args:
        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).

    Returns:
        jax.interpreters.xla.DeviceArray: Array containing row sums for each matrix in the batch with shape (batch_size, num_rows).
    """
    mask = x != 0
    # Use vmap to apply outstrength to each matrix in the batch
    return jax.vmap(sum, in_axes=0)(mask)

def vec_indegree(matrix):
    """
    Compute column sums across a batch of matrices using JAX's vmap.

    Args:
        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).

    Returns:
        jax.interpreters.xla.DeviceArray: Array containing column sums for each matrix in the batch with shape (batch_size, num_cols).
    """
    # Use vmap to apply instrength to each matrix in the batch
    mask = x != 0
    return jax.vmap(sum, in_axes=1)(mask)

def para_outdegree(x):
    x = split_array_to_cores(x)
    return outdegree(x)

def para_indegree(x):
    x = split_array_to_cores(x)
    return indegree(x)


# Not working with @jit
@jax.jit
def power_iteration(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    x = jnp.ones(n) / jnp.sqrt(n)
    diff = 1.0  # initialize to a value greater than tol

    def body_fun(carry):
        x, i = carry
        Ax = jnp.dot(A, x)
        norm_Ax = jnp.linalg.norm(Ax)
        x_new = Ax / norm_Ax
        diff = jnp.linalg.norm(x_new - x)
        print(diff)
        return (x_new, i + 1), diff < tol

    carry = (x, 0)
    _, converged = jax.lax.while_loop(cond_fun = lambda carry: jnp.logical_not(converged),
                                  body_fun = body_fun,
                                  init_val = carry)

    return x

# You will need to implement `mat_row_wise_multiplication` and `euclidean` functions in JAX.
# Replace these functions with their JAX equivalents.


# Compute the eigenvector centrality of the network
#centrality = power_iteration(x)
#
## Print the eigenvector centrality
#print(centrality)
#
