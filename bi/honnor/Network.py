from numpyro import sample as lk
from samplers import sampler
sample = sampler()
from dists import Dist
dist = Dist()
from Network import Net
net = Net()
from Mutils import Mgaussian
from Mutils import Mrandom
gaussian = Mgaussian()
random = Mrandom()
from jax import vmap

#from Darray import *
from functools import partial
import jax as jax
import jax.numpy as jnp
from jax import jit

# vector related functions -----------------------------------
@partial(jit, static_argnums=(1, 2,))
def vec_to_mat_jax(arr, N, K):
    return jnp.reshape(arr, (N, K))

# Matrices related functions ------------------------------------------------------------------
def upper_tri(array, diag=1):
    """Extracts the upper triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=1 excludes the diagonal, diag=0 includes it.
    """
    upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
    upper_triangle_elements = array[upper_triangle_indices]
    return upper_triangle_elements
# JIT compile the function with static_argnums
get_upper_tri = jit(upper_tri, static_argnums=(1,))


def lower_tri(array, diag=-1):
    """Extracts the lower triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=0 includes the diagonal, diag=-1 excludes it.
    """
    lower_triangle_indices = jnp.tril_indices(array.shape[0], k=diag)
    lower_triangle_elements = array[lower_triangle_indices]
    return lower_triangle_elements
# JIT compile the function with static_argnums
get_lower_tri = jit(lower_tri, static_argnums=(1,))

def get_tri(array, type='upper', diag=0):
    """Extracts the upper, lower, or both triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        type (str): A string indicating which part of the triangle to extract.
                    It can be 'upper', 'lower', or 'both'.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=1 excludes the diagonal, diag=0 includes it.

    Returns:
        If argument type is 'upper', 'lower', it return a 1D JAX array containing the requested triangle elements.
        If argument type is 'both', it return a 2D JAX array containing the the first column the lower triangle and in the second ecolumn the upper triangle
    """
    if type == 'upper':
        upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
        triangle_elements = array[upper_triangle_indices]
    elif type == 'lower':
        lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
        triangle_elements = array[lower_triangle_indices]
    elif type == 'both':
        upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
        lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
        upper_triangle_elements = array[upper_triangle_indices]
        lower_triangle_elements = array[lower_triangle_indices]
        triangle_elements = jnp.stack((upper_triangle_elements,lower_triangle_elements), axis = 1)
    else:
        raise ValueError("type must be 'upper', 'lower', or 'both'")

    return triangle_elements

    
@jit
def mat_to_edgl_jax(mat):
    N = mat.shape[0]
    # From to 
    urows, ucols   = jnp.triu_indices(N, k=1)
    ft = mat[(urows,ucols)]
    m2 = jnp.transpose(mat)
    tf = m2[(urows,ucols)]
    return jnp.stack([ft, tf], axis = -1)


## strength ------------------------------------------
#def sum(x):
#    return jax.numpy.sum(x)
#
#@jit
#def outstrength_jax(x):
#    return jax.numpy.sum(x, axis=1)
#
#@jit
#def instrength_jax(x):
#    return jax.numpy.sum(x, axis=0)
#
#@jit
#def strength_jax(x):
#    return outstrength_jax(x) +  instrength_jax(x)
#
#def vec_outstrength(matrix):
#    """
#    Compute row sums across a batch of matrices using JAX's vmap.
#
#    Args:
#        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).
#
#    Returns:
#        jax.interpreters.xla.DeviceArray: Array containing row sums for each matrix in the batch with shape (batch_size, num_rows).
#    """
#    # Use vmap to apply outstrength to each matrix in the batch
#    return jax.vmap(sum, in_axes = 0)(matrix)
#
#def vec_instrength(matrix):
#    """
#    Compute column sums across a batch of matrices using JAX's vmap.
#
#    Args:
#        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).
#
#    Returns:
#        jax.interpreters.xla.DeviceArray: Array containing column sums for each matrix in the batch with shape (batch_size, num_cols).
#    """
#    # Use vmap to apply instrength to each matrix in the batch
#    return jax.vmap(sum, in_axes = 1)(matrix)
#
#def para_outstrength(x):
#    return jnp.sum(split_array_to_cores(x), axis = 1)
#
#def para_instrength(x):
#    return jnp.sum(split_array_to_cores(x), axis = 0)
#
## degree ------------------------------------------
#def outdegree(x):
#    mask = x != 0
#    return jax.numpy.sum(mask, axis=1)
#
#def indegree(x):
#    mask = x != 0
#    return jax.numpy.sum(mask, axis=0)
#
#def vec_outdegree(matrix):
#    """
#    Compute row sums across a batch of matrices using JAX's vmap.
#
#    Args:
#        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).
#
#    Returns:
#        jax.interpreters.xla.DeviceArray: Array containing row sums for each matrix in the batch with shape (batch_size, num_rows).
#    """
#    mask = x != 0
#    # Use vmap to apply outstrength to each matrix in the batch
#    return jax.vmap(sum, in_axes=0)(mask)
#
#def vec_indegree(matrix):
#    """
#    Compute column sums across a batch of matrices using JAX's vmap.
#
#    Args:
#        matrices (jax.interpreters.xla.DeviceArray): Batch of matrices with shape (batch_size, num_rows, num_cols).
#
#    Returns:
#        jax.interpreters.xla.DeviceArray: Array containing column sums for each matrix in the batch with shape (batch_size, num_cols).
#    """
#    # Use vmap to apply instrength to each matrix in the batch
#    mask = x != 0
#    return jax.vmap(sum, in_axes=1)(mask)
#
#def para_outdegree(x):
#    x = split_array_to_cores(x)
#    return outdegree(x)
#
#def para_indegree(x):
#    x = split_array_to_cores(x)
#    return indegree(x)

# Eigenvector ------------------------------------------
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

class Net:
    def __init__(self) -> None:
        pass

    # Matrix manipulations -------------------------------------
    @staticmethod 
    @partial(jit, static_argnums=(1, 2,))
    def vec_to_mat(arr, N, K):
        return jnp.reshape(arr, (N, K))

    def get_tri(self, array, type='upper', diag=0):
        return get_tri(array, type=type, diag=diag)
    
    @staticmethod 
    @jit
    def mat_to_edgl(mat):
        N = mat.shape[0]
        # From to 
        urows, ucols   = jnp.triu_indices(N, k=1)
        ft = mat[(urows,ucols)]

        m2 = jnp.transpose(mat)
        tf = m2[(urows,ucols)]
        return jnp.stack([ft, tf], axis = -1)

    @staticmethod 
    @jit    
    def vec_node_to_edgle(sr):
        """_summary_

        Args:
            sr (2D array): Each column represent an characteristic or effect and  each row represent the value of i for the characteristic of the given column

        Returns:
            (2D array): return and edgelist of all dyads combination (excluding diagonal).
            First column represent the value fo individual i  in the first column of argument sr, the second column the value of j in the second column of argument sr
        """
        N = sr.shape[0]
        lrows, lcols   = jnp.tril_indices(N, k=-1)
        urows, ucols   = jnp.triu_indices(N, k=1)
        ft = sr[urows,0]
        tf = sr[ucols, 1]
        return jnp.stack([ft, tf], axis = -1)
    
    # Netowrk metrics ----------------------
    @staticmethod 
    @jit    
    def outstrength(x):
        return jax.numpy.sum(x, axis=1)
    
    @staticmethod 
    @jit
    def instrength(x):
        return jax.numpy.sum(x, axis=0)

    @staticmethod 
    @jit
    def strength(x):
        return Net.outstrength(x) +  Net.instrength(x)

    @staticmethod 
    @jit
    def outdegree(x):
        mask = x != 0
        return jax.numpy.sum(mask, axis=1)

    @staticmethod 
    @jit
    def indegree(x):
        mask = x != 0
        return jax.numpy.sum(mask, axis=0)
    
    @staticmethod 
    @jit
    def degree(x):
        return Net.indegree(x) + Net.outdegree(x)

    # Sender receiver  ----------------------
    @staticmethod 
    @jit
    def apply_row_dotproduct(A, v):
        """
        Perform matrix-vector multiplication for each row of the array v.

        Parameters:
        A (jax.numpy.ndarray): A 2x2 matrix.
        v (jax.numpy.ndarray): An array of shape (n, 2) where each row is a 2-vector.

        Returns:
        jax.numpy.ndarray: An array of shape (n, 2) where each row is the result of the matrix-vector multiplication.
        """
        # Define a function that performs the matrix-vector multiplication
        def dotvec(A, v):
            return jnp.dot(A, v)

        # Vectorize the function using jax.vmap
        vmap_dotvec = jax.vmap(lambda v: dotvec(A, v))

        # Apply the vectorized function to the array of vectors
        result = vmap_dotvec(v)

        return result
    
    @staticmethod 
    @jit
    def apply_row_matmul(v, A):
        """
        Perform matrix-vector multiplication for each row of the array v.

        Parameters:
        A (jax.numpy.ndarray): A 2x2 matrix.
        v (jax.numpy.ndarray): An array of shape (n, 2) where each row is a 2-vector.

        Returns:
        jax.numpy.ndarray: An array of shape (n, 2) where each row is the result of the matrix-vector multiplication.
        """
        # Define a function that performs the matrix-vector multiplication
        def matvec(A, v):
            return A * v

        # Vectorize the function using jax.vmap
        vmap_matvec = jax.vmap(lambda v: matvec(A, v))

        # Apply the vectorized function to the array of vectors
        result = vmap_matvec(v)

        return result
    
    @staticmethod 
    @jit
    def prerpare_dyadic_effect(list_mat):
        if len(list_mat.shape) == 2:
            d = Net.mat_to_edgl(list_mat)
            d_s = d[:,0]
            d_r = d[:,1]

        elif len(list_mat.shape) == 3:
            dyadic_effects = vmap(Net.mat_to_edgl)(list_mat) 
            d_s = dyadic_effects[:,:,0].T
            d_r = dyadic_effects[:,:,1].T
            return d_s, d_r
        else:
            raise ValueError("Input must be a 2D or 3D array")   
        return d_s, d_r

    @staticmethod 
    @jit
    def prepare_outcome_effects(list_mat):
        return Net.prerpare_dyadic_effect(list_mat)

    def nodes_random_effects(N_id, sr_mu = 0, sr_sd = 1, sr_sigma = 1, cholesky_dim = 2, cholesky_density = 2 ):
        sr_raw =  dist.normal('sr_raw', sr_mu, sr_sd, sample_shape=[N_id, 2])
        sr_sigma =  dist.exponential('sr_sigma', sr_sigma, sample_shape= [2])
        sr_L = dist.lkjcholesky("sr_L", cholesky_dim, cholesky_density)
        rf = vmap(lambda x: random.factor_centered(sr_sigma, sr_L, x))(sr_raw)
        return rf, sr_raw, sr_sigma, sr_L # we return everything to get posterior distributions for each parameters

    def nodes_terms(self, N_var, focal_individual_predictors, target_individual_predictors,
                    s_mu = 0, s_sd = 1, r_mu = 0, r_sd = 1 ):
        """_summary_

        Args:
            focal_individual_predictors (2D jax array): each column represent node characteristics.
            target_individual_predictors (2D jax array): each column represent node characteristics.
            s_mu (int, optional): Default mean prior for focal_effect, defaults to 0.
            s_sd (int, optional): Default sd prior for focal_effect, defaults to 1.
            r_mu (int, optional): Default mean prior for target_effect, defaults to 0.
            r_sd (int, optional): Default sd prior for target_effect, defaults to 1.

        Returns:
            _type_: terms, focal_effects, target_effects
        """
        focal_effects = dist.normal('focal_effects', s_mu, s_sd, sample_shape=[N_var,])
        target_effects =  dist.normal('target_effects', r_mu, r_sd, sample_shape=[N_var,])
        terms = jnp.stack([Net.apply_row_dotproduct(focal_effects, focal_individual_predictors)[:,0],
                   Net.apply_row_dotproduct(target_effects, target_individual_predictors)[:,0]], axis = -1)
        return terms, focal_effects, target_effects # we return everything to get posterior distributions for each parameters

    def dyadic_random_effects(self, N_id, dr_mu = 0, dr_sd = 1, dr_sigma = 1, cholesky_dim = 2, cholesky_density = 2):
        dr_raw =  dist.normal('dr_raw', dr_mu, dr_sd, sample_shape=[N_id,2])
        dr_sigma = dist.exponential('dr_sigma', dr_sigma, sample_shape=[1])
        dr_L = dist.lkjcholesky("dr_L", cholesky_dim, cholesky_density)
        rf = vmap(lambda x: random.factor_centered(jnp.repeat(dr_sigma,2), dr_L, x))(dr_raw)
        return rf, dr_raw, dr_sigma, dr_L # we return everything to get posterior distributions for each parameters

    def dyadic_terms(self, d_s, d_r, d_m = 0, d_sd = 1):
        dyad_effects = dist.normal('dyad_effects', d_m, d_sd)
        terms1 = Net.apply_row_dotproduct(dyad_effects,  d_s)
        terms2 = Net.apply_row_dotproduct(dyad_effects,  d_r)
        rf = jnp.stack([terms1, terms2], axis = 1)
        return rf, dyad_effects
    
    def block_model_prior(self, N_grp, 
                      b_ij_mean = 0.01, b_ij_sd = 2.5, 
                      b_ii_mean = 0.1, b_ii_sd = 2.5,
                      name_b_ij = 'b_ij', name_b_ii = 'b_ii'):
        """Build block model prior matrix for within and between group links probabilities

        Args:
            N_grp (int): Number of blocks
            b_ij_mean (float, optional): mean prior for between groups. Defaults to 0.01.
            b_ij_sd (float, optional): sd prior for between groups. Defaults to 2.5.
            b_ii_mean (float, optional): mean prior for within groups. Defaults to 0.01.
            b_ii_sd (float, optional): sd prior for between groups. Defaults to 2.5.

        Returns:
            _type_: _description_
        """
        b_ij = dist.normal(name_b_ij, logit(b_ij_mean/jnp.sqrt(N_grp*0.5 + N_grp*0.5)), b_ij_sd, sample_shape=(N_grp, N_grp)) # transfers more likely within groups
        b_ii = dist.normal(name_b_ii, logit(b_ii_mean/jnp.sqrt(N_grp)), b_ii_sd, sample_shape=(N_grp, )) # transfers less likely between groups
        b = b_ij
        b = b.at[jnp.diag_indices_from(b)].set(b_ii)
        return b, b_ij, b_ii

    def block_prior_to_edglelist(v, b):
        """Convert block vector id group belonging to edgelist of i->j group values

        Args:
            v (1D array):  Vector of id group belonging
            b (2D array): Matrix of block model prior matrix

        Returns:
            _type_: 1D array representing the probability of links from i-> j 
        """

        v = Net.vec_node_to_edgle(jnp.stack([v, v], axis= 1), axis = 1)
        return b[(v[:,0], v[:,1])]
    
