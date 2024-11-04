import inspect
from numpyro import deterministic
from unified_dists import UnifiedDist as dist
from Metrics import metrics as met

from jax import vmap
#' Test
#region
#from Darray import *
from functools import partial
import jax as jax
import jax.numpy as jnp
from jax import jit

# vector related functions -----------------------------------
@partial(jit, static_argnums=(1, 2,))
def vec_to_mat_jax(arr, N, K):
    return jnp.tile(arr, (N, K))

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
    return jnp.stack([tf, ft], axis = -1)

class net(object):
    """docstring for ClassName."""
    def __init__(self, arg):
        super(ClassName, self).__init__()
    arg

    
class Net(met):
    def __init__(self) -> None:
        pass

    @staticmethod 
    @jit
    def logit(x):
        return jnp.log(x / (1 - x))

    # Matrix manipulations -------------------------------------
    @staticmethod 
    @partial(jit, static_argnums=(1, ))
    def vec_to_mat(vec, shape = ()):
        return jnp.tile(vec, shape)

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
    @partial(jit, static_argnums=(1, ))
    def edgl_to_mat(edgl, N_id):
        m = jnp.zeros((N_id,N_id))
        urows, ucols   = jnp.triu_indices(N_id, 1)
        m = m.at[(ucols, urows)].set(edgl[:,1])
        m = m.at[(urows, ucols)].set(edgl[:,0])
        return m
    
    @staticmethod 
    @jit
    def remove_diagonal(arr):
        n = arr.shape[0]
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Array must be square to remove the diagonal.")

        # Create a mask for non-diagonal elements
        mask = ~jnp.eye(n, dtype=bool)

        # Apply the mask to the array to get non-diagonal elements
        non_diag_elements = arr[mask]  # Reshape as needed, here to an example shape
    
        return non_diag_elements
    
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
        lrows, lcols = jnp.tril_indices(N, k=-1)
        urows, ucols = jnp.triu_indices(N, k=1)
        ft = sr[urows,0]
        tf = sr[ucols,1]
        return jnp.stack([ft, tf], axis = -1)
    
    
    # Sender receiver  ----------------------
    @staticmethod 
    def nodes_random_effects(N_id, sr_mu = 0, sr_sd = 1, sr_sigma_rate = 1, cholesky_dim = 2, cholesky_density = 2, sample = False, diag = False ):
        sr_raw =  dist.normal(sr_mu, sr_sd, shape=(2, N_id), name = 'sr_raw', sample = sample)
        sr_sigma =  dist.exponential( sr_sigma_rate, shape= (2,), name = 'sr_sigma', sample = sample)
        sr_L = dist.lkjcholesky(cholesky_dim, cholesky_density, name = "sr_L", sample = sample)
        rf = deterministic('sr_rf',(((sr_L @ sr_raw).T * sr_sigma)))
        #rf = deterministic('sr_rf', jax.vmap(lambda x: factors.random_centered(sr_sigma, sr_L, x))(sr_raw.T))
        #ids = jnp.arange(0,N_id)
        #edgl_idx = Net.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
        #sender_random = rf[edgl_idx[:,0],0] + rf[edgl_idx[:,1],1]
        #receiver_random = rf[edgl_idx[:,1],0] + rf[edgl_idx[:,0],1]
        #random_effects = jnp.stack([sender_random, receiver_random], axis = 1)
        if diag:
            print("sr_raw--------------------------------------------------------------------------------")
            print(sr_raw)
            print("sr_sigma--------------------------------------------------------------------------------")
            print(sr_sigma)
            print("sr_L--------------------------------------------------------------------------------")
            print(sr_L)
            print("rf--------------------------------------------------------------------------------")
            print(rf)
            #print("sr_rf--------------------------------------------------------------------------------")
            #print(random_effects)
        #return random_effects, sr_raw, sr_sigma, sr_L # we return everything to get posterior distributions for each parameters
        return rf, sr_raw, sr_sigma, sr_L
   
    def nodes_terms(focal_individual_predictors, target_individual_predictors,
                    N_var = 1, s_mu = 0, s_sd = 1, r_mu = 0, r_sd = 1, sample = False, diag = False  ):
        """_summary_

        Args:
            idx (2D, jax array): An edglist of ids.
            focal_individual_predictors (2D jax array): each column represent node characteristics.
            target_individual_predictors (2D jax array): each column represent node characteristics.
            s_mu (int, optional): Default mean prior for focal_effect, defaults to 0.
            s_sd (int, optional): Default sd prior for focal_effect, defaults to 1.
            r_mu (int, optional): Default mean prior for target_effect, defaults to 0.
            r_sd (int, optional): Default sd prior for target_effect, defaults to 1.

        Returns:
            _type_: terms, focal_effects, target_effects
        """
        focal_effects = dist.normal(s_mu, s_sd, shape=(N_var,), sample = sample, name = 'focal_effects')
        target_effects =  dist.normal( r_mu, r_sd, shape= (N_var,), sample = sample, name = 'target_effects')
        terms = jnp.stack([focal_effects @ focal_individual_predictors, target_effects @  target_individual_predictors], axis = -1)

        
        #ids = jnp.arange(0,focal_individual_predictors[0].shape[0])
        #edgl_idx = Net.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
        #sender_receiver_ij = terms[edgl_idx[:,0],0] + terms[edgl_idx[:,1],1] # Sender effect between i and j is the sum of sender effects of i and j 
        #receiver_sender_ji = terms[edgl_idx[:,1],0] + terms[edgl_idx[:,0],1]
        if diag:
            print("focal_effects--------------------------------------------------------------------------------")
            print(focal_effects)
            print("target_effects--------------------------------------------------------------------------------")
            print(target_effects)
            print("terms--------------------------------------------------------------------------------")
            print(terms)
            #print("sr_ff--------------------------------------------------------------------------------")
            #print( jnp.stack([sender_receiver_ij, receiver_sender_ji], axis = 1))
            return terms, focal_effects, target_effects
        #return jnp.stack([sender_receiver_ij, receiver_sender_ji], axis = 1), focal_effects, target_effects # we return everything to get posterior distributions for each parameters
        return terms, focal_effects, target_effects
    
    @staticmethod 
    @jit
    def node_effects_to_dyadic_format(sr_effects):
        ids = jnp.arange(0,sr_effects.shape[0])
        edgl_idx = Net.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
        sender = sr_effects[edgl_idx[:,0],0] + sr_effects[edgl_idx[:,1],1]
        receiver = sr_effects[edgl_idx[:,1],0] + sr_effects[edgl_idx[:,0],1]
        return jnp.stack([sender, receiver], axis = 1)

    @staticmethod 
    def sender_receiver(focal_individual_predictors, target_individual_predictors,  s_mu = 0, s_sd = 1, r_mu = 0, r_sd = 1, #Fixed effect parameters
                        sr_mu = 0, sr_sd = 1, sr_sigma_rate = 1, cholesky_dim = 2, cholesky_density = 2, #Random effect parameters
                        sample = False, diag = False ):    
        N_var = focal_individual_predictors.shape[0]
        N_id = focal_individual_predictors.shape[1]            

        sr_ff, focal_effects, target_effects = Net.nodes_terms(focal_individual_predictors, target_individual_predictors, N_var = N_var, s_mu = s_mu, s_sd = s_sd, r_mu = r_mu, r_sd = r_sd, sample = sample, diag = diag )
        sr_rf, sr_raw, sr_sigma, sr_L = Net.nodes_random_effects(N_id, sr_mu = r_mu, sr_sd = sr_sd, sr_sigma_rate = sr_sigma_rate, cholesky_dim = cholesky_dim, cholesky_density = cholesky_density,  sample = sample, diag = diag ) # shape = N_id
        sr_to_dyads = Net.node_effects_to_dyadic_format(sr_ff + sr_rf) # sr_ff and sr_rf are nodal values that need to be converted to dyadic values
        return sr_to_dyads

    # dyadic effects ------------------------------------------
    @staticmethod 
    @jit
    def prepare_dyadic_effect(dyadic_effect_mat):
        if dyadic_effect_mat.ndim == 2:
            return Net.mat_to_edgl(dyadic_effect_mat)
        else:
            return  jax.vmap(Net.mat_to_edgl)(jnp.stack(dyadic_effect_mat))

    @staticmethod 
    def dyadic_random_effects(N_dyads, dr_mu = 0, dr_sd = 1, dr_sigma = 1, cholesky_dim = 2, cholesky_density = 2, sample = False, diag = False):
        dr_raw =  dist.normal(dr_mu, dr_sd, shape=(2,N_dyads), name = 'dr_raw', sample = sample)
        dr_sigma = dist.exponential(dr_sigma, shape=(1,), name = 'dr_sigma', sample = sample )
        dr_L = dist.lkjcholesky(cholesky_dim, cholesky_density, name = 'dr_L', sample = sample)
        dr_rf = deterministic('dr_rf', (((dr_L @ dr_raw).T * jnp.repeat(dr_sigma, 2))))
        if diag :
            print("dr_raw--------------------------------------------------------------------------------")
            print(dr_raw)
            print("dr_sigma--------------------------------------------------------------------------------")
            print(dr_sigma)
            print("dr_L--------------------------------------------------------------------------------")
            print(dr_L)
            print("rf--------------------------------------------------------------------------------")
            print(dr_rf)
        return dr_rf, dr_raw, dr_sigma, dr_L # we return everything to get posterior distributions for each parameters

    @staticmethod 
    def dyadic_terms(dyadic_predictors, d_m = 0, d_sd = 1, sample = False, diag = False):
        dyad_effects = dist.normal(d_m, d_sd, name= 'dyad_effects', shape = (dyadic_predictors.ndim - 1,), sample = sample)
        
        if dyadic_predictors.ndim == 2:
            dr_ff = dyad_effects * dyadic_predictors
            if diag :
                print("dyad_effects--------------------------------------------------------------------------------")
                print(dyad_effects)
                print("rf--------------------------------------------------------------------------------")
                print(rf)
            return dr_ff, dyad_effects
        else:
            if diag :
                print("dyad_effects--------------------------------------------------------------------------------")
                print(dyad_effects)
                print("rf--------------------------------------------------------------------------------")
                print(rf)
            dr_ff = dyadic_predictors * dyad_effects[:,None, None]
            return jnp.sum(dr_ff, axis=0), dyad_effects

    @staticmethod 
    def dyadic_effect(dyadic_predictors = None, shape = None, d_m = 0, d_sd = 1, # Fixed effect arguments
                     dr_mu = 0, dr_sd = 1, dr_sigma = 1, cholesky_dim = 2, cholesky_density = 2,
                     sample = False):
        if dyadic_predictors is not None and shape is None:
            print('Error: Argument shape must be defined if argument dyadic_predictors is not define')
            return 'Argument shape must be defined if argument dyadic_predictors is not define'
        if dyadic_predictors is not None :
            dr_ff, dyad_effects = Net.dyadic_terms(dyadic_predictors, d_m = d_m, d_sd = d_sd, sample = sample)
            dr_rf, dr_raw, dr_sigma, dr_L =  Net.dyadic_random_effects(dr_ff.shape[0], dr_mu = dr_mu, dr_sd = dr_sd, dr_sigma = dr_sigma, 
            cholesky_dim = cholesky_dim, cholesky_density = cholesky_density, sample = sample)
            return dr_ff + dr_rf
        else:
            dr_rf, dr_raw, dr_sigma, dr_L =  Net.dyadic_random_effects(shape, dr_mu = dr_mu, dr_sd = dr_sd, dr_sigma = dr_sigma, 
            cholesky_dim = cholesky_dim, cholesky_density = cholesky_density, sample = sample)
        return  dr_rf
  
    @staticmethod 
    def block_model_prior(N_grp, 
                          b_ij_mean = 0.01, b_ij_sd = 2.5, 
                          b_ii_mean = 0.1, b_ii_sd = 2.5,
                          name_b_ij = 'b_ij', name_b_ii = 'b_ii', sample = False):
        """Build block model prior matrix for within and between group links probabilities

        Args:
            N_grp (int): Number of groups to build
            b_ij_mean (float, optional): mean prior for between groups. Defaults to 0.01.
            b_ij_sd (float, optional): sd prior for between groups. Defaults to 2.5.
            b_ii_mean (float, optional): mean prior for within groups. Defaults to 0.01.
            b_ii_sd (float, optional): sd prior for between groups. Defaults to 2.5.

        Returns:
            _type_: _description_
        """
        N_dyads = int(((N_grp*(N_grp-1))/2))
        b_ij = dist.normal(Net.logit(b_ij_mean/jnp.sqrt(N_grp*0.5 + N_grp*0.5)), b_ij_sd, shape=(N_dyads, 2), name = name_b_ij, sample = sample) # transfers more likely within groups
        b_ii = dist.normal(Net.logit(b_ii_mean/jnp.sqrt(N_grp)), b_ii_sd, shape=(N_grp, ), name = name_b_ii, sample = sample) # transfers less likely between groups
        b = Net.edgl_to_mat(b_ij, N_grp)
        b = b.at[jnp.diag_indices_from(b)].set(b_ii)
        return b, b_ij, b_ii

    @staticmethod 
    @jit
    def block_prior_to_edglelist(v, b):
        """Convert block vector id group belonging to edgelist of i->j group values

        Args:
            v (1D array):  Vector of id group belonging
            b (2D array): Matrix of block model prior matrix (squared)

        Returns:
            _type_: 1D array representing the probability of links from i-> j 
        """

        v = Net.vec_node_to_edgle(jnp.stack([v, v], axis= 1)).astype(int)
        return jnp.stack([b[v[:,0],v[:,1]], b[v[:,1],v[:,0]]], axis = 1)

    @staticmethod 
    def block_model(grp, N_grp, b_ij_mean = 0.01, b_ij_sd = 2.5, b_ii_mean = 0.1, b_ii_sd = 2.5, sample = False):
        """Generate block model model matrix.

        Args:
            grp (_type_): _description_
            b_ij_mean (float, optional): _description_. Defaults to 0.01.
            b_ij_sd (float, optional): _description_. Defaults to 2.5.
            b_ii_mean (float, optional): _description_. Defaults to 0.1.
            b_ii_sd (float, optional): _description_. Defaults to 2.5.
            name_b_ij (str, optional): _description_. Defaults to 'b_ij'.
            name_b_ii (str, optional): _description_. Defaults to 'b_ii'.
            sample (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Get grp name from user. This seems to slower down the code operations, but from user perspective it is more convenient.....
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        name = string[string.find('(') + 1:-1].split(',')[0]
        name_b_ij = 'b_ij_' + str(name)
        name_b_ii = 'b_ii_' + str(name) 

        #N_grp = len(jnp.unique(grp))
        b, b_ij, b_ii = Net.block_model_prior(N_grp, 
                         b_ij_mean = b_ij_mean, b_ij_sd = b_ij_sd, 
                         b_ii_mean = b_ii_mean, b_ii_sd = b_ii_sd,
                         name_b_ij = name_b_ij, name_b_ii = name_b_ii, sample = sample)
        edgl_block = Net.block_prior_to_edglelist(grp, b)
        #edgl_block = jnp.sum(edgl_block, axis = 1)
        #edgl_block = jnp.stack([edgl_block, edgl_block], axis = 1)
        #return edgl_block, b, b_ij, b_ii
        return edgl_block

