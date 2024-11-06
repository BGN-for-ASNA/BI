from network.util import array_manip
import jax 
from jax import jit
import jax.numpy as jnp
from numpyro import deterministic
import os
import sys
# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
directory_to_add = os.path.join(current_file_directory, '..', 'utils')
directory_to_add = os.path.normpath(directory_to_add)
if directory_to_add not in sys.path:
    sys.path.append(directory_to_add)

from utils.unified_dists import UnifiedDist as dist

class Neteffect(array_manip):
    def __init__(self) -> None:
        pass
   
    # Sender receiver  ----------------------
    @staticmethod 
    def nodes_random_effects(N_id, sr_mu = 0, sr_sd = 1, sr_sigma_rate = 1, cholesky_dim = 2, cholesky_density = 2, sample = False, diag = False ):
        sr_raw =  dist.normal(sr_mu, sr_sd, shape=(2, N_id), name = 'sr_raw', sample = sample)
        sr_sigma =  dist.exponential( sr_sigma_rate, shape= (2,), name = 'sr_sigma', sample = sample)
        sr_L = dist.lkjcholesky(cholesky_dim, cholesky_density, name = "sr_L", sample = sample)
        rf = deterministic('sr_rf',(((sr_L @ sr_raw).T * sr_sigma)))
        #rf = deterministic('sr_rf', jax.vmap(lambda x: factors.random_centered(sr_sigma, sr_L, x))(sr_raw.T))
        #ids = jnp.arange(0,N_id)
        #edgl_idx = Neteffect.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
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
        #edgl_idx = Neteffect.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
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
        edgl_idx = Neteffect.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))
        sender = sr_effects[edgl_idx[:,0],0] + sr_effects[edgl_idx[:,1],1]
        receiver = sr_effects[edgl_idx[:,1],0] + sr_effects[edgl_idx[:,0],1]
        return jnp.stack([sender, receiver], axis = 1)

    @staticmethod 
    def sender_receiver(focal_individual_predictors, target_individual_predictors,  s_mu = 0, s_sd = 1, r_mu = 0, r_sd = 1, #Fixed effect parameters
                        sr_mu = 0, sr_sd = 1, sr_sigma_rate = 1, cholesky_dim = 2, cholesky_density = 2, #Random effect parameters
                        sample = False, diag = False ):    
        N_var = focal_individual_predictors.shape[0]
        N_id = focal_individual_predictors.shape[1]            

        sr_ff, focal_effects, target_effects = Neteffect.nodes_terms(focal_individual_predictors, target_individual_predictors, N_var = N_var, s_mu = s_mu, s_sd = s_sd, r_mu = r_mu, r_sd = r_sd, sample = sample, diag = diag )
        sr_rf, sr_raw, sr_sigma, sr_L = Neteffect.nodes_random_effects(N_id, sr_mu = r_mu, sr_sd = sr_sd, sr_sigma_rate = sr_sigma_rate, cholesky_dim = cholesky_dim, cholesky_density = cholesky_density,  sample = sample, diag = diag ) # shape = N_id
        sr_to_dyads = Neteffect.node_effects_to_dyadic_format(sr_ff + sr_rf) # sr_ff and sr_rf are nodal values that need to be converted to dyadic values
        return sr_to_dyads

    # dyadic effects ------------------------------------------
    @staticmethod 
    @jit
    def prepare_dyadic_effect(dyadic_effect_mat):
        if dyadic_effect_mat.ndim == 2:
            return Neteffect.mat_to_edgl(dyadic_effect_mat)
        else:
            return  jax.vmap(Neteffect.mat_to_edgl)(jnp.stack(dyadic_effect_mat))

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
            dr_ff, dyad_effects = Neteffect.dyadic_terms(dyadic_predictors, d_m = d_m, d_sd = d_sd, sample = sample)
            dr_rf, dr_raw, dr_sigma, dr_L =  Neteffect.dyadic_random_effects(dr_ff.shape[0], dr_mu = dr_mu, dr_sd = dr_sd, dr_sigma = dr_sigma, 
            cholesky_dim = cholesky_dim, cholesky_density = cholesky_density, sample = sample)
            return dr_ff + dr_rf
        else:
            dr_rf, dr_raw, dr_sigma, dr_L =  Neteffect.dyadic_random_effects(shape, dr_mu = dr_mu, dr_sd = dr_sd, dr_sigma = dr_sigma, 
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
        b_ij = dist.normal(Neteffect.logit(b_ij_mean/jnp.sqrt(N_grp*0.5 + N_grp*0.5)), b_ij_sd, shape=(N_dyads, 2), name = name_b_ij, sample = sample) # transfers more likely within groups
        b_ii = dist.normal(Neteffect.logit(b_ii_mean/jnp.sqrt(N_grp)), b_ii_sd, shape=(N_grp, ), name = name_b_ii, sample = sample) # transfers less likely between groups
        b = Neteffect.edgl_to_mat(b_ij, N_grp)
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

        v = Neteffect.vec_node_to_edgle(jnp.stack([v, v], axis= 1)).astype(int)
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
        b, b_ij, b_ii = Neteffect.block_model_prior(N_grp, 
                         b_ij_mean = b_ij_mean, b_ij_sd = b_ij_sd, 
                         b_ii_mean = b_ii_mean, b_ii_sd = b_ii_sd,
                         name_b_ij = name_b_ij, name_b_ii = name_b_ii, sample = sample)
        edgl_block = Neteffect.block_prior_to_edglelist(grp, b)
        #edgl_block = jnp.sum(edgl_block, axis = 1)
        #edgl_block = jnp.stack([edgl_block, edgl_block], axis = 1)
        #return edgl_block, b, b_ij, b_ii
        return edgl_block

