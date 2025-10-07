import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit
from BI.Network.Net import Neteffect
from BI.Utils.np_dists import UnifiedDist as dist

def logit(x):
    return jnp.log(x / (1 - x))
class SRM:
    def __init__(self, network, print_info = False):
        
        if network.ndim!=2:
             print('Error: Network dimension must be 2')

        self.N_id = network.shape[0]        
        self.N_dyad = (self.N_id * (self.N_id - 1) )/ 2
        self.N_dyad = int(self.N_dyad)
        self.network = Neteffect.mat_to_edgl(network)
        self.print_info = print_info
        self.focal_predictors = None
        self.focal_predictors_names = None
        self.receiver_predictors = None
        self.receiver_predictors_names = None
        self.dyadic_predictors = None
        self.dyadic_predictors_names = None
        self.exposure = None
        self.block_predictors = None
        self.block_predictors_names = None
        self.blocks_base_rate = None     
        self.block_intercept = None
        self.dyadic_intercept = None
        
    def import_focal_predictors(self, focal_predictors, names = None):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
            print("Ensure that the focal predictors are in the same order as the network")
            print("Ensure that the focal predictors are have a (V,N) shape, where V is the number of focal predictors and N is the number of nodes")
        # if it is a data frame convert to jax array
        if isinstance(focal_predictors, pd.DataFrame):
            focal_predictors = jnp.array(focal_predictors) # transpose for the dot product

        # if jax array check if it is a 2D array
        if isinstance(focal_predictors, jnp.ndarray):
            if focal_predictors.ndim != 2:
                print('Error: Focal factors must be a 2D array')


        self.focal_predictors = focal_predictors.T
        self.focal_predictors_names = names
        if self.print_info:
            print(f'Focal factors {self.focal_predictors_names}, imported')

    def import_receiver_predictors(self, receiver_predictors, names = None):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
            print("Ensure that the focal predictors are in the same order as the network")
            print("Ensure that the focal predictors are have a (V,N) shape, where V is the number of focal predictors and N is the number of nodes")
        # if it is a data frame convert to jax array
        if isinstance(receiver_predictors, pd.DataFrame):
            receiver_predictors = jnp.array(receiver_predictors)
        # if jax array check if it is a 2D array
        if isinstance(receiver_predictors, jnp.ndarray):
            if receiver_predictors.ndim != 2:
                print('Error: Focal factors must be a 2D array')
 

        self.receiver_predictors = receiver_predictors.T
        self.receiver_predictors_names = names
        if self.print_info:
            print(f'Focal factors {self.receiver_predictors_names}, imported')
    
    def import_dyadic_predictors(self, dyadic_predictors, names = None, intercept_present = False):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
        if isinstance(dyadic_predictors, jnp.ndarray):
            if dyadic_predictors.ndim == 2: 
                if dyadic_predictors.shape[0] != self.network.shape[0] and  dyadic_predictors.shape[1] != self.network.shape[1]:
                    print('Error: Dyadic factors must be a 2D array with the same first two dimensions as the network')
                else:
                    self.dyadic_predictors = dyadic_predictors
                    self.dyadic_predictors_names = names
                    if self.print_info:
                        print(f'Dyadic factors {self.dyadic_predictors_names}, imported')

            elif dyadic_predictors.ndim == 3:
                tmp2 =  []
                for a in range(dyadic_predictors.shape[2]):
                    tmp2.append(Neteffect.mat_to_edgl(dyadic_predictors[:,:,a]))               

                tmp2 = jnp.transpose(jnp.stack(tmp2), (1,2,0))
                #dyadic_predictors = tmp2[,,0:dim(data$dyad_set)[3]]
                self.dyadic_predictors = tmp2
                self.dyadic_predictors_names = names
                if not intercept_present:
                    dyadic_intercept = jnp.ones((self.N_dyad, 2))
                    self.dyadic_predictors= jnp.concatenate([dyadic_intercept[..., None], self.dyadic_predictors], axis = -1)


                if self.print_info:
                    print(f'Dyadic factors {self.dyadic_predictors_names}, imported')

            else:
                print('Error: Dyadic factors must be a 2D or 3D array')
        else:
            print('Error: Dyadic factors must be a jax array')
    
    def import_exposure(self, exposure):
        print('--------------------------------------------------------------------------------')
        # if jax array check if it is a 2D array
        if isinstance(exposure, jnp.ndarray):
            if exposure.ndim != 1:
                print('Error: Exposure must be a 1D array')
            else:
                self.exposure = exposure
                if self.print_info:
                    print(f'Exposure {self.exposure}, imported')
        else:
            print('Error: Exposure must be a jax array')

    def import_block_predictors2(self, block_predictors, names = None, index1 = False, intercept_present = False):
        print('Be sure that the block predictors are integers starting at 0')
        if self.print_info:
            print('--------------------------------------------------------------------------------')
        # if it is a data frame convert to jax array
        if isinstance(block_predictors, pd.DataFrame):
            block_predictors = jnp.array(block_predictors)
        # if jax array check if it is a 2D array
        if isinstance(block_predictors, jnp.ndarray):
            if block_predictors.ndim != 2:
                print('Error: Block(s) factors must be a 2D or 3D array')
                

        self.block_predictors = block_predictors
        if not intercept_present:
            block_intercept = jnp.ones(self.focal_predictors.shape[1])
            self.block_predictors = jnp.concat([block_intercept[..., None], block_predictors], axis = -1)
        if index1:
            self.block_predictors = block_predictors - 1
        self.block_predictors_names = names
        self.prepare_block_base_rate()
        self.prepare_block_to_edglelist()
        self.prepare_block_effects()
        if self.print_info:
            print(f'Focal factors {self.block_predictors_names}, imported')

    def prepare_block_base_rate(self):
        blocks_base_rate = []
        blocks_mu_ij = []

        for a in range(self.block_predictors.shape[1]):
            if self.block_predictors[:,a].sum() == 0:
                N_groups = 1
                N_by_group = jnp.array([self.N_id])
            else:   
                N_by_group = jnp.unique(self.block_predictors[:,a], return_counts = True)[1]
                N_groups = N_by_group.shape[0]
            base_rate = jnp.tile(0.01, (N_groups,N_groups))
            blocks_base_rate.append(base_rate.at[jnp.diag_indices_from(base_rate)].set(0.1))
            blocks_mu_ij.append(base_rate/jnp.sqrt(jnp.outer(N_by_group, N_by_group)))
        
        self.blocks_base_rate = blocks_base_rate
        self.blocks_mu_ij = blocks_mu_ij
    
    def prepare_block_to_edglelist(self):
        blocks = self.block_predictors
        blocks_edgl = []
        for a in range(blocks.shape[1]):
            blocks_edgl.append(Neteffect.vec_node_to_edgle(jnp.stack([blocks[:,a], blocks[:,a]], axis= 1).astype(int)))
        blocks_edgl = jnp.stack(blocks_edgl, axis = 1).transpose(0,2,1)
        self.blocks_edgl = blocks_edgl
    
    def prepare_block_effects(self):
        blocks_mu_ij = self.blocks_mu_ij        
        # --- Preparation for JIT ---
        max_dim = max(b.shape[0] for b in blocks_mu_ij)
        #    We find the largest dimension and pad the smaller arrays.
        blocks_mu_ij_padded =  [jnp.pad(b, [(0, max_dim - b.shape[0]), (0, max_dim - b.shape[1])], constant_values=jnp.nan) for b in blocks_mu_ij]
        blocks_mu_ij_padded = jnp.stack(blocks_mu_ij_padded)
        num_blocks = len(blocks_mu_ij)
        self.num_blocks = num_blocks
        self.blocks_mu_ij_padded = blocks_mu_ij_padded

    @staticmethod
    def update_block_effect(i, tmp, blocks_padded, block_predictors, b_ij_sd):
        """
        The body of the fori_loop, executed for each block.
        This function should not be JIT-compiled directly, as it's part of the loop.
        """
        # Select the current block and predictor vector
        current_block_mu = blocks_padded[i]
        current_predictor = block_predictors[:, i]

        # Calculate 'b' using the helper functions
        b = dist.normal(logit(current_block_mu), b_ij_sd, name = 'b' + str(i))

        update = Neteffect.block_prior_to_edglelist(current_predictor, b)
        tmp = tmp + jnp.sum(update, axis=1, keepdims=True) # Assumption: sum updates

        return tmp

    @staticmethod
    def block_prior_fn(initial_tmp, blocks_padded, block_predictors, num_blocks, b_ij_sd):
        """
        JIT-compatible function to run the update loop over all blocks.
        """
        # Create a lambda that captures the static arguments for fori_loop
        loop_body_fn = lambda i, tmp: SRM.update_block_effect(i, tmp, blocks_padded, block_predictors, b_ij_sd)

        # Run the loop
        final_tmp = jax.lax.fori_loop(0, num_blocks, loop_body_fn, initial_tmp)

        return final_tmp
      
    def block_prior(self):
        tmp = jnp.zeros_like(self.blocks_edgl[:,:,0], dtype=jnp.float64)
        Block_effect = SRM.block_prior_fn(tmp, self.blocks_mu_ij_padded, self.block_predictors, self.num_blocks, b_ij_sd = 0.1)
        return Block_effect
    
    def import_block_predictors(self, block_predictors, names = None, index1 = False):
        #! index1 for R users
        N_id = block_predictors.shape[0]
        N_vars = block_predictors.shape[1]
        N_groups_per_var = block_predictors.max(axis=0).astype(int)
        max_N_groups_per_var = max(N_groups_per_var)
        jnp.unique(block_predictors, return_counts=True)
        N_per_group = jnp.ones((N_vars, max_N_groups_per_var))
        for a in range(N_vars):
            N_per_group = N_per_group.at[a,:N_groups_per_var[a]].set(jnp.unique(block_predictors[:,a], return_counts = True)[1])
        N_per_group
        N_pars = (N_groups_per_var * N_groups_per_var).sum()
    
        B_V = B_I = B_J = B_In = B_Base = B_SS = jnp.zeros((N_vars, max_N_groups_per_var, max_N_groups_per_var))
        for v in range(N_vars):
            for i in range(N_groups_per_var[v]):
                for j in range(N_groups_per_var[v]):
                    B_V= B_V.at[v,i,j].set(v)
                    B_I = B_I.at[v,i,j].set(i)
                    B_J= B_J.at[v,i,j].set(j)
                    B_In = B_In.at[v,i,j].set(1)
                    B_Base =  B_Base.at[v,i,j].set(0.1 if i == j else 0.01)        
                    B_SS = B_SS.at[v,i,j].set(jnp.sqrt(N_per_group[v,i]*0.5 + N_per_group[v,j]*0.5))
    
        B_SS = B_SS.flatten()
        B_V = B_V.flatten()
        B_I = B_I.flatten()
        B_J = B_J.flatten()
        B_In = B_In.flatten()
        B_Base = B_Base.flatten()
        dat_B = jnp.stack([B_V, B_I, B_J, B_In, B_Base, B_SS], axis = -1)
        dat_B = dat_B[jnp.where(B_In==1)]
        Sigma = jnp.full((N_pars,), 2.5)
        Mu = logit(dat_B[:,-2]/dat_B[:,-1])
    
        dat_B = jnp.concatenate([dat_B, Mu[:,None], Sigma[:,None]], axis = 1)


        def build_block_tensor(dat_B, block_predictors):
            dat_B2 = dat_B[:, :3].astype(int)
            dat_B2_col0 = dat_B2[:, 0]
            dat_B2_col1 = dat_B2[:, 1]
            dat_B2_col2 = dat_B2[:, 2]
            block_preds_indexed = block_predictors[:, dat_B2_col0]
            scrap_i_matrix = (block_preds_indexed == dat_B2_col1)
            scrap_j_matrix = (block_preds_indexed == dat_B2_col2)
            # 5. Transpose the matrices to get a shape of (N_pars, N_id)
            scrap_i_T = scrap_i_matrix.T
            scrap_j_T = scrap_j_matrix.T
            Y_optimized = (scrap_i_T[:, :, np.newaxis] * scrap_j_T[:, np.newaxis, :]).astype(int)
            return  jnp.array(Y_optimized.transpose(1,2,0))
        
        block_tensor = build_block_tensor(dat_B, block_predictors)

        Block_edgl =  []
        for a in range(block_tensor.shape[2]):
            Block_edgl.append(Neteffect.mat_to_edgl(block_tensor[:,:,a]))  
        Block_edgl = jnp.transpose(jnp.stack(Block_edgl), (1,2,0))

        self.Block_edgl = Block_edgl
        self.dat_B = dat_B
        self.block_names = names
        self.N_vars_Block = N_vars
        self.max_N_groups_per_var = max_N_groups_per_var
        self.N_pars = N_pars
        if index1:
            self.block_predictors = block_predictors - 1
        else:
            self.block_predictors = block_predictors

    @staticmethod
    def block_effect_srm_class(Block_edgl, dat_B, sample = False):
        coefs = dist.normal(dat_B[:,-2], dat_B[:,-1], name = 'block_effects',  sample = sample)
        return jnp.sum(coefs * Block_edgl, axis=2)
    

    def model(self, Block_edgl, focal_predictors, receiver_predictors, dyadic_predictors, network, dat_B):
        # Block ---------------------------------------
        B = self.block_effect_srm_class(Block_edgl, dat_B)

        ## SR shape =  N individuals---------------------------------------
        sr =  Neteffect.sender_receiver(focal_predictors,receiver_predictors)

        # Dyadic shape = N dyads--------------------------------------  
        dr = Neteffect.dyadic_effect(dyadic_predictors)

        dist.bernoulli(logits = B + sr + dr , obs=network)

