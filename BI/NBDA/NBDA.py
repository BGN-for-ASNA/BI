import pandas as pd
import jax
import jax.numpy as jnp
from jax import vmap
from BI.Utils.dists import UnifiedDist as dist
from BI.Utils.link import link
import numpyro
from functools import partial 
from IPython.display import Markdown

class NBDA:

    def __init__(self,network=None,status=None,names_network=None,names_status=None): 
        """
        Initialize an NBDA object with network and status arrays.

        Args:
            network (Array): A 2-dimensional array of shape (n, n) or a 4-dimensional array of shape (n, n, t, num_networks). 
                             If 2D, it is repeated across time. If 4D, it should be created by the user.
            status (Array): A 2-dimensional array of shape (n, t) representing the status of nodes over time.

        Returns:
            None
        """
        if network is None or status is None:
            pass
        else:
            # Names of the networks and status
            self.names = dict(
                intercept = [],
                status = [],
                covNF = [],
                covNV = [],
                covDF = [],
                covDV = [],
                network = [],
                observed = [],
            )
            self.nbdaModel = True

            # Covariates locations
            self.locations = dict(
                covNF_i = [],
                covNF_j = [],
                covNV_i = [],                
                covNV_j = [],
            )

            # Get j covariate
            self.covNF_get_j = []
            self.covNV_get_j = []

            # Status 
            self.status=status
            self.n=status.shape[0]
            self.t=status.shape[1]

            ## Status at t-1
            self.status_i, self.status_j = self.convert_status(status)
            #self.give_name(self.status,'status',names_status)
            if names_status is None:
                self.names['status'].append('status')
            else:
                self.names['status'].append(names_status)

            # Network
            self.network=None
            if len(network.shape)==2:
                self.convert_network(network)
            elif (len(network.shape)==4):
                self.network=network
            elif (len(network.shape)!=4):
                raise ValueError("Network must be a 2 (n, n) or 4 dimensional array (n, n, t,   num_networks)")
            #self.give_name(self.network,'network',names_network)
            if names_network is None:
                self.names['network'].append('network')
            else:
                self.names['network'].append(names_network)

            # Intercetps for the network
            self.intercept = jnp.ones(self.network.shape)

            # fixed nodal covariates
            self.covNF_i=None
            self.covNF_j=None

            # Time-varying nodal covariates
            self.covNV_i=None
            self.covNV_j=None

            # fixed dyadic covariates
            self.covDF=None

            # Time-varying dyadic covariates
            self.covDV=None

            # individual observed
            self.observed=None

            # Conatenated covariates
            self.T_social=None
            self.T_asocial=None

            self.objects = None

    @partial(jax.jit, static_argnums=(0,)) 
    def scale_data(self, x):
        return (x - x.mean()) / x.std()

    @staticmethod
    @jax.jit
    def scale_along_time(covNV):
        return vmap(NBDA.scale, in_axes = 1, out_axes=1)(covNV)

    def give_name(self, object, key, names):
        if names is  None:
            if object.shape[-1] == 1:
                if key in list(self.names.keys()):
                    self.names[key].append([f'{key}'])
                else:
                    self.names[key] = [f'{key}']
            else:
                self.names[key] = [f'{key}_{i}' for i in range(object.shape[-1])] # last dim is the number of objects
        else:
            if len(names) != object.shape[-1]:
                raise ValueError(f'The number of names ({len(names)}) does not match the number of objects ({object.shape[-1]})')
            else:
                self.names[key] = names

    def convert_network(self, network, names = None): # To be used only if there is a single network

        """
        Convert a single network array to a 4D array.

        Args:
            network (Array): A 2-dimensional array of shape (n, n).

        Returns:
            network: A 4-dimensional array of shape (n, n, t, 1) after repeating the network across time.
        """

        self.network=jnp.repeat(network[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))

        if names is not None:
            if len(names)!=self.network.shape[3]:
                raise ValueError("The length of names must be equal to the number of networks.")
            self.names['network']=[name for name in names]
        return self.network

    def convert_status(self, status):#A 2-dimension arrray of (n,t)
        """
        Convert status array to lagged status arrays for i and j.

        Args:
            status (Array): A 2-dimensional array of shape (n, t).

        Returns:
            arrays_status_i (Array): A 3-dimensional array of shape (n, n, t) for i.
            arrays_status_j (Array): A 3-dimensional array of shape (n, n, t) for j.
             ! Both status are the lagged status (t-1) !
        """
        tmp=jnp.concatenate([jnp.zeros((self.n,1)),status[:,:-1]], axis=1)
        tmp2=jnp.array([tmp[:,i][:, None, None]* jnp.ones((self.n, self.n, 1)) for i in range(self.t)])
        tmp3=jnp.array([status[:,i][:, None]* jnp.ones((self.n,  1)) for i in range(self.t)])

        self.arrays_status_i = jnp.transpose(tmp2,(1,2,0,3))
        self.arrays_status_j = jnp.transpose(tmp2,(2,1,0,3))
        self.status = jnp.transpose(tmp3,(1,0,2))
        return self.arrays_status_i, self.arrays_status_j

    def covNF_dims(self,df, n, t, num_variables):
        """
        Convert fixed nodal covariates into 4D arrays.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).
            n (int): Number of nodes.
            t (int): Number of time points.
            num_variables (int): Number of covariates.

        Returns:
            result_array_i (Array): A 4-dimensional array of shape (n, n, t, num_variables) for i.
            result_array_j (Array): A 4-dimensional array of shape (n, n, t, num_variables) for j.
        """
        # Create arrays using broadcasting
        arrays = jnp.array([df[:, var_idx].reshape(-1, 1) * jnp.ones((n, n, t)) for var_idx in range(num_variables)])

        # Transpose arrays to get the required shape
        result_array_j = jnp.transpose(arrays, (1, 2, 3, 0))  # (n, n, t, num_variables)
        result_array_i = jnp.transpose(arrays, (2, 1, 3, 0))  # (n, n, t, num_variables)

        return result_array_i, result_array_j

    def convert_covNF(self, df, n, t, num_variables, scale = True):
        """
        Convert fixed nodal covariates into 4D arrays.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).
            n (int): Number of nodes.
            t (int): Number of time points.
            num_variables (int): Number of covariates.

        Returns:
            tuple: A tuple of two 4-dimensional arrays (result_array_i, result_array_j).
        """
        if isinstance(df, pd.DataFrame):
            if scale:
                df = df.apply(lambda x: self.scale_data(x.values), axis=0)
                df = jnp.array(df)
            else:
                df = jnp.array(df)
        else:
            if len(df.shape)>2:
                raise ValueError("covariates must be a data frame or a 2-dimensional array")
            else:
                if scale:
                    df = vmap(lambda x: self.scale_data(x), in_axes=1, out_axes=1)(df)

        return  self.covNF_dims(df, n, t, num_variables)

    def import_covNF(self, df, names=None, where = 'both', get_cov_j = True, scale = True):
        """
        Import fixed nodal covariates.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (covNF_i, covNF_j).
        """
        covNF_i, covNF_j = self.convert_covNF(df, self.n, self.t, df.shape[1], scale = scale)

        # Update self.covNF_i
        if where == 'asocial':
            get_cov_j = False
        if self.covNF_i is None:
            self.covNF_i = covNF_i
        else:
            self.covNF_i = jnp.concatenate((self.covNF_i, covNF_i), axis = -1)

        # Build cov from demonstrator
        if get_cov_j:
            if self.covNF_j is None:
                self.covNF_j = covNF_j
            else:
                self.covNF_j = jnp.concatenate((self.covNF_j, covNF_j), axis = -1)

        if names is None:
            self.names['covNF'].extend(str(len(self.names['covNF'])))
        else:
            self.names['covNF'].extend(names)

        self.covNF_get_j.extend([get_cov_j]*self.covNF_j.shape[3])
        self.locations['covNF_i'] = where
        self.locations['covNF_j'] = where

    def convert_covNV(self, covV):
        """
        Convert time-varying nodal covariates into 4D arrays.

        Args:
            covV (Array): A 3-dimensional array of shape (num_variables, n, t).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (result_array_i, result_array_j).
        """
        arrays=jnp.array([covV[:,:, i].T[:, None, :]* jnp.ones((1, self.n, 1)) for i in range(covV.shape[2])])
        result_array_j = jnp.transpose(arrays, (2, 3, 1, 0))  # (n, n, t, num_variables)
        result_array_i = jnp.transpose(arrays, (3, 2, 1, 0))
        
        return result_array_i, result_array_j

    def import_covNV(self, covV, names = None, scale = True, where = 'both', get_cov_j = True): #covV need to be a 3 dimensional array of shape (num_var, n, t)i.e. A list of matrices of time-varying covariates
        if scale:
            covV = NBDA.scale_along_time(covV)
        else: 
            print("Not scaling covariates along time can result in correlation between regressions coeffcients in of temporal covariates and the intercepts (i.e. social rate)")
            print("\nAlternative solution would be to add time varying coefficients, but it does affet computational time drastically.")

        covNV_i, covNV_j = self.convert_covNV(covV)

        if where == 'asocial':
            get_cov_j = False
        if self.covNV_i is None:
            self.covNV_i = covNV_i
        else:
            self.covNV_i = jnp.concatenate((self.covNV_i, covNV_i), axis = -1)

        # Build cov from demonstrator
        if get_cov_j:
            if self.covNV_j is None:
                self.covNV_j = covNV_j
            else:
                self.covNV_j = jnp.concatenate((self.covNV_j, covNV_j), axis = -1)

        #self.give_name(self.covNV_i,'covNV',names)
        if names is None:
            self.names['covNV'].extend(str(len(self.names['covNV'])))
        else:
            self.names['covNV'].extend(names)

        self.covNF_get_j.extend([get_cov_j]*self.covNV_j.shape[3])
        self.locations['covNV_i'] = where
        self.locations['covNV_j'] = where

    def import_covDF(self, covDF, names = None,  scale = True):
        """
        Import fixed dyadic covariates.

        Args:
            covDF (Array): A 2-dimensional array of shape (n, n) or a 3-dimensional array of shape (n, n, num_dyads).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables) if input is 3D, otherwise shape (n, n, t, 1).
        """
        if len(covDF.shape)==2:
            tmp = self.scale_data(covDF)
            covDF = jnp.repeat(tmp[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))

        else:
            res = []
            for i in range(covDF.shape[2]):
                tmp = self.scale_data(covDF[:,:,i])
                res.append(jnp.repeat(tmp[jnp.newaxis,:,:], self.t, axis=0).transpose((1,2,0)))
            covDF = jnp.stack(res, axis = -1)

        if self.covDF is None:
            self.covDF = covDF
        else:
            self.covDF = jnp.concatenate((self.covDF, covDF), axis = -1)


        if names is None:
            self.names['covDF'].extend(str(len(self.names['covDF'])))
        else:
            self.names['covDF'].extend(names)

    def import_covDV(self, covDV, names = None, scale = True, where = 'both'):
        """
        Import time-varying dyadic covariates.

        Args:
            covDV (Array): A 3-dimensional array of shape (n, n, t) or a 4-dimensional array of shape (n, n, t, num_dyadics_effects).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables).
        """
        
        if len(covDV.shape)==3:# A list of matrices of a single time-varying covariate
            covDV =  NBDA.scale_along_time(covDV)
            covDV = covDV[:, :, :,jnp.newaxis]

        elif len(covDV.shape)==4:# A ist of list of matrices of a single time-varying covariate
            covDV = jnp.array([covDV[i,:, :,None]*jnp.ones((self.n, self.n, self.t)) for i in range(covDV.shape[0])]).transpose((1,2,3,0))

        if self.covDV is None:
            self.covDV = covDV
        else:
            self.covDV = jnp.concatenate((self.covDV, covDV), axis = -1)

        if names is None:
            self.names['covDV'].extend(str(len(self.names['covDV'])))
        else:
            self.names['covDV'].extend(names)

    def stack_cov(self):
        """
        Stack all covariates into a dictionary.

        Returns:
            dict: A dictionary containing all covariates.
        """
        tmp=dict(
            intercept = self.intercept,
            status = self.status,
            status_i = self.status_i, 
            status_j = self.status_j,
            covNF_i = self.covNF_i,
            covNF_j = self.covNF_j,
            covNV_i = self.covNV_i,
            covNV_j = self.covNV_j,
            covDF = self.covDF,
            covDV = self.covDV,
            network = self.network,
            observed = self.observed,
        )
        return {k: v for k, v in tmp.items() if v is not None}

    def get_cov(self):
        """
        Get all covariates in the NBDA object.

        Returns:
            tuple: A tuple of two arrays (D_social, D_asocial).
        """
        
        objects = self.stack_cov()
        self.objects = objects
        D_social = [objects['intercept']]
        D_asocial = [objects['intercept'][0,:,:,:]]
        D_social_names = ['social_learning']
        D_asocial_names = ['asocial_learning']

        for k in objects.keys():
            if k not in ['intercept', 'status', 'status_i', 'status_j', 'network']:
                if k is not None:
                    if k in ['covNF_i', 'covNV_i']:
                        for i in range(objects[k].shape[3]):
                            if self.locations[k] == 'both':
                                D_social.append(jnp.expand_dims(objects[k][:,:,:,i],-1))
                                D_asocial.append(jnp.expand_dims(objects[k][:,0,:,i],-1))
                                if 'covNF' in k:
                                    D_social_names.append(k + '_' + self.names['covNF'][i])
                                    D_asocial_names.append(k + '_' + self.names['covNF'][i])
                                else:
                                    D_social_names.append(k + '_' + self.names['covNV'][i])
                                    D_asocial_names.append(k + '_' + self.names['covNV'][i])                                    

                            elif self.locations[k] == 'social':
                                D_social.append(jnp.expand_dims(objects[k][:,:,:,i],-1))
                                if 'covNF' in k:
                                    D_social_names.append(k + '_' + self.names['covNF'][i])
                                else:
                                    D_social_names.append(self.names['covNV'][i])    

                            elif self.locations[k] == 'asocial':
                                D_asocial.append(jnp.expand_dims(objects[k][:,0,:,i],-1))
                                if 'covNF' in k:
                                    D_asocial_names.append(k + '_' + self.names['covNF'][i])
                                else:
                                    D_social_names.append(k + '_' + self.names['covNV'][i])                                
                    else:
                            D_social.append(objects[k])   
                            if 'covNF' in k:
                                filtered = [s for b, s in zip(self.covNF_get_j, self.names['covNF']) if b]
                                for i in range(len(filtered)):
                                    D_social_names.append(k + '_' + filtered[i])
                            elif 'covNV' in k:
                                filtered = [s for b, s in zip(self.covNV_get_j, self.names['covNV']) if b]
                                for i in range(len(filtered)):
                                    D_social_names.append(k + '_' + filtered[i])      
                            elif 'covDF' in k:
                                for i  in range(objects[k].shape[3]):
                                        D_social_names.append(k + '_' + self.names['covDF'][i])  
                            elif 'covDV' in k:
                                for i  in range(objects[k].shape[3]):
                                        D_social_names.append(k + '_' + self.names['covDV'][i])  

        self.T_social = jnp.concatenate(D_social, axis=-1)
        self.T_asocial = jnp.concatenate(D_asocial, axis=-1)
        self.T_social_names = D_social_names
        self.T_asocial_names = D_asocial_names
        #return dict(D_social=self.D_social, D_asocial=self.D_asocial, status=self.status, network=self.network)

    def model(self,social=None, asocial=None, D_asocial=None, D_social=None, status=None, network=None):
        N = status.shape[0]
        T = status.shape[1]
        lk = jnp.zeros((N,T))

        if social is None:
            # Priors for social effect covariates
            alpha_soc = dist.normal(0, 4, shape = (1,), sample=False,    name='alpha_soc')
            betas_soc = dist.normal(0, 1, shape = (D_social.shape[3]-1,),    sample=False, name='betas_soc')
            social = jnp.concatenate((alpha_soc, betas_soc))

        if asocial is None:
            # Priors for asocial effect covariates
            alpha_asoc = dist.normal(0, 4,  shape = (1,), sample=False,  name='alpha_asoc')
            betas_asoc = dist.normal(0, 1, shape = (D_asocial.shape[2]-1,),  sample=False, name='betas_asoc')
            asocial = jnp.concatenate((alpha_asoc, betas_asoc))

        # Asocial learning -----------------------
        R_asocial = jnp.tensordot(D_asocial[:,0,:], asocial, axes=(-1, 0))    
        theta = link.inv_logit(R_asocial)
        lk = lk.at[:,0].set(theta)      
        for t in range(1,T):
            ## Social learning-----------------------
            R_social = jnp.tensordot(D_social[:,:,t,:], social, axes=(-1, 0))
            phi = link.inv_logit(R_social)
            attention_weigthed_network = phi*network[:,:,t,0]
            social_influence_weight = link.inv_logit_scale(jnp.tensordot(attention_weigthed_network[:,:], status[:,t-1], axes=(-1, 0)))       
            ## Asocial learning -----------------------
            R_asocial = jnp.tensordot(D_asocial[:,t,:], asocial, axes=(-1, 0))
            theta = link.inv_logit(R_asocial)

            # Informed update at t!= 0-----------------------
            lk = lk.at[:,t].set(jnp.where(status[:, t-1][:,0] == 1, jnp.nan, theta + (1-theta)*social_influence_weight[:,0]))       


        mask = ~jnp.isnan(lk)
        with numpyro.handlers.mask(mask=mask): 
        #m.binomial(probs=lk, obs=status[:,:,0])
            numpyro.sample("y", numpyro.distributions.Binomial(probs=lk), obs=status[:,:,0])

    def compute_probs(D_asocial,asocial,
                      D_social,social,
                      network,status,T):
        # Asocial learning -----------------------
        R_asocial = jnp.tensordot(D_asocial[:,0,:], asocial, axes=(-1, 0))    
        theta = link.inv_logit(R_asocial)
        lk = lk.at[:,0].set(theta)      
        for t in range(1,T):
            ## Social learning-----------------------
            R_social = jnp.tensordot(D_social[:,:,t,:], social, axes=(-1, 0))
            phi = link.inv_logit(R_social)
            attention_weigthed_network = phi*network[:,:,t,0]
            social_influence_weight = link.inv_logit_scale(jnp.tensordot(attention_weigthed_network[:,:], status[:,t-1], axes=(-1, 0)))       
            ## Asocial learning -----------------------
            R_asocial = jnp.tensordot(D_asocial[:,t,:], asocial, axes=(-1, 0))
            theta = link.inv_logit(R_asocial)

            # Informed update at t!= 0-----------------------
            lk = lk.at[:,t].set(jnp.where(status[:, t-1][:,0] == 1, jnp.nan, theta + (1-theta)*social_influence_weight[:,0]))       

    def print_model(self):
        r="""$$
        \\text{Informed} = \\text{Binomial}(\\text{LK}) \\newline
        \\text{LK} = \\left[ \\theta + (1-\\theta)S \\right] (1- z_i) \\newline
        \\theta = \\alpha_a \\newline
        S = \\alpha_s \\left( \\sum_{j = 1}^{N} A_{ij} z_{j} \\right) \\newline 
        \\alpha_a\\sim Normal(0,4) \\newline
        \\alpha_s \\sim Normal(0,4) \\newline
        """

        asocialCov=True
        socialCov=True
        if len(self.names) > 2:
            fa="""""" 
            fs=""""""
            count = 0
            for k in self.names.keys():
                if k in ['covNF', 'covNV', 'covDF','covDV']:
                    tmp=self.names[k]
                    if k == 'covNF':
                        if self.covNF_location == 'both':
                            asocialCov = socialCov = True
                        if self.covNF_location == 'asocial':
                            asocialCov = True
                            socialCov = False
                        if self.covNF_location == 'social':
                            asocialCov = False
                            socialCov = True

                    if k == 'covNV':
                        if self.covNV_location == 'both':
                            asocialCov = socialCov = True
                        if self.covNV_location == 'asocial':
                            asocialCov = True
                            socialCov = False
                        if self.covNV_location == 'social':
                            asocialCov = False
                            socialCov = True

                    if asocialCov:     
                        if k in ['covNF', 'covNV'] :   # Dyadic variables can't be in asocial                
                            for i in range(len(tmp)):
                                if i < len(tmp):
                                    fa=fa+f"\\beta_{{a{{{count}}}}} {tmp[i]} + "
                                    count += 1

                    if socialCov:
                        for i in range(len(tmp)):
                            if i < len(tmp):
                                fs=fs+f"\\beta_{{s{{{count}}}}} {tmp[i]} + "
                                count += 1

                    count += 1

            r = r.replace(r"\alpha_s \left( \sum A_{ij} z_{j} \right)",
            """\\left(\\alpha_s + X\\right) \\sum A_{ij} z_{j}  \\newline  X = """ + fs.rstrip(' +') + """  \\newline""")

            r = r.replace(r"\alpha_a ", 
            """\\alpha_a + Z  \\newline  Z = """ + fa.rstrip(' +') + """ \\newline""")

            r = r+"""\\beta_{(s)} \\sim Normal(0,1) \\newline"""
        r=r+"""$$"""
        display(Markdown(r))

    # We can add individual observation information in the same forme as  an input time varying cov
    # We can add multiple behaviors acquisition in the form of a (n,n,t,num_behaviors)
    # Random variable to add
    # Do we add inverse of status_i? As we need net filter by j status that are informed (net*status_j) and cov array filtered by i status that are zero (cov*(1-status_i))
