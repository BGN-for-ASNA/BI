import pandas as pd
import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap


class NBDA:

    def __init__(self,network,status):
        """
        Initialize an NBDA object with network and status arrays.

        Args:
            network (Array): A 2-dimensional array of shape (n, n) or a 4-dimensional array of shape (n, n, t, num_networks). 
                             If 2D, it is repeated across time. If 4D, it should be created by the user.
            status (Array): A 2-dimensional array of shape (n, t) representing the status of nodes over time.

        Returns:
            None
        """
        # Status 
        self.status=status
        self.n=status.shape[0]
        self.t=status.shape[1]

        ## Status at t-1
        self.status_i, self.status_j = self.convert_status(status)

        # Network
        self.network=None
        if len(network.shape)==2:
            self.convert_network(network)
        elif (len(network.shape)==4):
            self.network=network
        elif (len(network.shape)!=4):
            raise ValueError("Network must be a 2 (n, n) or 4 dimensional array (n, n, t, num_networks)")

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
        self.D_social=None
        self.D_asocial=None

    def convert_network(self, network): # To be used only if there is a single network

        """
        Convert a single network array to a 4D array.

        Args:
            network (Array): A 2-dimensional array of shape (n, n).

        Returns:
            network: A 4-dimensional array of shape (n, n, t, 1) after repeating the network across time.
        """

        self.network=jnp.repeat(network[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))
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

    def convert_covNF(self, df, n, t, num_variables):
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
            df = jnp.array(df)
        else:
            if len(df)>2:
                raise ValueError("covariates must be a data frame or a 2-dimensional array")

        return  self.covNF_dims(df, n, t, num_variables)

    def import_covNF(self, df):
        """
        Import fixed nodal covariates.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (covNF_i, covNF_j).
        """
        self.covNF_i, self.covNF_j = self.convert_covNF(df, self.n, self.t, df.shape[1])
        return self.covNF_i, self.covNF_j

    def convert_covNV(self, covV):
        """
        Convert time-varying nodal covariates into 4D arrays.

        Args:
            covV (Array): A 3-dimensional array of shape (num_variables, n, t).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (result_array_i, result_array_j).
        """
        arrays=jnp.array([covV[i,:, ].T[:, None, :]* jnp.ones((1, self.n, 1)) for i in range(covV.shape[0])])
        result_array_i = jnp.transpose(arrays, (2, 3, 1, 0))  # (n, n, t, num_variables)
        result_array_j = jnp.transpose(result_array_i, (1, 0, 2, 3))
        
        return result_array_i, result_array_j

    def import_covNV(self, covV): #covV need to be a 3 dimensional array of shape (num_var, n, t)i.e. A list of matrices of time-varying covariates
        self.covNV_i, self.covNV_j = self.convert_covNV(covV)
        return self.covNV_i, self.covNV_j

    def import_covDF(self, covDF):
        """
        Import fixed dyadic covariates.

        Args:
            covDF (Array): A 2-dimensional array of shape (n, n) or a 3-dimensional array of shape (n, n, num_dyads).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables) if input is 3D, otherwise shape (n, n, t, 1).
        """
        if len(covDF.shape)==2:
            self.covDF = jnp.repeat(covDF[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))
            return self.covDF
        else:
            self.covDF = jnp.array([covDF[i,:, :,None]*jnp.ones((self.n, self.n, self.t)) for i in range(covDF.shape[0])]).transpose((1,2,3,0))
            return self.covDF


    def import_covDV(self, covDV):
        """
        Import time-varying dyadic covariates.

        Args:
            covDV (Array): A 3-dimensional array of shape (n, n, t) or a 4-dimensional array of shape (n, n, t, num_dyads).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables).
        """
        if len(covDV.shape)==3:# A list of matrices of a single time-varying covariate
            self.covDV = covDV[:, :, :,jnp.newaxis]
            return self.covDV

        if len(covDV.shape)==4:# A ist of list of matrices of a single time-varying covariate
            self.covDF = jnp.array([covDV[i,:, :,None]*jnp.ones((self.n, self.n, self.t)) for i in range(covDF.shape[0])]).transpose((1,2,3,0))
            return self.covDV

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
            covNF_j = self.covNF_i,
            covNF_i = self.covNF_j,
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
        D_social = []
        D_asocial = []
        for k in objects.keys():
            if k not in ['status', 'status_i', 'status_j', 'network']:
                if k is not None:   
                    if k in ['intercept', 'covNF_i', 'covNV_i']: 
                        D_social.append(objects[k])
                        D_asocial.append(objects[k][0,:,:,:],)
                    else:
                        D_social.append(objects[k])

        self.D_social = jnp.concatenate(D_social, axis=-1)
        self.D_asocial = jnp.concatenate(D_asocial, axis=-1)

    
    

    @staticmethod
    def sum_cov_effect(n,t,stacked_betas, stacked_cov):
        """
        Calculate the sum of covariate effects.

        Args:
            n (int): Number of nodes.
            t (int): Number of time points.
            stacked_betas (Array): An array of coefficients.
            stacked_cov (Array): An array of covariates.

        Returns:
            Array: A 3-dimensional array of shape (n, n, t) representing the sum of covariate effects.
        """
        res=jnp.zeros((n,n,t))
        for a in range(len(stacked_cov)):
            res=res.at[:,:,:].set(
            res[:,:,:] +  
            jnp.sum(stacked_cov[a]*stacked_betas[a],axis=3))
    
        return res

    # We can add individual observation information in the same forme as  an input time varying cov
    # We can add multiple behaviors acquisition in the form of a (n,n,t,num_behaviors)
    # Random variable to add
    # Do we add inverse of status_i? As we need net filter by j status that are informed (net*status_j) and cov array filtered by i status that are zero (cov*(1-status_i))
