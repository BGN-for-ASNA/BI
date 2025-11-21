import pandas as pd
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import arviz as az
import numpyro
from BI.Distributions.np_dists import UnifiedDist as dist
dist = dist()

class survival_old:
    """    The survival class is designed to handle survival analysis data and perform various operations related to time-to-event data. It provides methods for extracting basic information from the dataset, plotting censoring status, converting continuous time data into discrete intervals, calculating cumulative hazards and survival probabilities, and visualizing the results. This class serves as a high-level interface for managing survival data, allowing users to easily analyze and interpret time-to-event outcomes in a structured manner.
    """
    def __init__(self, parent):
        self.parent = parent 
        self.n_patients = None
        self.patients = None
        self.time = None
        self.event = None
        self.cov = None
        self.interval_length = None
        self.interval_bounds = None
        self.n_intervals = None
        self.death = None
        self.exposure = None
        self.base_hazard = None
        self.met_hazard = None  
        self.data_on_model = {}
        self.cov = None
        self.df = None
        self.dist = dist()
    
    @property
    def df(self):
        return self.parent.df  # always get from bi

    @df.setter
    def df(self, value):
        self.parent.df = value  # always set in bi

    def get_basic_info(self, event='event', time='time', cov=None):
        ''' Get basic information about the dataset

            Parameters
            ----------
                event : str, optional
                    Name of the column containing the event status, by default 'event'
                time : str, optional
                    Name of the column containing the time, by default 'time'
                cov : str, optional
                    Name of the column containing the covariate, by default None

            Returns
            -------
                None

            Notes
            -----
                The function returns the following attributes:
                    - n_patients : int
                        Number of patients in the dataset
                    - patients : np.array
                        Array of patient indices
                    - time : np.array
                        Array of time points
                    - event : np.array
                        Array of event status
                    - data_on_model : dict
                        Dictionary containing the data of the model present ib the dataset                        
                    - cov : str
                        Name of the covariate
                    - df : pd.DataFrame
                        DataFrame containing the dataset

        ''' 

        # Number of patients in the dataset
        self.n_patients = self.df.shape[0]
        self.patients = np.arange(self.n_patients)  # Array of patient indices
        self.time = self.df.loc[:, time].values
        self.event = self.df.loc[:, event].values

        if self.data_on_model is None:
            self.data_on_model = {}
            
        if type(cov) is str:
            self.cov = cov # covariate
            tmp = jnp.reshape(self.df[cov].values, (1, len(self.df[cov].values)))
            self.data_on_model[cov] = tmp

        elif type(cov) is list:
            self.cov = cov
            a = 0
            for item in cov:
                if a == 0:
                    self.data_on_model['cov'] = jnp.array(self.df[item].values)
                    a += 1
                else:
                    self.data_on_model['cov'] = jnp.stack([self.data_on_model['cov'] , jnp.array(self.df[item].values)])
       
    def plot_censoring(self, event='event', time='time', cov='metastasized', xlabel='Time', ylabel='Subject'):
        """
        Plots the censoring status of subjects in a time-to-event dataset.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing the time-to-event data.
        event : str, optional
            The name of the column in `df` indicating the event status (1 = event occurred, 0 = censored).
            Default is 'event'.
        time : str, optional
            The name of the column in `df` representing the time variable. Default is 'time'.
        cov : str, optional
            The name of the column in `df` representing a covariate, such as metastasized status.
            Default is 'metastasized'.
        xlabel : str, optional
            Label for the x-axis. Default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Subject'.

        Returns:
        --------
        None
            This function generates a plot showing censored and uncensored subjects along with a specified covariate.


        """
        self.get_basic_info(event, time, cov)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot censored subjects (event = 0) as red horizontal lines
        ax.hlines(
            self.patients[ self.event == 0], 0,  self.df[ self.event == 0].loc[:, 'time'], color="C3", label="Censored"
        )

        # Plot uncensored subjects (event = 1) as gray horizontal lines
        ax.hlines(
             self.patients[ self.event == 1], 0,  self.df[ self.event == 1].loc[:, 'time'], color="C7", label="Uncensored"
        )

        # Add scatter points for subjects with the specified covariate (e.g., metastasized = 1)
        ax.scatter(
            self.df[self.df.loc[:,cov] == 1].loc[:, time],
            self.patients[self.df.loc[:,cov] == 1],
            color="k",
            zorder=10,
            label=cov,
        )

        # Set plot limits and labels
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylabel(ylabel)

        # Set y-axis limits to provide padding around subjects
        ax.set_ylim(-0.25, self.n_patients + 0.25)

        # Add legend to the plot
        ax.legend(loc="center right")

    def surv_object(self, time='time', event='event', cov=None, interval_length=3):
        """
        Converts continuous time and event data into discrete time intervals for survival analysis.

        Parameters:
        -----------
        time : str, optional
            The name of the column in `df` representing the continuous time variable (default is 'time').
        event : str, optional
            The name of the column in `df` representing the event indicator (default is 'event').
        interval_length : int, optional
            The length of each discrete time interval (default is 3).

        Returns:
        --------
        interval_bounds : numpy.ndarray
            Array of boundaries for discrete time intervals.
        n_intervals : int
            The total number of discrete intervals.
        intervals : numpy.ndarray
            Array of interval indices.
        death : numpy.ndarray
            A binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event
            in each interval (1 if the event occurred, 0 otherwise).
        exposure : numpy.ndarray
            A matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        Notes:
        ------
        - The function assumes that subjects who experienced the event did so at the end of the time period.
        - Exposure is capped by the interval bounds, and the last interval reflects the remaining time to the event or censoring.

        """
        print("Survival analysis is still in development. Use it with caution and report any issues.")
        self.get_basic_info(time = time, event = event, cov = cov)
        self.interval_length = interval_length
        
        # Define interval bounds and calculate the number of intervals
        interval_bounds = np.arange(0, self.time.max() + interval_length + 1, interval_length)
        n_intervals = interval_bounds.size - 1
        intervals = np.arange(n_intervals)
        self.n_intervals = n_intervals

        # Determine the last interval each patient belongs to
        last_period = np.floor((self.time - 0.01) / self.interval_length).astype(int)
        self.last_period = last_period

        # Create a binary death matrix (n_patients x n_intervals)
        death = np.zeros((self.n_patients, self.n_intervals))
        death[self.patients, last_period] = self.event

        # Calculate exposure times for each interval
        exposure = np.greater_equal.outer(self.time, interval_bounds[:-1]) * interval_length
        exposure[self.patients, last_period] = self.time - interval_bounds[last_period]

        self.interval_bounds = interval_bounds # Array of boundaries for discrete time intervals.
        self.intervals = intervals # Array of interval indices.
        self.death = death # Binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event in each interval (1 if the event occurred, 0 otherwise).
        self.exposure = exposure # Matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        # data for the model
        if self.data_on_model is None:
            self.data_on_model = {}
        self.data_on_model['intervals'] = jnp.array(intervals)
        self.data_on_model['death'] = jnp.array(death)
        self.data_on_model['exposure']= jnp.array(exposure)

        if type(cov) is str:
            tmp = jnp.reshape(self.df[cov].values, (1, len(self.df[cov].values)))
            self.data_on_model[cov] = tmp
        elif type(cov) is list:
            for item in cov:
                self.data_on_model[item] = jnp.array(self.df[item].values)
        self.parent.data_on_model = self.data_on_model  # update the parent data_on_model

    def cum_hazard(self, hazard):
        """
        Calculates the cumulative hazard from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The cumulative hazard calculated by summing the hazard over time steps.

        Notes:
        ------
        - The cumulative hazard is computed as the cumulative sum of the hazard values, 
          scaled by the `interval_length` factor.

        """
        return (self.interval_length * hazard).cumsum(axis=-1)

    def survival(self, hazard):
        """
        Calculates the survival probability from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The survival probability, computed as the exponential of the negative cumulative hazard.

        Notes:
        ------
        - Survival is calculated as `exp(-cumulative hazard)` where the cumulative hazard 
          is calculated using the `cum_hazard` function.
        """
        return np.exp(-self.cum_hazard(hazard))

    def hazards(self, m, lambda0 = 'lambda0', beta = 'beta'):
        """
        Calculates two hazard values: the base hazard and the covariates hazard, based on posterior samples of 
        the parameters `lambda0` and `beta`.

        Parameters:
        -----------
        m : object
            An object that contains posterior samples in the `posteriors` attribute. This attribute should include 
            the parameters `lambda0` and `beta` for calculating the hazards.

        lambda0 : str, optional, default='lambda0'
            The key for the base hazard parameter in the `posteriors` dictionary.

        beta : str, optional, default='beta'
            The key for the covariate effect parameter in the `posteriors` dictionary.

        Returns:
        --------
        tuple of numpy.ndarray or jax.numpy.ndarray
            - base_hazard2 : The base hazard calculated from `lambda0` parameter.
            - met_hazard2 : The metastasis hazard calculated as the product of `lambda0` and `exp(beta)`.

        Notes:
        ------
        - The base hazard is derived from the posterior samples of `lambda0`, while the covariate hazard 
          is computed by multiplying `lambda0` with the exponential of `beta`.
        - `np.expand_dims` is used to ensure the correct dimensions when computing the product.

        """
        base_hazard = m.posteriors["lambda0"]
        array_expanded = jnp.expand_dims(np.exp(m.posteriors["beta"]), axis=-1)
        met_hazard =m.posteriors["lambda0"] * array_expanded

        self.base_hazard = base_hazard
        self.met_hazard = met_hazard
        return base_hazard, met_hazard 

    def plot_surv(self, lambda0 = 'lambda0', beta = 'beta',
                  xlab='Time', ylab='Survival', covlab = 'treated', title = "Bayesian survival model"):

        base_hazard = self.parent.posteriors[lambda0]        
        met_hazard =self.parent.posteriors[lambda0] * self.parent.posteriors[beta]

        fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))   

        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(base_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C0",
            fill_kwargs={"label": "Had not metastasized"},
        )
        
        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(met_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C1",
            fill_kwargs={"label": "Metastasized"},
        )   

        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(base_hazard), axis = 0), color="darkblue")
        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(met_hazard), axis = 0), color="maroon")   

        hazard_ax.set_xlim(0, self.time.max())
        hazard_ax.set_xlabel(xlab)
        hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")
        hazard_ax.legend(loc=2) 

        az.plot_hdi(self.interval_bounds[:-1], self.survival(base_hazard), ax=surv_ax, smooth=False, color="C0")
        az.plot_hdi(self.interval_bounds[:-1], self.survival(met_hazard), ax=surv_ax, smooth=False, color="C1")  

        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(base_hazard), axis = 0), color="darkblue")
        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(met_hazard), axis = 0), color="maroon")   

        surv_ax.set_xlim(0, self.time.max())
        surv_ax.set_xlabel(ylab)
        surv_ax.set_ylabel("Survival function $S(t)$")  

        fig.suptitle(title);

    def mu(self, cov,  exposure, lambda0, beta):
        lambda_ = numpyro.deterministic('lambda_', jnp.outer(jnp.exp(beta * cov), lambda0)) 
        mu = numpyro.deterministic('mu', exposure * lambda_)
        return lambda_, mu

    def hazard_rate(self, cov, beta, lambda0):
        lambda_ = numpyro.deterministic('lambda_', jnp.outer(jnp.exp(beta @ cov), lambda0)) 
        return lambda_
    
    def model(self,intervals, death, metastasized, exposure):
        # Parameters priors distributions-------------------------
        ## Base hazard distribution
        lambda0 = self.dist.gamma(0.01, 0.01, shape= intervals.shape, name = 'lambda0')
        ## Covariate effect distribution
        beta = self.dist.normal(0, 1000, shape = (1,),  name='beta')
        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ = self.hazard_rate(cov = metastasized, beta = beta, lambda0 = lambda0)
        ### Compute exposure rates
        mu = exposure * lambda_
    
        # Likelihood calculation
        self.dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)

import jax
import jax.numpy as jnp
import numpy as np

# estimation will change because alphabetic names of prior generating different seed to get same resutls as PyMC, gange names of priors to 'lambda0', 'beta'
class survival():
    def __init__(self,parent):
        self.n_patients = None
        self.patients = None
        self.time = None
        self.event = None
        self.cov = None
        self.interval_length = None
        self.interval_bounds = None
        self.n_intervals = None
        self.death = None
        self.exposure = None
        self.base_hazard = None
        self.met_hazard = None  
        self.data_on_model = {}
        self.cov_names = [] # Time-invariant covariates names
        self.cov = np.array([])  # Time-invariant covariates
        self.cov_v_names = []  # Time-varying covariates names
        self.cov_v = None  # Time-varying covariates 
        self.df = None
        self.oberved = None
        self.parent = parent
        self.parent.model_name = 'pca' 
        #self.surv_object(time, event, interval_length)


    def import_time_even(self, time, event, interval_length=1):
        """
        Convert data into of survival information into a surv object.
        Converts continuous time and event data into discrete time intervals for survival analysis if interval_length > 1.

        Parameters:
        -----------
        time : 1D jnp.ndarray of shape (N,) representing the continuous time variable.
        event : 1D Binary jnp.ndarray of shape (N,) representing the event indicator.
        interval_length : int, optional
            The length of each discrete time interval (default is 3).

        Returns:
        --------
        - n_patients : int
            Number of patients in the dataset
        - patients : np.array
            Array of patient indices
        - time : np.array
            Array of time points
        - event : np.array
            Array of event status
        - interval_bounds : jnp.ndarray
            Array of boundaries for discrete time intervals.
        - n_intervals : int
            The total number of discrete intervals.
        - intervals : jnp.ndarray
            Array of interval indices.
        - death : jnp.ndarray
            A binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event
            in each interval (1 if the event occurred, 0 otherwise).
        - exposure : jnp.ndarray
            A matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        Notes:
        ------
        - The function assumes that subjects who experienced the event did so at the end of the time period.
        - Exposure is capped by the interval bounds, and the last interval reflects the remaining time to the event or censoring.

        """
        self.time = time
        self.event = event
        self.n_patients = self.time.shape[0]
        self.patients = jnp.arange(self.n_patients)  # Array of patient indices
        self.interval_length = interval_length
        
        # Define interval bounds and calculate the number of intervals
        interval_bounds = np.arange(0, self.time.max() + interval_length + 1, interval_length)
        n_intervals = interval_bounds.size - 1
        intervals = np.arange(n_intervals)
        self.n_intervals = n_intervals

        # Determine the last interval each patient belongs to
        last_period = np.floor((self.time - 0.01) / self.interval_length).astype(int)
        self.last_period = last_period

        # Create a binary death matrix (n_patients x n_intervals)
        death = np.zeros((self.n_patients, self.n_intervals))
        death[self.patients, last_period] = self.event

        # Calculate exposure times for each interval
        exposure = np.greater_equal.outer(self.time, interval_bounds[:-1]) * interval_length
        exposure[self.patients, last_period] = self.time - interval_bounds[last_period]

        self.interval_bounds = interval_bounds # Array of boundaries for discrete time intervals.
        self.intervals = intervals # Array of interval indices.
        self.death = death # Binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event in each interval (1 if the event occurred, 0 otherwise).
        self.exposure = exposure # Matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.
        
        print("------------------------------------------------------------------------------")
        print(f'Survival concern {self.n_patients} individuals in {self.n_intervals} intervals.')
        print(f'{self.n_patients - self.death.sum()} individuals experienced the event.')

        self.observed = jnp.ones((self.n_patients, self.n_intervals))
        self.build_data()
        self.parent.data_on_model = self.data_on_model

    def import_covF(self, cov, names):
        """
        Import fixed covariates (patient-level).
        """
        cov = np.asarray(cov)  # make sure it's array
    
        # validate
        if cov.ndim == 1:
            if len(cov) != self.n_patients:
                raise ValueError(f'Length mismatch: covariate ({len(cov)}) vs patients ({self.n_patients})')
            if len(names) != 1:
                raise ValueError(f'Names list must have length 1 for 1D covariate')
        else:
            if cov.shape[0] != self.n_patients:
                raise ValueError(f'Length mismatch: covariate ({cov.shape[0]}) vs patients ({self.n_patients})')
            if cov.shape[1] != len(names):
                raise ValueError(f'Number of names ({len(names)}) does not match covariates ({cov.shape[1]})')
    
        # add names
        self.cov_names += names
    
        # stack into existing covariates
        if self.cov.size == 0:
            self.cov = cov.reshape(-1, 1) if cov.ndim == 1 else cov
        else:
            self.cov = np.column_stack([self.cov, cov])
        print("------------------------------------------------------------------------------")
        print(f'Covariates imported: {names}')
        print(f'Surv object now has {self.cov.shape[1]} covariates: {self.cov_names}')
        self.build_data()
        self.n_cov = len(self.cov_names)
        self.parent.data_on_model = self.data_on_model

    def import_covV(self, cov, names):
        """
        Import time-varying covariates.
    
        Parameters
        ----------
        cov : np.ndarray
            Array of shape (N, T) for one covariate, 
            or (N, T, V) for multiple covariates.
        names : list of str
            Names of the covariates.
        """
        cov = np.asarray(cov)
    
        # Handle single covariate case (N,T) → (N,T,1)
        if cov.ndim == 2:
            if cov.shape[0] != self.n_patients or cov.shape[1] != self.n_intervals:
                raise ValueError(
                    f"Expected shape ({self.n_patients}, {self.n_intervals}), got {cov.shape}"
                )
            cov = cov[..., np.newaxis]   # (N,T,1)
            if len(names) != 1:
                raise ValueError("Must provide one name for a single covariate.")
        
        # Handle multiple covariates (N,T,V)
        elif cov.ndim == 3:
            if cov.shape[0] != self.n_patients or cov.shape[1] != self.n_intervals:
                raise ValueError(
                    f"Expected shape ({self.n_patients}, {self.n_intervals}, V), got {cov.shape}"
                )
            if cov.shape[2] != len(names):
                raise ValueError(
                    f"Number of names ({len(names)}) must match V ({cov.shape[2]})"
                )
        else:
            raise ValueError("cov must be 2D (N,T) or 3D (N,T,V).")
    
        # Initialize or stack
        print(self.cov_v)
        if self.cov_v is None:
            self.cov_v = cov
        else:
            self.cov_v = np.concatenate([self.cov_v, cov], axis=-1)
    
        # Store names
        if not hasattr(self, "cov_v_names"):
            self.cov_v_names = []
        self.cov_v_names.extend(names)
        self.build_data()
        self.parent.data_on_model = self.data_on_model
        print("------------------------------------------------------------------------------")
        print(f"Imported covariates {names}.")
        print(f"Surv object now has {self.cov_v.shape[2]} time-varying covariates: {self.cov_v_names}")

    def build_surv_covariates(self):
        """
        Build a full covariate array for survival modeling.
    
        - Fixed covariates (N,) are expanded to (N,T)
        - Stacked with existing time-varying covariates (N, T, V)
        - Updates covariate names with fixed covariates first
        """
        if len(self.cov_v_names) == 0:
            self.cov_all_names = self.cov_names
            self.cov_all = self.cov
            return None
            
        # 1️ Expand fixed covariates to (N, T)
        cov_fixed_expanded = None
        if self.cov is not None and self.cov.size > 0:
            if self.cov.ndim == 1:
                # single covariate → shape (N,T)
                cov_fixed_expanded = np.repeat(self.cov[:, np.newaxis], self.n_intervals, axis=1)
            elif self.cov.ndim == 2:
                # multiple fixed covariates → shape (N,T,F)
                cov_fixed_expanded = np.repeat(self.cov[:, np.newaxis, :], self.n_intervals, axis=1)
                # axes to (N,T,F)
            else:
                raise ValueError(f"Unexpected fixed covariate shape: {self.cov.shape}")
    
        # 2️ Time-varying covariates
        cov_v = getattr(self, "cov_v", None)
    
        # 3️ Combine fixed and time-varying covariates
        if cov_fixed_expanded is not None and cov_v is not None:
            # If cov_fixed_expanded has shape (N,T) → add last axis for single variable
            if cov_fixed_expanded.ndim == 2:
                cov_fixed_expanded = cov_fixed_expanded[..., np.newaxis]  # (N,T,1)
            self.cov_all = np.concatenate([cov_fixed_expanded, cov_v], axis=-1)
        elif cov_fixed_expanded is not None:
            self.cov_all = cov_fixed_expanded
        elif cov_v is not None:
            self.cov_all = cov_v
        else:
            raise ValueError("No covariates available to build.")
    
        # 4️ Update covariate names: fixed first, then time-varying
        fixed_names = self.cov_names if self.cov is not None else []
        v_names = getattr(self, "cov_v_names", []) if cov_v is not None else []
        self.cov_all_names = fixed_names + v_names
    
        print("------------------------------------------------------------------------------")
        print(f"Built all covariates array with shape {self.cov_all.shape}")
        print(f"Covariate names order: {self.cov_all_names}")

        self.cov_all = jnp.array(self.cov_all)

    def observed(self, observed):
        """
        Add observed data to the survival object.
        Args:
            observed (np.ndarray): Array of observed data with shape (N, T). This array represent if individuals have been observed at each time point and thus should be 1 if observed and 0 if censored.
        Raises:
            ValueError: If observed data shape does not match the survival object. 
        """
        if observed.shape[0] != self.n_patients:
            raise ValueError(f"Expected observed data shape ({self.n_patients}), got {observed.shape}")
        if observed.shape[1] != self.n_intervals:
            raise ValueError(f"Expected observed data shape ({self.n_intervals}), got {observed.shape}")
        self.observed = observed
    
    def cum_hazard(self, hazard):
        """
        Calculates the cumulative hazard from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The cumulative hazard calculated by summing the hazard over time steps.

        Notes:
        ------
        - The cumulative hazard is computed as the cumulative sum of the hazard values, 
          scaled by the `interval_length` factor.

        """
        return (self.interval_length * hazard).cumsum(axis=-1)
    
    
    def survival(self, hazard):
        """
        Calculates the survival probability from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The survival probability, computed as the exponential of the negative cumulative hazard.

        Notes:
        ------
        - Survival is calculated as `exp(-cumulative hazard)` where the cumulative hazard 
          is calculated using the `cum_hazard` function.
        """
        return np.exp(-self.cum_hazard(hazard))

    @staticmethod
    @jax.jit
    def calculate_hazard_rate_multi_cov(beta, cov, lambda0):
        return jnp.outer(jnp.exp(cov @ beta), lambda0)
    
    @staticmethod
    @jax.jit
    def calculate_hazard_rate_uni_cov(beta, cov, lambda0):
        return jnp.outer(jnp.exp(beta * cov), lambda0)
    
    @staticmethod
    @jax.jit
    def calculate_hazard_rate_time_varying_cov(beta, cov, lambda0):
        tmp = cov @ beta
        return jnp.exp(tmp[:,:,0]) * lambda0
    
    def build_data(self):
        self.build_surv_covariates()
        self.data_on_model['cov'] = self.cov_all
        self.data_on_model['death'] = self.death
        self.data_on_model['exposure'] = self.exposure
        #self.data_on_model['T'] = self.n_intervals
        #self.data_on_model['N_cov'] = self.cov_all.shape[1]

    def priors(self, sample = False):
        ## Base hazard distribution
        lambda0 = dist.gamma(0.01, 0.01, shape= (self.n_intervals,), name = 'Baseline_rate', sample = sample)
        
        if self.cov_all.ndim == 2:
            ## Covariate effect distribution
            if self.cov_all.shape[1] == 1:
                beta = dist.normal(0, 1000, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[0]}', sample = sample)
            else:
                beta = []
                for i in range(self.cov_all.shape[1]):
                    beta.append(dist.normal(0, 1000, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[i]}', sample = sample))
                beta = jnp.array(beta)
        elif self.cov_all.ndim == 3:
            ## Covariate effect distribution
            beta = []
            for i in range(self.cov_all.shape[2]):
                beta.append(dist.normal(0, 1000, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[i]}', sample = sample))
            beta = jnp.array(beta)

        return lambda0, beta

    def model_univariate(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_uni_cov(beta, cov, lambda0)

        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        dist.poisson(rate = mu + jnp.finfo(mu.dtype).tiny, obs = death)

    def model_multivariate(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_multi_cov(beta, cov, lambda0)

        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)

    def model_time_varying(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_time_varying_cov(beta, cov, lambda0)
        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)
    
    def model(self, death, cov,exposure, censoring = None):
        print("⚠️This function is still in development. Use it with caution. ⚠️")

        if self.cov_all.ndim == 2:
            if self.cov_all.shape[1] == 1: 
                return self.model_univariate(death, cov, exposure, censoring)
            else:
                return self.model_multivariate(death, cov, exposure, censoring)
        elif self.cov_all.ndim == 3:
                return self.model_time_varying(death, cov, exposure, censoring)

    def plot_censoring(self, event='event', time='time', cov='metastasized', xlabel='Time', ylabel='Subject'):
        """
        Plots the censoring status of subjects in a time-to-event dataset.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing the time-to-event data.
        event : str, optional
            The name of the column in `df` indicating the event status (1 = event occurred, 0 = censored).
            Default is 'event'.
        time : str, optional
            The name of the column in `df` representing the time variable. Default is 'time'.
        cov : str, optional
            The name of the column in `df` representing a covariate, such as metastasized status.
            Default is 'metastasized'.
        xlabel : str, optional
            Label for the x-axis. Default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Subject'.

        Returns:
        --------
        None
            This function generates a plot showing censored and uncensored subjects along with a specified covariate.


        """
        self.get_basic_info(event, time, cov)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot censored subjects (event = 0) as red horizontal lines
        ax.hlines(
            self.patients[ self.event == 0], 0,  self.df[ self.event == 0].loc[:, 'time'], color="C3", label="Censored"
        )

        # Plot uncensored subjects (event = 1) as gray horizontal lines
        ax.hlines(
             self.patients[ self.event == 1], 0,  self.df[ self.event == 1].loc[:, 'time'], color="C7", label="Uncensored"
        )

        # Add scatter points for subjects with the specified covariate (e.g., metastasized = 1)
        ax.scatter(
            self.df[self.df.loc[:,cov] == 1].loc[:, time],
            self.patients[self.df.loc[:,cov] == 1],
            color="k",
            zorder=10,
            label=cov,
        )

        # Set plot limits and labels
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylabel(ylabel)

        # Set y-axis limits to provide padding around subjects
        ax.set_ylim(-0.25, self.n_patients + 0.25)

        # Add legend to the plot
        ax.legend(loc="center right")

    def plot_surv(self, lambda0 = 'Baseline_rate', beta = 'beta',
                  xlab='Time', ylab='Survival', covlab = 'treated', title = "Bayesian survival model"):

        base_hazard = self.parent.posteriors[lambda0]        
        met_hazard =self.parent.posteriors[lambda0] * self.parent.posteriors[beta]

        fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))   

        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(base_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C0",
            fill_kwargs={"label": "Had not metastasized"},
        )
        
        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(met_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C1",
            fill_kwargs={"label": "Metastasized"},
        )   

        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(base_hazard), axis = 0), color="darkblue")
        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(met_hazard), axis = 0), color="maroon")   

        hazard_ax.set_xlim(0, self.time.max())
        hazard_ax.set_xlabel(xlab)
        hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")
        hazard_ax.legend(loc=2) 

        az.plot_hdi(self.interval_bounds[:-1], self.survival(base_hazard), ax=surv_ax, smooth=False, color="C0")
        az.plot_hdi(self.interval_bounds[:-1], self.survival(met_hazard), ax=surv_ax, smooth=False, color="C1")  

        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(base_hazard), axis = 0), color="darkblue")
        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(met_hazard), axis = 0), color="maroon")   

        surv_ax.set_xlim(0, self.time.max())
        surv_ax.set_xlabel(ylab)
        surv_ax.set_ylabel("Survival function $S(t)$")  

        fig.suptitle(title);
