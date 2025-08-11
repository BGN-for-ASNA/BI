import pandas as pd
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import arviz as az
import numpyro
from BI.Utils.np_dists import UnifiedDist as dist

class survival():
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
        lambda0 = dist.gamma(0.01, 0.01, shape= intervals.shape, name = 'lambda0')
        ## Covariate effect distribution
        beta = dist.normal(0, 1000, shape = (1,),  name='beta')
        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ = self.hazard_rate(cov = metastasized, beta = beta, lambda0 = lambda0)
        ### Compute exposure rates
        mu = exposure * lambda_
    
        # Likelihood calculation
        dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)

