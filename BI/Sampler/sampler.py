import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition
from BI.data.manip import manip
import jax 
class sampler:   
    def __init__(self):
        self.nbdaModel = False
        self.data_on_model = None
        self.nbda.model = None

    def run(self, 
            model = None, 
            potential_fn=None,
            kinetic_fn=None,
            step_size=1.0,
            inverse_mass_matrix=None,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=0.8,
            trajectory_length=None,
            max_tree_depth=10,
            init_strategy= numpyro.infer.init_to_uniform,
            find_heuristic_step_size=False,
            forward_mode_differentiation=False,
            regularize_mass_matrix=True,
            
            num_warmup = 500,
            num_samples = 500,
            num_chains=1,
            thinning=1,
            postprocess_fn=None,
            chain_method="parallel",
            progress_bar=True,
            jit_model_args=False,
            seed = 0):
        """
        Runs the MCMC sampler using the NUTS algorithm.
    
        This method initializes and runs the No-U-Turn Sampler (NUTS) algorithm 
        to generate samples from the posterior distribution defined by the model.
    
        Parameters
        ----------
        model: callable, optional
            The model function that generates the potential function. If not provided,
            `potential_fn` must be specified.
        potential_fn: callable, optional
            A function that computes the potential (negative log density) of the model.
        kinetic_fn: callable, optional
            A function that computes the kinetic energy for HMC. Defaults to a quadratic kinetic
            energy: 0.5 * v^T v.
        step_size: float, optional
            Initial step size for the leapfrog integration. Default is 1.0.
        inverse_mass_matrix: numpy.ndarray, optional
            The inverse mass matrix. If not specified, it's initialized to the identity matrix.
        adapt_step_size: bool, optional
            Whether to adapt the step size during warmup. Defaults to True.
        adapt_mass_matrix: bool, optional
            Whether to adapt the mass matrix during warmup. Defaults to True.
        dense_mass: bool, optional
            Whether to use a dense or diagonal mass matrix. Defaults to False.
        target_accept_prob: float, optional
            The target acceptance probability for NUTS adaptation. Defaults to 0.8.
        trajectory_length: float, optional
            The length of the trajectory for the NUTS algorithm. If not specified, it will
            be calculated automatically.
        max_tree_depth: int, optional
            The maximum depth of the binary tree for NUTS. Defaults to 10.
        init_strategy: callable, optional
            Initialization strategy for the initial parameters. Defaults to 
            `numpyro.infer.init_to_uniform`.
        find_heuristic_step_size: bool, optional
            Whether to use heuristic step size adaptation. Defaults to False.
        forward_mode_differentiation: bool, optional
            Whether to use forward-mode differentiation. Defaults to False.
        regularize_mass_matrix: bool, optional
            Whether to regularize the mass matrix. Defaults to True.
        num_warmup: int, optional
            Number of warmup iterations. Defaults to 500.
        num_samples: int, optional
            Number of samples to generate from the Markov chain. Defaults to 500.
        num_chains: int, optional
            Number of MCMC chains to run. Defaults to 1.
        thinning: int, optional
            Positive integer that controls the fraction of post-warmup samples that are retained.
            Defaults to 1 (no thinning).
        postprocess_fn: callable, optional
            Post-processing function to convert samples to constrained space.
        chain_method: str, optional
            Method for running chains. Options are "parallel", "sequential", or "vectorized".
            Defaults to "parallel".
        progress_bar: bool, optional
            Whether to display a progress bar. Defaults to True.
        jit_model_args: bool, optional
            Whether to compile the potential energy computation. Defaults to False.
        seed: int, optional
            Random seed for the PRNG. Defaults to 0.
    
        Notes
        -----
        The NUTS algorithm parameters (`step_size`, `inverse_mass_matrix`, `target_accept_prob`, 
        etc.) are passed directly to the NUTS kernel. For more details on these parameters, 
        refer to the NumPyro NUTS documentation.  
        """       
        if model is None:
            raise CustomError("Argument model can't be None")
         
        self.model = model

        if self.data_on_model is None and not self.nbdaModel == False:
            self.data_on_model = manip.pd_to_jax(self.model)
        if self.nbdaModel:
            self.model = self.nbda.model

        self.sampler = MCMC(NUTS(self.model,
                                potential_fn=potential_fn,
                                kinetic_fn=kinetic_fn,
                                step_size=step_size,
                                inverse_mass_matrix=inverse_mass_matrix,
                                adapt_step_size=adapt_step_size,
                                adapt_mass_matrix=adapt_mass_matrix,
                                dense_mass=dense_mass,
                                target_accept_prob=target_accept_prob,
                                trajectory_length=trajectory_length,
                                max_tree_depth=max_tree_depth,
                                init_strategy=init_strategy,
                                find_heuristic_step_size=find_heuristic_step_size,
                                forward_mode_differentiation=forward_mode_differentiation,
                                regularize_mass_matrix=regularize_mass_matrix), 
                                num_warmup = num_warmup,
                                num_samples = num_samples,
                                num_chains=num_chains,
                                thinning=thinning,
                                postprocess_fn=postprocess_fn,
                                chain_method=chain_method,
                                progress_bar=progress_bar,
                                jit_model_args=jit_model_args)

        self.sampler.run(jax.random.PRNGKey(seed), **self.data_on_model)
        self.posteriors = self.sampler.get_samples()