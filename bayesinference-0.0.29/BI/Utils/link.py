from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm

class link:
    """    A class to store and manage various mathematical link functions and their inverses.
    """
    def __init__(self):
        pass
    """
    A class to store and manage various mathematical link functions and their inverses.
    """

    @staticmethod
    @jit
    def logit(x):
        """
        Computes the logit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s) in the range (0, 1).

        Returns
        -------
        float or array-like
            The logit-transformed value(s): log(x / (1 - x)).
        """
        return jnp.log(x / (1 - x))

        return (inv_logit(x) - 0.5) * 2
    
    @staticmethod
    @jit
    def inv_logit(x):
        """
        Computes the inverse logit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The inverse logit-transformed value(s): 1 / (1 + exp(-x)).
        """
        return 1 / (1 + jnp.exp(-x))


    @staticmethod
    @jit
    def inv_logit_scale(x):
        """
        Modified inverse logit function scaling between 0 and 1.
        From https://www.science.org/action/downloadSupplement?doi=10.1126%2Fsciadv.aax9070&file=aax9070_sm.pdf

        Parameters
        ----------
        x : float or array-like
            Input value(s). 
        
        Returns
        -------
        float or array-like
            The inverse logit-transformed value(s): 1 / (1 + exp(-x)).
        """
        return (1 / (1 + jnp.exp(-x)) - 0.5) * 2
        
    @staticmethod
    def probit(p):
        """
        Computes the probit transformation.

        Parameters
        ----------
        p : float or array-like
            Input probability value(s) in the range (0, 1).

        Returns
        -------
        float or array-like
            The probit-transformed value(s), corresponding to the quantile of the 
            standard normal distribution.
        """
        return norm.ppf(p)

    @staticmethod
    def inv_probit(x):
        """
        Computes the inverse probit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The probability value(s) from the cumulative distribution function 
            of the standard normal distribution.
        """
        return norm.cdf(x)

    @staticmethod
    def log(p):
        """
        Computes the natural logarithm.

        Parameters
        ----------
        p : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The natural logarithm of the input value(s).
        """
        return jnp.log(p)

    @staticmethod
    def exp(x):
        """
        Computes the exponential function.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The exponential of the input value(s).
        """
        return jnp.exp(x)

    @staticmethod
    def identity(x):
        """
        Computes the identity function.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The same input value(s), unchanged.
        """
        return x

    @staticmethod
    def cloglog(p):
        """
        Computes the complementary log-log transformation.

        Parameters
        ----------
        p : float or array-like
            Input probability value(s) in the range (0, 1).

        Returns
        -------
        float or array-like
            The complementary log-log transformed value(s): log(-log(1 - p)).
        """
        return jnp.log(-jnp.log(1 - p))

    @staticmethod
    def inv_cloglog(x):
        """
        Computes the inverse complementary log-log transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The inverse complementary log-log transformed probability value(s): 
            1 - exp(-exp(x)).
        """
        return 1 - jnp.exp(-jnp.exp(x))

    @staticmethod
    def reciprocal(x):
        """
        Computes the reciprocal of the input.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The reciprocal of the input value(s): 1 / x.
        """
        return 1 / x

    @staticmethod
    def inv_reciprocal(x):
        """
        Computes the inverse of the reciprocal function.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The same value(s), as the reciprocal function is its own inverse.
        """
        return 1 / x

    @staticmethod
    def sqrt(p):
        """
        Computes the square root.

        Parameters
        ----------
        p : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The square root of the input value(s).
        """
        return jnp.sqrt(p)

    @staticmethod
    def square(x):
        """
        Computes the square of the input.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The square of the input value(s): x ** 2.
        """
        return x ** 2

    @staticmethod
    def LKJcor(corr_matrix, eta):
        """
        Computes the LKJ correlation density.

        The LKJ prior is used for modeling correlation matrices. It is controlled by
        a single parameter, `eta`. Higher values of `eta` imply stronger shrinkage
        towards the identity matrix.

        Parameters
        ----------
        corr_matrix : array-like
            A square, symmetric correlation matrix.
        eta : float
            The shape parameter of the LKJ distribution.

        Returns
        -------
        float
            The LKJ density of the input correlation matrix.
        """
        from jax.scipy.special import gammaln

        k = corr_matrix.shape[0]
        term1 = (2 * eta - 2) * jnp.sum(jnp.log(jnp.diag(corr_matrix)))
        term2 = - (k * (k - 1) / 2) * gammaln(eta)
        term3 = jnp.sum(jnp.log(jnp.linalg.det(corr_matrix)))
        return jnp.exp(term1 + term2 + term3)
