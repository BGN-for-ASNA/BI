from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm

class link:
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
    @jit
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
