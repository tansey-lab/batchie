"""Methods for sampling from multivariate normal distributions."""
import numpy as np
import scipy as sp
from scipy.linalg import solve_triangular


def sample_mvn_from_precision(
    Q, mu=None, mu_part=None, chol_factor=False, Q_shape=None
):
    """
    Fast sampling from a multivariate normal with precision parameterization.

    Supports sparse arrays.

    :param Q: The precision matrix
    :param mu: If provided, assumes the model is N(mu, Q^-1)
    :param mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky
    decomposition of the precision matrix
    :param Q_shape:
    :return:
    """
    assert np.any([Q_shape is not None, not chol_factor])
    Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
    z = np.random.normal(size=Q.shape[0])
    if isinstance(Lt, np.ma.core.MaskedArray):
        result = np.linalg.solve(
            Lt, z
        )  ## is this lower=True? https://github.com/pykalman/pykalman/issues/83
    else:
        result = solve_triangular(Lt, z, lower=False)
    if mu_part is not None:
        result += sp.linalg.cho_solve((Lt, False), mu_part)
    elif mu is not None:
        result += mu
    return result


def sample_mvn_from_covariance(Q, mu=None, mu_part=None, chol_factor=False):
    """
    Fast sampling from a multivariate normal with covariance parameterization.

    Supports sparse arrays.

    :param Q: The covariance matrix
    :param mu: If provided, assumes the model is N(mu, Q^-1)
    :param mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
    :param chol_factor: If true, assumes Q is a (lower triangular) Cholesky
    decomposition of the precision matrix
    :return:
    """
    # Cholesky factor LL' = Q of the covariance matrix Q
    if chol_factor:
        Lt = Q
        Q = Lt.dot(Lt.T)
    else:
        Lt = np.linalg.cholesky(Q)

    # Get the sample as mu + Lz for z ~ N(0, I)
    z = np.random.normal(size=Q.shape[0])
    result = Lt.dot(z)
    if mu_part is not None:
        result += Q.dot(mu_part)
    elif mu is not None:
        result += mu
    return result


def sample_mvn(
    Q, mu=None, mu_part=None, precision=False, chol_factor=False, Q_shape=None
):
    """
    Fast sampling from a multivariate normal with covariance or precision.

    :param Q:
    :param mu:
    :param mu_part:
    :param precision:
    :param chol_factor:
    :param Q_shape:
    :return:
    """
    assert np.any(
        (mu is None, mu_part is None)
    )  # The mean and mean-part are mutually exclusive
    if precision:
        return sample_mvn_from_precision(
            Q, mu=mu, mu_part=mu_part, chol_factor=chol_factor, Q_shape=Q_shape
        )
    return sample_mvn_from_covariance(
        Q, mu=mu, mu_part=mu_part, chol_factor=chol_factor
    )
