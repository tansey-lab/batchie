"""Methods for sampling from multivariate normal distributions."""
import numpy as np
import scipy as sp
from scipy.linalg import solve_triangular


def sample_mvn_from_precision(
    Q, mu=None, mu_part=None, chol_factor=False, Q_shape=None, rng=None
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
    :param rng:
    :return:
    """
    if rng is None:
        rng = np.random.default_rng()

    assert np.any([Q_shape is not None, not chol_factor])
    Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
    z = rng.normal(size=Q.shape[0])
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
