import numpy as np
import tqdm
from typing import Union
from scipy.special import logsumexp, comb

from batchie.common import ArrayType
from batchie.core import (
    Scorer,
    BayesianModel,
    ThetaHolder,
)
from batchie.data import ScreenSubset
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.main import (
    predict_mean_all,
    predict_variance_all,
)


def generate_combination_at_sorted_index(index, n, k):
    """
    Generate all range(n) choose k combinations.

    Represent each combination as a descending sorted tuple.

    Sort all the tuples is ascending order, and return the tuple that would be found at `index`.

    Do this without materializing the actual list of combinations.

    :param index: The index of the combination to return
    :param n: The number of items to choose from
    :param k: The number of items to choose
    :return: A tuple of length k representing the combination
    """
    n_ck = 1
    for n_minus_i, i_plus_1 in zip(range(n, n - k, -1), range(1, k + 1)):
        n_ck *= n_minus_i
        n_ck //= i_plus_1
    current_index = n_ck
    for k in range(k, 0, -1):
        n_ck *= k
        n_ck //= n
        while current_index - n_ck > index:
            current_index -= n_ck
            n_ck *= n - k
            n_ck -= n_ck % k
            n -= 1
            n_ck //= n
        n -= 1
        yield n


def get_combination_at_sorted_index(index, n, k):
    return tuple(generate_combination_at_sorted_index(index, n, k))


def pad_ragged_arrays_to_dense_array(arrays: list[ArrayType], pad_value: float = 0.0):
    """
    Given a list of arrays, each with N dimensions,
    each of which have different sizes, return a dense array of N + 1 dimensions,
    of size (len(array), maximum_of_dimension_0, ... maximum_of_dimension_N)
    where all the arrays are padded to the maximum size.
    Padding value defaults to 0.0.

    :param arrays: A list of arrays
    :param pad_value: A floating point number (default is 0)
    :return: A dense array of the arrays
    """
    max_sizes = np.max([np.array(array.shape) for array in arrays], axis=0)
    result = pad_value * np.ones((len(arrays), *max_sizes), dtype=arrays[0].dtype)
    for i, array in enumerate(arrays):
        result[i, : array.shape[0], : array.shape[1]] = array
    return result


def dbal_fast_gaussian_scoring_heteroscedastic(
    per_plate_predictions: list[ArrayType],
    variances: list[ArrayType],
    distance_matrix: ArrayType,
    rng: np.random.Generator,
    max_combos: int = 5000,
    distance_factor: float = 1.0,
):
    for plate_predictions, plate_variances in zip(per_plate_predictions, variances):
        if plate_predictions.shape != plate_variances.shape:
            raise ValueError(
                "plate_predictions and plate_variances must have the same shape"
            )

    padded_predictions = pad_ragged_arrays_to_dense_array(
        per_plate_predictions, pad_value=0.0
    )

    padded_variances = pad_ragged_arrays_to_dense_array(variances, pad_value=np.nan)

    return dbal_fast_gauss_scoring_vectorized(
        predictions=padded_predictions,
        variances=padded_variances,
        distance_matrix=distance_matrix,
        rng=rng,
        max_combos=max_combos,
        distance_factor=distance_factor,
    )


def dbal_fast_gaussian_scoring_homoscedastic(
    per_plate_predictions: list[ArrayType],
    variances: ArrayType,
    distance_matrix: ArrayType,
    rng: np.random.Generator,
    max_combos: int = 5000,
    distance_factor: float = 1.0,
):
    """
    :param per_plate_predictions: Ragged array of model predictions of length n_plates,
                                  each list element is an array of shape (n_thetas, n_plate_experiments)
    :param variances: an array of variances for model predictions over all plates, of size (n_plate, n_thetas).
    :param distance_matrix: a square array of shape (n_thetas, n_thetas) of distances between model parameterizations
    :param rng: PRNG
    :param max_combos: the maximum number of theta triplets to sample
    :param distance_factor: a multiplicative factor for the distance matrix
    :return: an array of shape (n_plates,) of approximated scores for each plate in per_plate_predictions
    """

    if len(per_plate_predictions) != variances.shape[0]:
        raise ValueError(
            "per_plate_predictions and variances must have the same n_plates dimension"
        )

    for plate_predictions in per_plate_predictions:
        if plate_predictions.shape[0] != variances.shape[1]:
            raise ValueError(
                "plate_predictions and plate_variances must have the same n_thetas dimension"
            )

    padded_predictions = pad_ragged_arrays_to_dense_array(
        per_plate_predictions, pad_value=0.0
    )

    variances_ragged_array = []

    for idx, plate_predictions in enumerate(per_plate_predictions):
        plate_variances = variances[idx]
        n_thetas = plate_variances.shape[0]
        n_experiments = plate_predictions.shape[1]
        variances_ragged_array.append(
            plate_variances[:, None] * np.ones((n_thetas, n_experiments))
        )

    padded_variances = pad_ragged_arrays_to_dense_array(
        variances_ragged_array, pad_value=np.nan
    )

    return dbal_fast_gauss_scoring_vectorized(
        predictions=padded_predictions,
        variances=padded_variances,
        distance_matrix=distance_matrix,
        rng=rng,
        max_combos=max_combos,
        distance_factor=distance_factor,
    )


def dbal_fast_gauss_scoring_vectorized(
    predictions: ArrayType,
    variances: ArrayType,
    distance_matrix: ArrayType,
    rng: np.random.Generator,
    max_combos: int = 5000,
    distance_factor: float = 1.0,
):
    r"""
    Compute the Monte Carlo approximation of the DBAL ideal score $\widehat{s}_n(P)$
    in a vectorized way for each of the given plates.

    $$\widehat{s}_n(P) = \frac{1}{{m \choose 3}} \sum_{i < j < k} d(\theta_i, \theta_j) L_{\theta_i}(\theta_j, \theta_k ; P ) e^{2H_{\theta_i}(P)}$$

    :param predictions: model predictions over all plates of shape (n_plates, n_thetas, n_experiments)
    :param variances: an array of variances for model predictions over all plates, of size (n_plate, n_thetas, max_n_experiments).
                      For plates smaller than the maximum size, the variances should be padded with NaNs up to the maximum size.
    :param distance_matrix: a square array of shape (n_thetas, n_thetas) of distances between model parameterizations
    :param rng: PRNG
    :param max_combos: the maximum number of theta triplets to sample
    :param distance_factor: a multiplicative factor for the distance matrix
    :return: an array of shape (n_plates,) of approximated scores for each plate in per_plate_predictions
    """
    if variances.shape != predictions.shape:
        raise ValueError(
            "variances and predictions should have same shape, got {} and {}".format(
                variances.shape, predictions.shape
            )
        )

    if distance_matrix.shape[1] != distance_matrix.shape[0]:
        raise ValueError("dists must be square, got {}".format(distance_matrix.shape))

    if distance_matrix.shape[0] != predictions.shape[1]:
        raise ValueError(
            "distance_matrix, predictions, and variances must have the same n_thetas dimension"
        )

    mask = ~np.isnan(variances)
    padded_variances = np.nan_to_num(variances, nan=1.0)

    n_plates, n_thetas, max_experiments_per_plate = predictions.shape
    n_theta_combinations = comb(n_thetas, 3, exact=True)

    if not n_theta_combinations:
        raise ValueError("Need at least 3 thetas to compute PDBAL")

    n_combos = min(n_theta_combinations, max_combos)
    unpacked_indices = rng.choice(n_theta_combinations, size=n_combos, replace=False)
    idx1, idx2, idx3 = zip(
        *[get_combination_at_sorted_index(ind, n_thetas, 3) for ind in unpacked_indices]
    )
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    idx3 = np.array(idx3)

    with np.errstate(divide="ignore"):
        log_triple_dists = distance_factor * np.log(
            distance_matrix[idx1, idx2]
            + distance_matrix[idx2, idx3]
            + distance_matrix[idx1, idx3]
        )

    alpha = (
        padded_variances[:, idx1, :] * padded_variances[:, idx2, :]
        + padded_variances[:, idx2, :] * padded_variances[:, idx3, :]
        + padded_variances[:, idx1, :] * padded_variances[:, idx3, :]
    )
    exp_factor = (
        0.5
        * (
            padded_variances[:, idx1, :]
            * padded_variances[:, idx2, :]
            * padded_variances[:, idx3, :]
        )
        / np.square(alpha)
    )

    log_norm_factor = np.sum(mask[:, idx1, :] * 0.5 * np.log(1.0 / alpha), axis=-1)

    d12 = padded_variances[:, idx3, :] * np.square(
        predictions[:, idx1, :] - predictions[:, idx2, :]
    )
    d13 = padded_variances[:, idx2, :] * np.square(
        predictions[:, idx1, :] - predictions[:, idx3, :]
    )
    d23 = padded_variances[:, idx1, :] * np.square(
        predictions[:, idx2, :] - predictions[:, idx3, :]
    )
    ll = np.sum(-exp_factor * (d12 + d13 + d23), axis=-1)  ## n_plates x n_combos

    scores = logsumexp(log_norm_factor + ll + log_triple_dists[np.newaxis, :], axis=1)
    return scores


class GaussianDBALScorer(Scorer):
    def __init__(self, max_chunk=50, max_triples=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_triples = max_triples
        self.max_chunk = max_chunk

    def score(
        self,
        plates: dict[int, ScreenSubset],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
        progress_bar: bool,
    ) -> dict[int, float]:
        if not len(plates):
            return {}

        n_subs = np.ceil(len(plates) / self.max_chunk)
        plate_subgroups = np.array_split(list(plates.keys()), n_subs)
        dense_distance_matrix = distance_matrix.to_dense()

        progress_bar = tqdm.tqdm(total=len(plate_subgroups), disable=not progress_bar)

        result = {}
        for plate_subgroup in plate_subgroups:
            current_plates = [plates[k] for k in plate_subgroup]
            plate_subgroup_mask = None

            for plate in current_plates:
                if plate_subgroup_mask is None:
                    plate_subgroup_mask = plate.selection_vector
                else:
                    plate_subgroup_mask = plate_subgroup_mask | plate.selection_vector

            per_plate_means = [
                predict_mean_all(screen=plate, thetas=samples)
                for plate in current_plates
            ]

            per_plate_variances = [
                predict_variance_all(screen=plate, thetas=samples)
                for plate in current_plates
            ]

            for plate_predictions, plate_variances in zip(
                per_plate_means, per_plate_variances
            ):
                if plate_predictions.shape != plate_variances.shape:
                    raise ValueError(
                        "plate_predictions and plate_variances must have the same shape"
                    )

            padded_means = pad_ragged_arrays_to_dense_array(
                per_plate_means, pad_value=0.0
            )

            padded_variances = pad_ragged_arrays_to_dense_array(
                per_plate_variances, pad_value=np.nan
            )

            vals = dbal_fast_gauss_scoring_vectorized(
                predictions=padded_means,
                variances=padded_variances,
                distance_matrix=dense_distance_matrix,
                rng=rng,
                max_combos=self.max_triples,
            )

            result.update(dict(zip(plate_subgroup, vals)))
            progress_bar.update(len(current_plates))

        if len(result) != len(plates):
            raise ValueError(
                "Expected {} plates to be scored, got {}".format(
                    len(plates), len(result)
                )
            )

        return result
