import numpy as np
import tqdm
from scipy.special import logsumexp, comb

from batchie.common import ArrayType
from batchie.core import (
    Scorer,
    BayesianModel,
    ThetaHolder,
)
from batchie.data import ScreenSubset
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.main import predict_all


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


def zero_pad_ragged_arrays_to_dense_array(arrays: list[ArrayType]):
    """
    Given a list of arrays, each with N dimensions,
    each of which have different sizes, return a dense array of N + 1 dimensions,
    of size (len(array), maximum_of_dimension_0, ... maximum_of_dimension_N)
    where all the arrays are zero-padded to the maximum size.

    :param arrays: A list of arrays
    :return: A dense array of the arrays
    """
    max_sizes = np.max([np.array(array.shape) for array in arrays], axis=0)
    result = np.zeros((len(arrays), *max_sizes), dtype=arrays[0].dtype)
    for i, array in enumerate(arrays):
        result[i, : array.shape[0], : array.shape[1]] = array
    return result


def dbal_fast_gauss_scoring_vec(
    per_plate_predictions: list[ArrayType],
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

    :param per_plate_predictions: a list of arrays (one for each plate) of shape (n_thetas, n_experiments)
    :param variances: an array of shape (n_thetas,) of variances for each model parameterization
    :param distance_matrix: a square array of shape (n_thetas, n_thetas) of distances between model parameterizations
    :param rng: PRNG
    :param max_combos: the maximum number of theta triplets to sample
    :param distance_factor: a multiplicative factor for the distance matrix
    :return: an array of shape (n_plates,) of approximated scores for each plate in per_plate_predictions
    """
    if not len(per_plate_predictions):
        raise ValueError("per_plate_predictions must be non-empty")

    if not len(set([x.shape[0] for x in per_plate_predictions])):
        raise ValueError(
            "All plate_predictions in per_plate_predictions must have the same number of predictors"
        )

    if variances.shape[0] != per_plate_predictions[0].shape[0]:
        raise ValueError(
            "variances has unexpected shape, expected {} got {}".format(
                per_plate_predictions[0].shape[0], variances.shape[0]
            )
        )

    if distance_matrix.shape[1] != distance_matrix.shape[0]:
        raise ValueError("dists must be square, got {}".format(distance_matrix.shape))

    # for performance reasons, we will represent the per plate predictions in a single dense array
    padded_predictions = zero_pad_ragged_arrays_to_dense_array(per_plate_predictions)

    n_plates, n_thetas, max_experiments_per_plate = padded_predictions.shape
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
        variances[idx1] * variances[idx2]
        + variances[idx2] * variances[idx3]
        + variances[idx1] * variances[idx3]
    )
    exp_factor = (
        0.5 * (variances[idx1] * variances[idx2] * variances[idx3]) / np.square(alpha)
    )
    log_norm_factor = 0.5 * max_experiments_per_plate * np.log(1.0 / alpha)
    d12 = np.sum(
        np.square(padded_predictions[:, idx1, :] - padded_predictions[:, idx2, :]),
        axis=-1,
    )
    d13 = np.sum(
        np.square(padded_predictions[:, idx1, :] - padded_predictions[:, idx3, :]),
        axis=-1,
    )
    d23 = np.sum(
        np.square(padded_predictions[:, idx2, :] - padded_predictions[:, idx3, :]),
        axis=-1,
    )
    ll = -exp_factor[np.newaxis, :] * (
        variances[np.newaxis, idx3] * d12
        + variances[np.newaxis, idx2] * d13
        + variances[np.newaxis, idx1] * d23
    )  ## n_triples x m
    scores = logsumexp(
        log_norm_factor[:, np.newaxis] + ll.T + log_triple_dists[:, np.newaxis], axis=0
    )

    return scores


class GaussianDBALScorer(Scorer):
    def __init__(self, max_chunk=50, max_triples=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_triples = max_triples
        self.max_chunk = max_chunk

    def score(
        self,
        model: BayesianModel,
        plates: dict[int, ScreenSubset],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
        progress_bar: bool,
    ) -> dict[int, float]:
        if not len(plates):
            return {}

        variances = np.array([samples.get_variance(i) for i in range(samples.n_thetas)])

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

            per_plate_predictions = [
                predict_all(screen=plate, model=model, thetas=samples)
                for plate in current_plates
            ]

            vals = dbal_fast_gauss_scoring_vec(
                per_plate_predictions=per_plate_predictions,
                variances=variances,
                distance_matrix=dense_distance_matrix,
                max_combos=self.max_triples,
                rng=rng,
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
