import numpy as np
import pytest
from batchie.scoring import gaussian_dbal


def test_iter_combination():
    results = []
    for i in range(10):
        results.append(gaussian_dbal.get_combination_at_sorted_index(i, 5, 2))

    expected = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ]

    assert results == expected


def test_zero_pad_ragged_arrays_to_dense_array():
    arrs = [np.ones((2, 3)), np.ones((5, 5))]

    result = gaussian_dbal.zero_pad_ragged_arrays_to_dense_array(arrs)

    assert result.sum() == 25 + 6
    assert result[0, :, :].sum() == 6
    assert result[1, :, :].sum() == 25


def test_dbal_fast_gauss_scoring_vec():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    def create_plate():
        n_experiments = rng.choice(range(1, max_n_experiments))

        return rng.random((n_thetas, n_experiments))

    per_plate_predictions = [create_plate() for _ in range(n_plates)]

    variances = rng.random((n_thetas,))

    dists = rng.random((n_thetas, n_thetas))

    result = gaussian_dbal.dbal_fast_gauss_scoring_vec(
        per_plate_predictions=per_plate_predictions,
        variances=variances,
        distance_matrix=dists,
        rng=rng,
    )

    assert result.shape == (n_plates,)


def test_dbal_fast_gauss_scoring_vec_fails_if_not_enough_thetas():
    rng = np.random.default_rng(0)

    n_thetas = 2
    n_plates = 10
    max_n_experiments = 96

    def create_plate():
        n_experiments = rng.choice(range(1, max_n_experiments))

        return rng.random((n_thetas, n_experiments))

    per_plate_predictions = [create_plate() for _ in range(n_plates)]

    variances = rng.random((n_thetas,))

    dists = rng.random((n_thetas, n_thetas))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            per_plate_predictions=per_plate_predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )


def test_dbal_fast_gauss_scoring_vec_fails_if_theta_mismatch():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    def create_plate():
        n_experiments = rng.choice(range(1, max_n_experiments))
        random_n_thetas = rng.choice(range(1, n_thetas))

        return rng.random((random_n_thetas, n_experiments))

    per_plate_predictions = [create_plate() for _ in range(n_plates)]

    variances = rng.random((n_thetas,))

    dists = rng.random((n_thetas, n_thetas))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            per_plate_predictions=per_plate_predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )


def test_dbal_fast_gauss_scoring_vec_fails_if_variance_dimension_is_wrong():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    def create_plate():
        n_experiments = rng.choice(range(1, max_n_experiments))

        return rng.random((n_thetas, n_experiments))

    per_plate_predictions = [create_plate() for _ in range(n_plates)]

    variances = rng.random((n_thetas - 1,))

    dists = rng.random((n_thetas, n_thetas))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            per_plate_predictions=per_plate_predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )
