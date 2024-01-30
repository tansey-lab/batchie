from unittest import mock

import numpy as np
import pytest

from batchie.core import (
    BayesianModel,
    HomoscedasticBayesianModel,
    Screen,
    ThetaHolder,
    ScreenSubset,
)
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.scoring import gaussian_dbal


@pytest.fixture
def unobserved_dataset():
    test_dataset = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "b", "c", "d", "a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
            ],
            dtype=str,
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
    )
    return test_dataset


@pytest.fixture
def chunked_distance_matrix():
    dm = ChunkedDistanceMatrix(4, chunk_size=2)

    for i in range(4):
        for j in range(i):
            dm.add_value(i, j, 1.0)

    return dm


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

    result = gaussian_dbal.pad_ragged_arrays_to_dense_array(arrs)

    assert result.sum() == 25 + 6
    assert result[0, :, :].sum() == 6
    assert result[1, :, :].sum() == 25


def test_dbal_fast_gauss_scoring_vec():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    # Given 10 plates, one of which has very high prediction variance
    variances = np.vstack(
        [
            rng.gamma(1, 1, size=(n_plates - 1, n_thetas, max_n_experiments)),
            rng.gamma(10, 1, size=(1, n_thetas, max_n_experiments)),
        ]
    )

    predictions = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_plates,
            n_thetas,
            max_n_experiments,
        ),
    )

    dists = rng.random((n_thetas, n_thetas))

    # When: we calculate the gaussian DBAL scores
    result = gaussian_dbal.dbal_fast_gauss_scoring_vec(
        predictions=predictions,
        variances=variances,
        distance_matrix=dists,
        rng=rng,
        max_combos=120,
    )

    # Then: we expect the plate with the highest variance to have the lowest score
    assert result.shape == (n_plates,)
    assert np.argmin(result) == 9


def test_dbal_fast_gauss_scoring_vec_fails_if_not_enough_thetas():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    variances = rng.gamma(1, 1, size=(n_plates, n_thetas, max_n_experiments))

    predictions = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_plates,
            n_thetas,
            max_n_experiments,
        ),
    )

    dists = rng.random((n_thetas - 1, n_thetas - 1))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            predictions=predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )


def test_dbal_fast_gauss_scoring_vec_fails_if_theta_mismatch():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    variances = rng.gamma(1, 1, size=(n_plates, n_thetas, max_n_experiments))

    predictions = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_plates,
            n_thetas,
            max_n_experiments,
        ),
    )

    dists = rng.random((n_thetas - 1, n_thetas))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            predictions=predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )


def test_dbal_fast_gauss_scoring_vec_fails_if_variance_dimension_is_wrong():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    variances = rng.gamma(1, 1, size=(n_plates, n_thetas - 1, max_n_experiments))

    predictions = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_plates,
            n_thetas,
            max_n_experiments,
        ),
    )

    dists = rng.random((n_thetas - 1, n_thetas))

    with pytest.raises(ValueError):
        gaussian_dbal.dbal_fast_gauss_scoring_vec(
            predictions=predictions,
            variances=variances,
            distance_matrix=dists,
            rng=rng,
        )


def test_dbal_fast_gaussian_scoring_homoscedastic():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    predictions = []

    for i in range(n_plates):
        plate_size = rng.choice(np.arange(10, max_n_experiments))
        predictions.append(
            rng.normal(
                loc=0.0,
                scale=1.0,
                size=(
                    n_thetas,
                    plate_size,
                ),
            )
        )

    variances = np.vstack(
        [
            rng.gamma(1, 1, size=(n_plates - 1, n_thetas)),
            rng.gamma(10, 1, size=(1, n_thetas)),
        ]
    )

    dists = rng.random((n_thetas, n_thetas))

    result = gaussian_dbal.dbal_fast_gaussian_scoring_homoscedastic(
        per_plate_predictions=predictions,
        variances=variances,
        distance_matrix=dists,
        rng=rng,
        max_combos=120,
    )

    assert result.shape == (n_plates,)
    assert np.argmin(result) == 9


def test_dbal_fast_gaussian_scoring_heteroscedastic():
    rng = np.random.default_rng(0)

    n_thetas = 10
    n_plates = 10
    max_n_experiments = 96

    predictions = []
    variances = []

    for i in range(n_plates):
        plate_size = rng.choice(np.arange(10, max_n_experiments))
        predictions.append(
            rng.normal(
                loc=0.0,
                scale=1.0,
                size=(
                    n_thetas,
                    plate_size,
                ),
            )
        )
        if i < 9:
            variances.append(rng.gamma(1, 1, size=(n_thetas, plate_size)))
        else:
            variances.append(rng.gamma(10, 1, size=(n_thetas, plate_size)))

    dists = rng.random((n_thetas, n_thetas))

    result = gaussian_dbal.dbal_fast_gaussian_scoring_heteroscedastic(
        per_plate_predictions=predictions,
        variances=variances,
        distance_matrix=dists,
        rng=rng,
        max_combos=120,
    )

    assert result.shape == (n_plates,)
    assert np.argmin(result) == 9


def test_gaussian_dbal_scorer_plates(
    mocker, unobserved_dataset, chunked_distance_matrix
):
    rng = np.random.default_rng(0)
    scorer = gaussian_dbal.GaussianDBALScorer(max_chunk=2, max_triples=5000)

    assert chunked_distance_matrix.is_complete()

    model = mock.Mock(spec=HomoscedasticBayesianModel)
    theta_holder = mock.MagicMock(ThetaHolder)

    theta_holder.n_thetas = chunked_distance_matrix.size
    model.predict.return_value = 1.0
    model.variance.return_value = 1.0

    plates = {p.plate_id: p for p in unobserved_dataset.plates}

    mocker.patch(
        "batchie.scoring.gaussian_dbal.dbal_fast_gauss_scoring_vec",
        return_value=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    result = scorer.score(
        model=model,
        plates=plates,
        distance_matrix=chunked_distance_matrix,
        samples=theta_holder,
        rng=rng,
        progress_bar=False,
    )
    assert result == {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.0}


def test_gaussian_dbal_scorer_subsets(
    mocker, unobserved_dataset, chunked_distance_matrix
):
    rng = np.random.default_rng(0)
    scorer = gaussian_dbal.GaussianDBALScorer(max_chunk=2, max_triples=5000)

    assert chunked_distance_matrix.is_complete()

    model = mock.MagicMock(BayesianModel)
    theta_holder = mock.MagicMock(ThetaHolder)

    theta_holder.n_thetas = chunked_distance_matrix.size
    model.predict.return_value = 1.0
    model.variance.return_value = 1.0

    plates = {p.plate_id: p for p in unobserved_dataset.plates}
    subsets = {}
    for k, v in plates.items():
        subsets[k] = ScreenSubset(v.screen, v.selection_vector)

    mocker.patch(
        "batchie.scoring.gaussian_dbal.dbal_fast_gauss_scoring_vec",
        return_value=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    result = scorer.score(
        model=model,
        plates=subsets,
        distance_matrix=chunked_distance_matrix,
        samples=theta_holder,
        rng=rng,
        progress_bar=False,
    )
    assert result == {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.0}


def test_gaussian_dbal_scorer_empty(unobserved_dataset, chunked_distance_matrix):
    rng = np.random.default_rng(0)
    scorer = gaussian_dbal.GaussianDBALScorer(max_chunk=2, max_triples=5000)

    assert chunked_distance_matrix.is_complete()

    model = mock.MagicMock(BayesianModel)
    theta_holder = mock.MagicMock(ThetaHolder)

    theta_holder.n_thetas = chunked_distance_matrix.size
    model.predict.return_value = 1.0
    model.variance.return_value = 1.0

    res = scorer.score(
        model=model,
        plates={},
        distance_matrix=chunked_distance_matrix,
        samples=theta_holder,
        rng=rng,
        progress_bar=False,
    )

    assert len(res) == 0
