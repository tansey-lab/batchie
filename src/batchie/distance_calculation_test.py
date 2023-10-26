from unittest import mock

import numpy as np
import pytest
from batchie import distance_calculation
from batchie.core import BayesianModel, DistanceMetric, ThetaHolder
from batchie.data import Experiment


@pytest.fixture
def test_dataset():
    test_dataset = Experiment(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2]),
        sample_names=np.array(["a", "a", "a", "b", "b", "b"], dtype=str),
        plate_names=np.array(["1"] * 6, dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
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
            ]
        ),
        control_treatment_name="control",
    )
    return test_dataset


def test_calculate_pairwise_distance_matrix_on_predictions(test_dataset):
    distance_metric = mock.MagicMock(DistanceMetric)
    distance_metric.distance = mock.MagicMock(return_value=0.0)

    samples_holder = mock.MagicMock(ThetaHolder)
    samples_holder.n_thetas = 3

    result = distance_calculation.calculate_pairwise_distance_matrix_on_predictions(
        model=mock.MagicMock(BayesianModel),
        thetas=samples_holder,
        distance_metric=mock.MagicMock(DistanceMetric),
        data=test_dataset,
        chunk_index=0,
        n_chunks=2,
    )

    assert not result.is_complete()

    result2 = distance_calculation.calculate_pairwise_distance_matrix_on_predictions(
        model=mock.MagicMock(BayesianModel),
        thetas=samples_holder,
        distance_metric=mock.MagicMock(DistanceMetric),
        data=test_dataset,
        chunk_index=0,
        n_chunks=1,
    )

    assert result2.is_complete()


@pytest.mark.parametrize(
    "n,n_chunks",
    [
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
        (10, 9),
        (10, 10),
    ],
)
def test_get_lower_triangular_indices_chunk(n, n_chunks):
    mat = np.zeros((n, n))

    for chunk_index in range(n_chunks):
        for x, y in distance_calculation.get_lower_triangular_indices_chunk(
            n=n, chunk_index=chunk_index, n_chunks=n_chunks
        ):
            mat[x, y] += 1

    for x, y in distance_calculation.lower_triangular_indices(n):
        assert mat[x, y] == 1

    assert np.sum(mat) == distance_calculation.get_number_of_lower_triangular_indices(n)

    np.testing.assert_array_equal(mat, np.tril(mat))
