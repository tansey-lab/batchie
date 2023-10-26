from unittest import mock

import numpy as np
import pytest
from batchie import distance_calculation
from batchie.core import BayesianModel, DistanceMetric, ThetaHolder
from batchie.data import Experiment
from batchie.distance_calculation import ChunkedDistanceMatrix


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


def test_combine_and_concat():
    dm = ChunkedDistanceMatrix(5, chunk_size=2)
    dm.add_value(1, 0, 2.5)
    dm.add_value(3, 2, 1.5)
    dm2 = ChunkedDistanceMatrix(5, chunk_size=2)
    dm2.add_value(2, 1, 3.5)
    dm2.add_value(4, 3, 0.5)
    dm3 = ChunkedDistanceMatrix(5, chunk_size=2)
    dm3.add_value(2, 2, 6.5)
    dm3.add_value(4, 0, 7.5)

    composed = dm.combine(dm2)

    assert composed.values[0] == 2.5
    assert composed.values[1] == 1.5
    assert composed.values[2] == 3.5
    assert composed.values[3] == 0.5

    with pytest.raises(ValueError):
        dm_other_size = ChunkedDistanceMatrix(6, chunk_size=10)  # Different size
        dm2.combine(dm_other_size)

    concatted = ChunkedDistanceMatrix.concat([dm, dm2])

    assert concatted.values[0] == 2.5
    assert concatted.values[1] == 1.5
    assert concatted.values[2] == 3.5
    assert concatted.values[3] == 0.5

    concatted2 = ChunkedDistanceMatrix.concat([dm, dm2, dm3])

    assert concatted2.values[0] == 2.5
    assert concatted2.values[1] == 1.5
    assert concatted2.values[2] == 3.5
    assert concatted2.values[3] == 0.5
    assert concatted2.values[4] == 6.5
    assert concatted2.values[5] == 7.5
