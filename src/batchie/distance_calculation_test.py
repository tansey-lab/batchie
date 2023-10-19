from unittest import mock
from batchie import distance_calculation
from batchie.core import BayesianModel, DistanceMetric, SamplesHolder
import numpy as np
from batchie.data import Experiment
import pytest


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

    samples_holder = mock.Mock()
    samples_holder.n_samples = 3

    result = distance_calculation.calculate_pairwise_distance_matrix_on_predictions(
        model=mock.MagicMock(BayesianModel),
        samples=samples_holder,
        distance_metric=mock.MagicMock(DistanceMetric),
        data=test_dataset,
        chunk_indices=(np.array([0, 1]), np.array([0, 1])),
    )

    assert not result.is_complete()

    result2 = distance_calculation.calculate_pairwise_distance_matrix_on_predictions(
        model=mock.MagicMock(BayesianModel),
        samples=samples_holder,
        distance_metric=mock.MagicMock(DistanceMetric),
        data=test_dataset,
        chunk_indices=(np.array([0, 1, 2]), np.array([0, 1, 2])),
    )

    assert result2.is_complete()
