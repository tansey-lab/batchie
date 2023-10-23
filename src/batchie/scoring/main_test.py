from unittest import mock

import numpy as np
import pytest
from batchie.core import (
    BayesianModel,
    DistanceMatrix,
    ThetaHolder,
    Scorer,
    PlatePolicy,
)
from batchie.scoring import main
from batchie.data import Experiment


@pytest.fixture()
def test_experiment():
    return Experiment(
        observations=np.array([0.1, 0.2, 0, 0, 0, 0]),
        observation_mask=np.array([True, True, False, False, False, False]),
        sample_names=np.array(["a", "b", "c", "d", "e", "f"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "c", "c"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]],
            dtype=str,
        ),
        treatment_doses=np.array(
            [[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0.1], [2.0, 1.0], [2.0, 1.0]]
        ),
    )


def test_select_next_batch_with_policy(test_experiment):
    scorer = mock.MagicMock(Scorer)
    scorer.score.return_value = {
        1: 0.1,
    }

    policy = mock.MagicMock(PlatePolicy)
    policy.filter_eligible_plates.return_value = [test_experiment.get_plate(1)]

    next_batch = main.select_next_batch(
        model=mock.MagicMock(BayesianModel),
        scorer=scorer,
        samples=mock.MagicMock(ThetaHolder),
        experiment_space=test_experiment,
        distance_matrix=mock.MagicMock(DistanceMatrix),
        policy=policy,
    )

    assert len(next_batch) == 1
    assert next_batch[0].plate_id == 1


def test_select_next_batch_without_policy(test_experiment):
    scorer = mock.MagicMock(Scorer)
    scorer.score.return_value = {1: 0.1, 2: 0.01}
    next_batch = main.select_next_batch(
        model=mock.MagicMock(BayesianModel),
        scorer=scorer,
        samples=mock.MagicMock(ThetaHolder),
        experiment_space=test_experiment,
        distance_matrix=mock.MagicMock(DistanceMatrix),
        policy=None,
    )

    assert len(next_batch) == 1
    assert next_batch[0].plate_id == 1
