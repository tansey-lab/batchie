from unittest import mock

import numpy as np
import pytest
from batchie.core import (
    BayesianModel,
    ThetaHolder,
    Scorer,
    PlatePolicy,
)
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.scoring import main
from batchie.data import Screen


@pytest.fixture()
def test_screen():
    return Screen(
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


def test_select_next_plate_with_policy(test_screen):
    scores = main.ChunkedScoresHolder(size=1)
    scores.add_score(1, 0.1)

    policy = mock.MagicMock(PlatePolicy)
    policy.filter_eligible_plates.return_value = [test_screen.get_plate(1)]

    next_plate = main.select_next_plate(
        scores=scores, screen=test_screen, policy=policy
    )

    assert next_plate.plate_id == 1


def test_select_next_batch_without_policy(test_screen):
    scores = main.ChunkedScoresHolder(size=2)
    scores.add_score(1, 0.2)
    scores.add_score(2, 0.1)

    next_plate = main.select_next_plate(scores=scores, screen=test_screen, policy=None)

    assert next_plate.plate_id == 2


def test_score_chunk(test_screen):
    scorer = mock.MagicMock(spec=Scorer)
    scorer.score.return_value = {0: 0.1}
    result = main.score_chunk(
        model=mock.MagicMock(spec=BayesianModel),
        scorer=scorer,
        samples=mock.MagicMock(ThetaHolder),
        screen=test_screen,
        distance_matrix=mock.MagicMock(ChunkedDistanceMatrix),
        progress_bar=True,
        chunk_index=0,
        n_chunks=3,
    )

    assert result.scores.shape == (1,)

    scorer.score.return_value = {0: 0.1, 1: 0.2}
    result = main.score_chunk(
        model=mock.MagicMock(spec=BayesianModel),
        scorer=scorer,
        samples=mock.MagicMock(ThetaHolder),
        screen=test_screen,
        distance_matrix=mock.MagicMock(ChunkedDistanceMatrix),
        progress_bar=True,
        chunk_index=0,
        n_chunks=1,
    )

    assert result.scores.shape == (2,)
