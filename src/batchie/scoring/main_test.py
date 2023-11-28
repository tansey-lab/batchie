import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest

from batchie.core import (
    BayesianModel,
    ThetaHolder,
    Scorer,
    PlatePolicy,
)
from batchie.data import Screen
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.scoring import main


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


def test_chunked_scores_holder():
    scores = main.ChunkedScoresHolder(size=1)
    scores.add_score(1, 0.1)

    tmpdir = tempfile.mkdtemp()

    scores.save_h5(os.path.join(tmpdir, "scores.h5"))

    scores_read = main.ChunkedScoresHolder.load_h5(os.path.join(tmpdir, "scores.h5"))

    assert scores_read.scores[0] == 0.1
    assert scores_read.plate_ids[0] == 1

    try:
        pass
    finally:
        shutil.rmtree(tmpdir)


def test_chunked_scores_holder_empty():
    scores = main.ChunkedScoresHolder(size=0)

    tmpdir = tempfile.mkdtemp()

    scores.save_h5(os.path.join(tmpdir, "scores.h5"))

    scores_read = main.ChunkedScoresHolder.load_h5(os.path.join(tmpdir, "scores.h5"))

    assert scores_read.scores.size == 0
    assert scores_read.plate_ids.size == 0
    try:
        pass
    finally:
        shutil.rmtree(tmpdir)


def test_chunked_scores_holder_concat():
    scores0 = main.ChunkedScoresHolder(size=0)
    scores1 = main.ChunkedScoresHolder(size=1)
    scores1.add_score(1, 0.1)
    scores2 = main.ChunkedScoresHolder(size=2)
    scores2.add_score(2, 0.2)
    scores2.add_score(3, 0.3)

    concatted = main.ChunkedScoresHolder.concat([scores0, scores1, scores2])

    assert concatted.scores[0] == 0.1
    assert concatted.plate_ids[0] == 1
    assert concatted.scores[1] == 0.2
    assert concatted.plate_ids[1] == 2
    assert concatted.scores[2] == 0.3
    assert concatted.plate_ids[2] == 3


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
        thetas=mock.MagicMock(ThetaHolder),
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
        thetas=mock.MagicMock(ThetaHolder),
        screen=test_screen,
        distance_matrix=mock.MagicMock(ChunkedDistanceMatrix),
        progress_bar=True,
        chunk_index=0,
        n_chunks=1,
    )

    assert result.scores.shape == (2,)
