from unittest import mock

import numpy as np
import pytest

from batchie.data import Screen
from batchie.scoring import size


@pytest.fixture
def test_dataset():
    test_dataset = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d", "a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "a", "a", "b", "b"], dtype=str),
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


def test_size_scorer(test_dataset):
    rng = np.random.default_rng(0)

    result = size.SizeScorer().score(
        plates=[x for x in test_dataset.plates],
        samples=mock.MagicMock(),
        distance_matrix=mock.MagicMock(),
        rng=rng,
        model=mock.MagicMock(),
        progress_bar=False,
    )

    assert result == {0: 4, 1: 4}
