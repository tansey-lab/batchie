from unittest import mock

import numpy as np
import pytest
from batchie.data import Experiment
from batchie.plate_policies.k_per_sample import KPerSamplePlatePolicy


@pytest.fixture
def test_dataset():
    test_dataset = Experiment(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.1, 0.2]),
        sample_names=np.array(["a", "a", "b", "b", "c", "c", "c", "c"], dtype=str),
        plate_names=np.array(["1", "2", "3", "4", "5", "6", "7", "8"], dtype=str),
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


def test_k_per_sample_batcher(test_dataset):
    batcher = KPerSamplePlatePolicy(k=2)

    all_plates = test_dataset.plates

    result = batcher.filter_eligible_plates(
        observed_plates=[],
        unobserved_plates=all_plates,
        rng=mock.MagicMock(),
    )

    assert len(result) == 8

    result_after_one_sample_has_been_selected_twice = batcher.filter_eligible_plates(
        observed_plates=[all_plates[6], all_plates[7]],
        unobserved_plates=[
            all_plates[0],
            all_plates[1],
            all_plates[2],
            all_plates[3],
            all_plates[4],
            all_plates[5],
        ],
        rng=mock.MagicMock(),
    )

    assert len(result_after_one_sample_has_been_selected_twice) == 4

    samples_selected = [
        x.sample_ids[0] for x in result_after_one_sample_has_been_selected_twice
    ]
    assert set(samples_selected) == {0, 1}


def test_k_per_sample_batcher_preconditions():
    bad_plates = Experiment(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.1, 0.2]),
        sample_names=np.array(["a", "a", "b", "b", "c", "c", "c", "c"], dtype=str),
        plate_names=np.array(["1", "1", "1", "1", "2", "2", "2", "2"], dtype=str),
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
    batcher = KPerSamplePlatePolicy(k=2)
    with pytest.raises(ValueError):
        batcher.filter_eligible_plates(
            observed_plates=[],
            unobserved_plates=bad_plates.plates,
            rng=mock.MagicMock(),
        )
