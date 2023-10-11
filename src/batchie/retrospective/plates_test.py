from batchie.data import Dataset, DatasetSubset
import numpy as np
import pytest
from batchie.retrospective.plates import (
    create_sparse_cover_plate,
    create_sarcoma_plates,
)


@pytest.fixture
def test_dataset():
    test_dataset = Dataset(
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


def test_create_sparse_cover_plate(test_dataset):
    rng = np.random.default_rng(0)

    result = create_sparse_cover_plate(test_dataset, rng)

    assert list(np.sort(np.unique(result.sample_ids))) == [0, 1, 2, 3]
    assert list(np.sort(np.unique(result.treatment_ids))) == [0, 1]


@pytest.mark.parametrize("anchor_size", [0, 1])
def test_create_sarcoma_plates(anchor_size, test_dataset):
    rng = np.random.default_rng(0)

    result = create_sarcoma_plates(
        test_dataset, subset_size=1, anchor_size=anchor_size, rng=rng
    )

    assert sum([x.n_experiments for x in result]) == test_dataset.n_experiments
    assert len(result) == len(np.unique(test_dataset.sample_ids))
