from batchie.data import Experiment, ExperimentSubset
import numpy as np
import pytest
from batchie.retrospective import plates


@pytest.fixture
def test_dataset():
    test_dataset = Experiment(
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

    result = plates.create_sparse_cover_plate(test_dataset, rng)

    assert list(np.sort(np.unique(result.sample_ids))) == [0, 1, 2, 3]
    assert list(np.sort(np.unique(result.treatment_ids))) == [0, 1]


@pytest.mark.parametrize("anchor_size", [0, 1])
def test_create_sarcoma_plates(anchor_size, test_dataset):
    rng = np.random.default_rng(0)

    result = plates.create_sarcoma_plates(
        test_dataset, subset_size=1, anchor_size=anchor_size, rng=rng
    )

    assert sum([x.size for x in result]) == test_dataset.size
    assert len(result) == len(np.unique(test_dataset.sample_ids))


def test_randomly_sample_plates(test_dataset):
    result = plates.randomly_sample_plates(
        test_dataset, proportion_of_plates_to_sample=0.5, rng=np.random.default_rng(0)
    )

    assert result.size == 4
    assert result.unique_plate_ids.shape[0] == 1


def test_randomly_sample_plates_with_force_include(test_dataset):
    result = plates.randomly_sample_plates(
        test_dataset,
        proportion_of_plates_to_sample=0.5,
        force_include_plate_ids=[0],
        rng=np.random.default_rng(0),
    )

    assert result.size == 4
    np.testing.assert_array_equal(result.unique_plate_ids, np.array([0]))
