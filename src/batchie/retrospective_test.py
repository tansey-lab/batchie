import numpy as np
import pytest
from batchie.data import Experiment
from batchie import retrospective
from unittest import mock
from batchie.core import BayesianModel, SamplesHolder


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


@pytest.fixture
def masked_dataset():
    return Experiment(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, True, True, True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d", "a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=str),
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


def test_create_sparse_cover_plate(test_dataset):
    rng = np.random.default_rng(0)

    result = retrospective.create_sparse_cover_plate(test_dataset, rng)

    assert list(np.sort(np.unique(result.sample_ids))) == [0, 1, 2, 3]
    assert list(np.sort(np.unique(result.treatment_ids))) == [0, 1]


@pytest.mark.parametrize("anchor_size", [0, 1])
def test_create_sarcoma_plates(anchor_size, test_dataset):
    rng = np.random.default_rng(0)

    result = retrospective.create_sarcoma_plates(
        test_dataset, subset_size=1, anchor_size=anchor_size, rng=rng
    )

    assert sum([x.size for x in result]) == test_dataset.size
    assert len(result) == len(np.unique(test_dataset.sample_ids))


def test_randomly_sample_plates(test_dataset):
    result = retrospective.randomly_sample_plates(
        test_dataset, proportion_of_plates_to_sample=0.5, rng=np.random.default_rng(0)
    )

    assert result.size == 4
    assert result.unique_plate_ids.shape[0] == 1


def test_randomly_sample_plates_with_force_include(test_dataset):
    result = retrospective.randomly_sample_plates(
        test_dataset,
        proportion_of_plates_to_sample=0.5,
        force_include_plate_ids=[0],
        rng=np.random.default_rng(0),
    )

    assert result.size == 4
    np.testing.assert_array_equal(result.unique_plate_ids, np.array([0]))


def test_reveal_plates(test_dataset, masked_dataset):
    full_dataset = test_dataset

    result = retrospective.reveal_plates(
        full_experiment=full_dataset,
        masked_experiment=masked_dataset,
        plate_ids=[3],
    )

    assert result.observation_mask.all()
    np.testing.assert_array_equal(
        result.observations, np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    )


def test_calculate_mse(test_dataset, masked_dataset):
    full_dataset = test_dataset

    model = mock.MagicMock(BayesianModel)
    model.predict.return_value = 1.0

    samples_holder = mock.MagicMock(SamplesHolder)
    samples_holder.n_samples = 10

    result = retrospective.calculate_mse(
        masked_experiment=masked_dataset,
        full_experiment=full_dataset,
        samples_holder=samples_holder,
        model=model,
    )

    assert result == np.mean((np.array([1.0, 1.0]) - np.array([0.3, 0.4])) ** 2)
