import numpy as np
import pytest
from batchie.data import Screen
from batchie import retrospective
from unittest import mock
from batchie.core import BayesianModel, ThetaHolder


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


@pytest.fixture
def unobserved_dataset():
    test_dataset = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False]
        ),
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
    return Screen(
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

    inital_plate_generator = retrospective.SparseCoverPlateGenerator()

    result = inital_plate_generator.generate_and_unmask_initial_plate(test_dataset, rng)

    assert result.unique_plate_ids.size == 2
    assert result.observation_mask.sum() > 0


@pytest.mark.parametrize("anchor_size", [0, 1])
def test_pairwise_plate_generator(anchor_size, unobserved_dataset):
    rng = np.random.default_rng(0)

    plate_generator = retrospective.PairwisePlateGenerator(
        subset_size=1, anchor_size=anchor_size
    )

    result = plate_generator.generate_plates(unobserved_dataset, rng)

    assert result.size == unobserved_dataset.size


@pytest.mark.parametrize("force_include_plate_names", [["a"], None])
def test_randomly_sample_plates(unobserved_dataset, force_include_plate_names):
    rng = np.random.default_rng(0)

    plate_generator = retrospective.PlatePermutationPlateGenerator(
        force_include_plate_names=force_include_plate_names
    )

    result = plate_generator.generate_plates(unobserved_dataset, rng)

    assert list(result.plate_names) != list(unobserved_dataset.plate_names)


def test_reveal_plates(test_dataset, masked_dataset):
    full_dataset = test_dataset

    result = retrospective.reveal_plates(
        observed_screen=full_dataset,
        masked_screen=masked_dataset,
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

    samples_holder = mock.MagicMock(ThetaHolder)
    samples_holder.n_thetas = 10

    result = retrospective.calculate_mse(
        masked_screen=masked_dataset,
        observed_screen=full_dataset,
        thetas=samples_holder,
        model=model,
    )

    assert result == np.mean((np.array([1.0, 1.0]) - np.array([0.3, 0.4])) ** 2)


def test_sample_segregating_permutation_plate_generator():
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "b", "b", "b"], dtype=str),
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

    rng = np.random.default_rng(0)

    result = retrospective.SampleSegregatingPermutationPlateGenerator(
        max_plate_size=5
    ).generate_plates(input_screen, rng)

    assert result.n_plates == 3
    assert sorted([x.size for x in result.plates]) == [2, 3, 3]


def test_sample_segregating_permutation_plate_generator_does_not_affect_observed():
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False, True]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "b", "b", "b", "c"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "a", "a", "b", "b", "c"], dtype=str),
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
                [2.0, 2.0],
            ]
        ),
    )

    rng = np.random.default_rng(0)

    result = retrospective.SampleSegregatingPermutationPlateGenerator(
        max_plate_size=5
    ).generate_plates(input_screen, rng)

    assert result.n_plates == 4
    assert sorted([x.size for x in result.plates]) == [1, 2, 3, 3]


def test_merge_min_plate_smoother():
    smoother = retrospective.MergeMinPlateSmoother(min_size=2)

    input_screen = Screen(
        observations=np.array([0.0, 0.0, 0.0]),
        observation_mask=np.array([False, False, False]),
        sample_names=np.array(["a", "a", "b"], dtype=str),
        plate_names=np.array(["c", "d", "e"], dtype=str),
        treatment_names=np.array(
            [
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
            ]
        ),
    )

    result = smoother.smooth_plates(input_screen, mock.Mock())
    assert result.n_plates == 2
    assert result.size == 3


def test_merge_min_plate_smoother_stops_at_min_size():
    smoother = retrospective.MergeMinPlateSmoother(min_size=2)

    input_screen = Screen(
        observations=np.array([0.0, 0.0, 0.0, 0.0]),
        observation_mask=np.array([False, False, False, False]),
        sample_names=np.array(["a", "a", "a", "b"], dtype=str),
        plate_names=np.array(["c", "d", "e", "f"], dtype=str),
        treatment_names=np.array(
            [
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
            ]
        ),
    )

    result = smoother.smooth_plates(input_screen, mock.Mock())
    assert result.n_plates == 3
    assert result.size == 4


def test_merge_top_bottom_smoother():
    smoother = retrospective.MergeTopBottomPlateSmoother(n_iterations=1)
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "b", "b", "b"], dtype=str),
        plate_names=np.array(["a", "b", "c", "d", "e", "f", "g", "h"], dtype=str),
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

    result = smoother.smooth_plates(input_screen, mock.Mock())

    assert result.n_plates == 4

    smoother_2_iteration = retrospective.MergeTopBottomPlateSmoother(n_iterations=2)
    result = smoother_2_iteration.smooth_plates(input_screen, mock.Mock())

    assert result.n_plates == 3


def test_optimal_size_smoother():
    smoother = retrospective.OptimalSizeSmoother()
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 1.0]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"], dtype=str),
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
                [2.0, 2.0],
            ]
        ),
    )
    result = smoother.smooth_plates(input_screen, rng=np.random.default_rng(0))
    assert result.size == 6
    assert result.n_plates == 3
    assert set([x.size for x in result.plates]) == set([2])


def test_fixed_size_smoother():
    smoother = retrospective.FixedSizeSmoother(plate_size=3)
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 1.0]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"], dtype=str),
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
                [2.0, 2.0],
            ]
        ),
    )
    result = smoother.smooth_plates(input_screen, rng=np.random.default_rng(0))
    assert result.n_plates == 2
    assert set([x.size for x in result.plates]) == {3}


def test_n_plate_per_cell_line_smoother():
    smoother = retrospective.NPlatePerCellLineSmoother(min_n_cell_line_plates=2)
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [True, True, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "b", "b", "b", "a", "a", "a", "a"], dtype=str),
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
    result = smoother.smooth_plates(input_screen, rng=np.random.default_rng(0))
    assert result.n_plates == 3
    assert result.size == 6


def test_batchie_ensemble_plate_smoother():
    smoother = retrospective.BatchieEnsemblePlateSmoother(
        min_size=1, n_iterations=1, min_n_cell_line_plates=1
    )
    input_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        observation_mask=np.array(
            [True, True, False, False, False, False, False, False]
        ),
        sample_names=np.array(["a", "b", "b", "b", "a", "a", "a", "a"], dtype=str),
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
    result = smoother.smooth_plates(input_screen, rng=np.random.default_rng(0))
    assert result.n_plates == 3
    assert result.size == 6
