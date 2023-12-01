import os.path
import shutil
import tempfile

import numpy as np
import numpy.testing
import pytest

import batchie.data
from batchie.common import CONTROL_SENTINEL_VALUE
from batchie.data import (
    Screen,
    ScreenSubset,
    Plate,
    encode_treatment_arrays_to_0_indexed_ids,
    filter_dataset_to_treatments_that_appear_in_at_least_one_combo,
)


def test_encode_treatment_arrays_to_0_indexed_ids():
    result = encode_treatment_arrays_to_0_indexed_ids(
        treatment_name_arr=np.array(["a", "b", "a", "b"]),
        treatment_dose_arr=np.array([1, 2, 3, 4]),
    )

    assert result.size == 4
    assert np.sort(np.unique(result)).tolist() == [0, 1, 2, 3]


def test_encode_treatment_arrays_with_controls_to_0_indexed_ids():
    result = encode_treatment_arrays_to_0_indexed_ids(
        treatment_name_arr=np.array(["a", "b", "", "b"], dtype=object),
        treatment_dose_arr=np.array([1, 2, 3, 0]),
        control_treatment_name="",
    )

    assert result.size == 4
    assert np.sort(np.unique(result)).tolist() == [-1, 0, 1]
    assert np.sort(result).tolist() == [-1, -1, 0, 1]


def test_screen_props():
    experiment = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    np.testing.assert_array_equal(
        experiment.observation_mask, np.array([True, True, False, False])
    )
    np.testing.assert_array_equal(experiment.observations, np.array([0.1, 0.2, 0, 0]))
    np.testing.assert_array_equal(
        experiment.sample_names, np.array(["a", "b", "c", "d"])
    )
    np.testing.assert_array_equal(
        experiment.plate_names, np.array(["a", "a", "b", "b"])
    )
    np.testing.assert_array_equal(
        experiment.treatment_names,
        np.array([["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]),
    )
    np.testing.assert_array_equal(experiment.unique_treatments, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(experiment.unique_sample_ids, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(experiment.unique_plate_ids, np.array([0, 1]))
    assert experiment.n_unique_treatments == 4
    assert experiment.n_unique_samples == 4
    assert len(experiment.plates) == 2
    assert experiment.is_observed == False
    assert experiment.size == 4
    assert experiment.treatment_arity == 2
    assert experiment.n_plates == 2

    first_half = experiment.subset(np.array([True, True, False, False])).to_screen()
    second_half = experiment.subset(np.array([False, False, True, True])).to_screen()

    assert first_half.size == 2
    assert second_half.size == 2
    together = first_half.combine(second_half)
    assert together.size == 4
    np.testing.assert_array_equal(
        experiment.treatment_names,
        together.treatment_names,
    )
    np.testing.assert_array_equal(
        experiment.treatment_doses,
        together.treatment_doses,
    )
    np.testing.assert_array_equal(
        experiment.sample_names,
        together.sample_names,
    )
    np.testing.assert_array_equal(
        experiment.plate_names,
        together.plate_names,
    )
    np.testing.assert_array_equal(
        experiment.observations,
        together.observations,
    )

    assert experiment.subset_unobserved().size == 2


def test_screen_hardcode_ids():
    screen = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
        treatment_ids=np.array([[0, 1], [0, 9], [0, 1], [0, 1]]),
        sample_ids=np.array([0, 1, 4, 5]),
        plate_ids=np.array([0, 0, 2, 2]),
    )

    np.testing.assert_array_equal(
        screen.treatment_ids, np.array([[0, 1], [0, 9], [0, 1], [0, 1]])
    )

    np.testing.assert_array_equal(screen.sample_ids, np.array([0, 1, 4, 5]))

    np.testing.assert_array_equal(screen.plate_ids, np.array([0, 0, 2, 2]))

    with pytest.raises(ValueError):
        Screen(
            observations=np.array([0.1, 0.2, 0, 0]),
            observation_mask=np.array([True, True, False, False]),
            sample_names=np.array(["a", "b", "c", "d"], dtype=str),
            plate_names=np.array(["a", "a", "b", "b"], dtype=str),
            treatment_names=np.array(
                [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
            ),
            treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
            treatment_ids=np.array([[0, 1], [0, 9], [0, 1], [0, 1]]),
            sample_ids=np.array([0, 1, 4]),
            plate_ids=np.array([0, 0, 2, 2]),
        )

    with pytest.raises(ValueError):
        Screen(
            observations=np.array([0.1, 0.2, 0, 0]),
            observation_mask=np.array([True, True, False, False]),
            sample_names=np.array(["a", "b", "c", "d"], dtype=str),
            plate_names=np.array(["a", "a", "b", "b"], dtype=str),
            treatment_names=np.array(
                [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
            ),
            treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
            treatment_ids=np.array([[0, 1], [0, 9], [0, 1], [0, 1]]),
            sample_ids=np.array([0, 1, 4, 3]),
            plate_ids=np.array([0, 0, 2]),
        )

    with pytest.raises(ValueError):
        Screen(
            observations=np.array([0.1, 0.2, 0, 0]),
            observation_mask=np.array([True, True, False, False]),
            sample_names=np.array(["a", "b", "c", "d"], dtype=str),
            plate_names=np.array(["a", "a", "b", "b"], dtype=str),
            treatment_names=np.array(
                [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
            ),
            treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0]]),
            treatment_ids=np.array([[0, 1], [0, 9], [0, 1], [0, 1]]),
            sample_ids=np.array([0, 1, 4, 3]),
            plate_ids=np.array([0, 0, 2, 2]),
        )


def test_screen_subset_props():
    experiment = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    experiment_subset = Plate(
        screen=experiment, selection_vector=np.array([True, True, False, False])
    )

    np.testing.assert_array_equal(
        experiment_subset.observation_mask, np.array([True, True])
    )
    np.testing.assert_array_equal(experiment_subset.observations, np.array([0.1, 0.2]))
    np.testing.assert_array_equal(
        experiment_subset.unique_treatments, np.array([0, 1, 3])
    )
    np.testing.assert_array_equal(experiment_subset.unique_sample_ids, np.array([0, 1]))
    np.testing.assert_array_equal(experiment_subset.unique_plate_ids, np.array([0]))
    assert experiment_subset.n_unique_treatments == 3
    assert experiment_subset.n_unique_samples == 2
    assert experiment_subset.is_observed == True
    assert experiment_subset.size == 2
    assert experiment_subset.treatment_arity == 2


def test_screen_resubset():
    experiment = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    experiment_subset = Plate(
        screen=experiment, selection_vector=np.array([True, True, False, False])
    )

    subsubset = experiment_subset.subset(np.array([False, True]))

    assert subsubset.size == 1
    assert subsubset.observations.item() == 0.2


def test_plate_merge():
    screen = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "b", "c", "d"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    plate1, plate2, plate3, plate4 = screen.plates

    plate1.merge(plate2)
    plate3.merge(plate4)

    numpy.testing.assert_array_equal(screen.plate_names, np.array(["a", "a", "c", "c"]))
    numpy.testing.assert_array_equal(screen.plate_ids, np.array([0, 0, 1, 1]))


def test_screen_subset_combine_and_concat():
    experiment = Screen(
        observations=np.array([0.1, 0.2, 0, 0]),
        observation_mask=np.array([True, True, False, False]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    experiment_subset1 = Plate(
        screen=experiment, selection_vector=np.array([True, True, False, False])
    )

    experiment_subset2 = Plate(
        screen=experiment, selection_vector=np.array([False, False, True, True])
    )

    assert experiment_subset1.combine(experiment_subset2).size == 4
    assert Plate.concat([experiment_subset1, experiment_subset2]).size == 4


def test_screen_validates_observation_mask():
    with pytest.raises(ValueError):
        Screen(
            observations=np.array([0.1, 0.2, 0, 0]),
            observation_mask=np.array([True, True, False, True]),
            sample_names=np.array(["a", "b", "c", "d"], dtype=str),
            plate_names=np.array(["a", "a", "b", "b"], dtype=str),
            treatment_names=np.array(
                [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
            ),
            treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
        )


def test_screen_initialization_succeeds_under_correct_condition():
    result = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    assert result.size == 4
    numpy.testing.assert_array_equal(result.plate_ids, np.array([0, 0, 1, 1]))
    numpy.testing.assert_array_equal(result.sample_ids, np.array([0, 1, 2, 3]))
    numpy.testing.assert_array_equal(
        result.treatment_ids, np.array([[1, 3], [0, 3], [1, 2], [1, -1]])
    )


def test_screen_serialization():
    dset = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
        control_treatment_name="zzz",
    )

    tempdir = tempfile.mkdtemp()
    fn = os.path.join(tempdir, "test.h5")

    try:
        dset.save_h5(fn)

        dset_loaded = Screen.load_h5(fn)

        numpy.testing.assert_array_equal(dset_loaded.plate_ids, np.array([0, 0, 1, 1]))
        numpy.testing.assert_array_equal(dset_loaded.sample_ids, np.array([0, 1, 2, 3]))
        numpy.testing.assert_array_equal(
            dset_loaded.treatment_ids, np.array([[1, 3], [0, 3], [1, 2], [1, -1]])
        )
        numpy.testing.assert_array_equal(
            dset_loaded.treatment_names,
            np.array([["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]),
        )
        numpy.testing.assert_array_equal(
            dset_loaded.treatment_doses,
            np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
        )
        numpy.testing.assert_array_equal(
            dset_loaded.sample_names, np.array(["a", "b", "c", "d"])
        )
        assert dset_loaded.control_treatment_name == "zzz"
    finally:
        shutil.rmtree(tempdir)


def test_filter_dataset_to_treatments_that_appear_in_at_least_one_combo():
    test_experiment = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", ""], ["d", "e"], ["", "b"], ["", "c"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]]),
    )

    result = filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
        screen=test_experiment
    )

    assert result.size == 1
    assert result.treatment_arity == 2


def test_screen_subset():
    test_experiment = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", ""], ["d", "e"], ["", "b"], ["", "c"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]]),
    )

    subset = Plate(
        screen=test_experiment,
        selection_vector=np.array([True, False, False, True]),
    )

    assert subset.size == 2
    assert subset.treatment_arity == 2
    np.testing.assert_array_equal(subset.sample_ids, [0, 3])
    np.testing.assert_array_equal(subset.plate_ids, [0, 1])
    np.testing.assert_array_equal(
        subset.treatment_ids, [[0, CONTROL_SENTINEL_VALUE], [CONTROL_SENTINEL_VALUE, 2]]
    )

    inverted_subset = subset.invert()

    np.testing.assert_array_equal(inverted_subset.sample_ids, [1, 2])
    np.testing.assert_array_equal(subset.plate_ids, [0, 1])


def test_create_single_treatment_effect_map():
    test_observations = np.array([1.0, 0.9, 0.1])

    test_treatment_ids = np.array(
        [
            [CONTROL_SENTINEL_VALUE, 1],
            [CONTROL_SENTINEL_VALUE, 2],
            [1, 2],
        ]
    )

    test_sample_ids = np.array([1, 1, 1])

    result = batchie.data.create_single_treatment_effect_map(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
    )

    assert result == {
        (1, 1): 1.0,
        (1, 2): 0.9,
        (1, CONTROL_SENTINEL_VALUE): 1.0,
    }


def test_create_single_treatment_effect_array():
    test_observations = np.array([0.9, 0.8, 0.1])

    test_treatment_ids = np.array(
        [
            [CONTROL_SENTINEL_VALUE, 1],
            [CONTROL_SENTINEL_VALUE, 2],
            [1, 2],
        ]
    )

    test_sample_ids = np.array([1, 1, 1])

    result = batchie.data.create_single_treatment_effect_array(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [1.0, 0.9],
                [1.0, 0.8],
                [0.9, 0.8],
            ],
        ),
    )


def test_filter_dataset_to_unique_treatments():
    test_experiment = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "a", "a", "a"], dtype=str),
        plate_names=np.array(["a", "a", "a", "a"], dtype=str),
        treatment_names=np.array(
            [["a", ""], ["a", ""], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]),
    )

    result = batchie.data.filter_dataset_to_unique_treatments(test_experiment)

    assert result.size == 2
    assert result.selection_vector.tolist() == [True, False, True, False]
