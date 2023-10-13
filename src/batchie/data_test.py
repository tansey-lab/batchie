import os.path
import shutil
import tempfile

import numpy as np
import numpy.testing
import pytest
import h5py


from batchie.data import (
    Dataset,
    DatasetSubset,
    randomly_subsample_dataset,
    encode_treatment_arrays_to_0_indexed_ids,
    filter_dataset_to_treatments_that_appear_in_at_least_one_combo,
)
from batchie.common import CONTROL_SENTINEL_VALUE


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


def test_dataset_initialization_succeeds_under_correct_condition():
    result = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    assert result.n_experiments == 4
    numpy.testing.assert_array_equal(result.plate_ids, np.array([0, 0, 1, 1]))
    numpy.testing.assert_array_equal(result.sample_ids, np.array([0, 1, 2, 3]))
    numpy.testing.assert_array_equal(
        result.treatment_ids, np.array([[1, 3], [0, 3], [1, 2], [1, -1]])
    )


def test_dataset_serialization():
    dset = Dataset(
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

        dset_loaded = Dataset.load_h5(fn)

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


def test_randomly_subsample_dataset():
    test_dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    to_keep, to_drop = randomly_subsample_dataset(
        dataset=test_dataset, sample_fraction=0.75
    )

    assert to_keep.n_experiments == 3
    assert to_drop.n_experiments == 1


def test_filter_dataset_to_treatments_that_appear_in_at_least_one_combo():
    test_dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", ""], ["d", "e"], ["", "b"], ["", "c"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]]),
    )

    result = filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
        dataset=test_dataset
    )

    assert result.n_experiments == 1
    assert result.n_treatments == 2


def test_data_subset():
    test_dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", ""], ["d", "e"], ["", "b"], ["", "c"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]]),
    )

    subset = DatasetSubset(
        dataset=test_dataset, selection_vector=np.array([True, False, False, True])
    )

    assert subset.n_experiments == 2
    assert subset.n_treatments == 2
    np.testing.assert_array_equal(subset.sample_ids, [0, 3])
    np.testing.assert_array_equal(subset.plate_ids, [0, 1])
    np.testing.assert_array_equal(
        subset.treatment_ids, [[0, CONTROL_SENTINEL_VALUE], [CONTROL_SENTINEL_VALUE, 2]]
    )

    inverted_subset = subset.invert()

    np.testing.assert_array_equal(inverted_subset.sample_ids, [1, 2])
    np.testing.assert_array_equal(subset.plate_ids, [0, 1])
