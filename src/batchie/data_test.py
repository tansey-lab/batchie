import os.path
import shutil
import tempfile

import numpy as np
import numpy.testing
import pytest

from batchie.data import (
    Dataset,
    randomly_subsample_dataset,
    encode_treatment_arrays_to_0_indexed_ids,
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
        treatment_name_arr=np.array(["a", "b", np.nan, "b"], dtype=object),
        treatment_dose_arr=np.array([1, 2, 3, 0]),
    )

    assert result.size == 4
    assert np.sort(np.unique(result)).tolist() == [-1, 0, 1]
    assert np.sort(result).tolist() == [-1, -1, 0, 1]


def test_dataset_initialization_succeeds_under_correct_condition():
    result = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"]),
        plate_names=np.array(["a", "a", "b", "b"]),
        treatment_names=np.array([["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]),
        treatment_doses=np.array([[2, 2], [1, 2], [2, 1], [2, 0]]),
    )

    assert result.n_experiments == 4
    numpy.testing.assert_array_equal(result.plate_ids, np.array([0, 0, 1, 1]))
    numpy.testing.assert_array_equal(result.sample_ids, np.array([0, 1, 2, 3]))
    numpy.testing.assert_array_equal(
        result.treatment_ids, np.array([[1, 3], [0, 3], [1, 2], [1, -1]])
    )


def test_dataset_serialization():
    dset = Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array([0, 1, 2, 3]),
        plate_names=np.array([0, 0, 1, 1]),
        treatment_names=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    tempdir = tempfile.mkdtemp()
    fn = os.path.join(tempdir, "test.h5")

    try:
        dset.save_h5(fn)

        dset_loaded = Dataset.load_h5(fn)

        numpy.testing.assert_array_equal(dset.treatments, dset_loaded.treatments)
        numpy.testing.assert_array_equal(dset.observations, dset_loaded.observations)
        numpy.testing.assert_array_equal(dset.sample_ids, dset_loaded.sample_ids)
        numpy.testing.assert_array_equal(dset.plate_ids, dset_loaded.plate_ids)
    finally:
        shutil.rmtree(tempdir)


def test_randomly_subsample_dataset():
    test_dataset = Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array([0, 1, 2, 3]),
        plate_names=np.array([0, 0, 1, 1]),
        treatment_names=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    to_keep, to_drop = randomly_subsample_dataset(
        dataset=test_dataset, sample_fraction=0.75
    )

    assert to_keep.n_experiments == 3
    assert to_drop.n_experiments == 1
