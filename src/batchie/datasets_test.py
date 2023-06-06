import tempfile
import os.path
import numpy as np
import pytest
import shutil
from batchie import datasets
import numpy.testing
from batchie.common import CONTROL_SENTINEL_VALUE


def test_dataset_initialization_succeeds_under_correct_condition():
    datasets.Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
    )


def test_dataset_properties():
    dset = datasets.Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
    )

    assert dset.n_treatments == 2
    assert sorted(dset.unique_plate_ids.tolist()) == [0, 1]
    assert sorted(dset.unique_sample_ids.tolist()) == [0, 1, 2, 3]
    assert sorted(dset.unique_treatments.tolist()) == [0, 1]


def test_dataset_serialization():
    dset = datasets.Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
    )

    tempdir = tempfile.mkdtemp()
    fn = os.path.join(tempdir, "test.h5")

    try:
        dset.save_h5(fn)

        dset_loaded = datasets.Dataset.load_h5(fn)

        numpy.testing.assert_array_equal(dset.treatments, dset_loaded.treatments)
        numpy.testing.assert_array_equal(dset.observations, dset_loaded.observations)
        numpy.testing.assert_array_equal(dset.sample_ids, dset_loaded.sample_ids)
        numpy.testing.assert_array_equal(dset.plate_ids, dset_loaded.plate_ids)
    finally:
        shutil.rmtree(tempdir)


def test_dataset_initialization_succeeds_control_sentinel():
    datasets.Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, CONTROL_SENTINEL_VALUE]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
    )

    datasets.Dataset(
        treatments=np.array([[0, 0], [0, 0], [0, 0], [0, CONTROL_SENTINEL_VALUE]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
    )


def test_dataset_initialization_fails_bad_dtypes():
    with pytest.raises(ValueError, match="observations must be floats"):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0, 0, 0, 1]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1]),
        )

    with pytest.raises(
        ValueError,
        match="treatments must be integers between 0 and the unique number of treatments",
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [3, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1]),
        )

    with pytest.raises(
        ValueError,
        match="plate_ids must be integers between 0 and the unique number of plates",
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 4]),
        )

    with pytest.raises(
        ValueError,
        match="sample_ids must be integers between 0 and the unique number of samples",
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 4]),
            plate_ids=np.array([0, 0, 1, 1]),
        )

    with pytest.raises(
        ValueError, match="sample_ids must have same length as observations"
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3, 4]),
            plate_ids=np.array([0, 0, 1, 1]),
        )

    with pytest.raises(
        ValueError, match="plate_ids must have same length as observations"
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1, 1]),
        )

    with pytest.raises(
        ValueError, match="treatments must have same length as observations"
    ):
        datasets.Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1, 1]),
        )
