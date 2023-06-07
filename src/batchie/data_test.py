import os.path
import shutil
import tempfile

import numpy as np
import numpy.testing
import pytest

from batchie.data import Dataset, randomly_subsample_dataset
from batchie.common import CONTROL_SENTINEL_VALUE


def test_dataset_initialization_succeeds_under_correct_condition():
    Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )


def test_dataset_properties():
    dset = Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    assert dset.n_treatments == 2
    assert sorted(dset.unique_plate_ids.tolist()) == [0, 1]
    assert sorted(dset.unique_sample_ids.tolist()) == [0, 1, 2, 3]
    assert sorted(dset.unique_treatments.tolist()) == [0, 1]


def test_dataset_serialization():
    dset = Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
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


def test_dataset_initialization_succeeds_control_sentinel():
    Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, CONTROL_SENTINEL_VALUE]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    Dataset(
        treatments=np.array([[0, 0], [0, 0], [0, 0], [0, CONTROL_SENTINEL_VALUE]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )


def test_dataset_initialization_fails_bad_dtypes():
    with pytest.raises(ValueError, match="observations must be floats"):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0, 0, 0, 1]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError,
        match="treatments must be integers between 0 and the unique number of treatments",
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [3, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError,
        match="plate_ids must be integers between 0 and the unique number of plates",
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 4]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError,
        match="sample_ids must be integers between 0 and the unique number of samples",
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 4]),
            plate_ids=np.array([0, 0, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError, match="sample_ids must have same length as observations"
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3, 4]),
            plate_ids=np.array([0, 0, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError, match="plate_ids must have same length as observations"
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )

    with pytest.raises(
        ValueError,
        match="treatments, treatment_classes, treatment_doses must all have the same shape",
    ):
        Dataset(
            treatments=np.array([[0, 0], [0, 1], [1, 0]]),
            observations=np.array([0.1, 0.2, 0.3, 0.4]),
            sample_ids=np.array([0, 1, 2, 3]),
            plate_ids=np.array([0, 0, 1, 1, 1]),
            treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        )


def test_randomly_subsample_dataset():
    test_dataset = Dataset(
        treatments=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_ids=np.array([0, 1, 2, 3]),
        plate_ids=np.array([0, 0, 1, 1]),
        treatment_classes=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        treatment_doses=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )

    to_keep, to_drop = randomly_subsample_dataset(
        dataset=test_dataset, sample_fraction=0.75
    )

    assert to_keep.n_experiments == 3
    assert to_drop.n_experiments == 1
