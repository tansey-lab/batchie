import numpy as np
from batchie.common import ArrayType, CONTROL_SENTINEL_VALUE
import h5py


def numpy_array_is_0_indexed_integers(arr: ArrayType):
    # Test numpy array arr contains integers, or something that can be safely cast to int32
    if not np.issubdtype(arr.dtype, int):
        return False

    if CONTROL_SENTINEL_VALUE in arr:
        return np.all(
            np.sort(np.unique(arr))
            == np.concatenate([[-1], np.arange(np.unique(arr).shape[0] - 1)])
        )
    else:
        return np.all(np.sort(np.unique(arr)) == np.arange(np.unique(arr).shape[0]))


class Dataset:
    def __init__(
        self,
        treatments: ArrayType,
        observations: ArrayType,
        sample_ids: ArrayType,
        plate_ids: ArrayType,
    ):
        n_experiments = observations.shape[0]
        if sample_ids.shape[0] != n_experiments:
            raise ValueError("sample_ids must have same length as observations")

        if treatments.shape[0] != n_experiments:
            raise ValueError("treatments must have same length as observations")

        if plate_ids.shape[0] != n_experiments:
            raise ValueError("plate_ids must have same length as observations")

        # Assert sample_ids and plate_ids are integers between 0 and the unique number of samples/plates
        if not numpy_array_is_0_indexed_integers(sample_ids):
            raise ValueError(
                "sample_ids must be integers between 0 and the unique number of samples"
            )

        if not numpy_array_is_0_indexed_integers(plate_ids):
            raise ValueError(
                "plate_ids must be integers between 0 and the unique number of plates"
            )

        if not numpy_array_is_0_indexed_integers(treatments):
            raise ValueError(
                "treatments must be integers between 0 and the unique number of treatments"
            )

        if not np.issubdtype(observations.dtype, float):
            raise ValueError("observations must be floats")

        self.treatments = treatments
        self.observations = observations
        self.sample_ids = sample_ids
        self.plate_ids = plate_ids

    def get_plate(self, plate_id: int) -> "Dataset":
        return Dataset(
            treatments=self.treatments[self.plate_ids == plate_id, :],
            sample_ids=self.sample_ids[self.plate_ids == plate_id],
            observations=self.observations[self.plate_ids == plate_id],
            plate_ids=self.plate_ids[self.plate_ids == plate_id],
        )

    @property
    def unique_plate_ids(self):
        return np.unique(self.plate_ids)

    @property
    def unique_sample_ids(self):
        return np.unique(self.sample_ids)

    @property
    def unique_treatments(self):
        return np.unique(self.treatments)

    @property
    def n_treatments(self):
        return self.treatments.shape[1]

    def save_h5(self, fn):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset("treatments", data=self.treatments)
            f.create_dataset("observations", data=self.observations)
            f.create_dataset("sample_ids", data=self.sample_ids)
            f.create_dataset("plate_ids", data=self.plate_ids)

    @staticmethod
    def load_h5(path):
        """Load saved data from h5 archive"""
        with h5py.File(path, "r") as f:
            treatments = f["treatments"][:]
            observations = f["observations"][:]
            sample_ids = f["sample_ids"][:]
            plate_ids = f["plate_ids"][:]

        return Dataset(
            treatments=treatments,
            observations=observations,
            sample_ids=sample_ids,
            plate_ids=plate_ids,
        )
