import h5py
import numpy as np
import logging
from typing import Optional, Callable, Any

import pandas

from batchie.common import ArrayType, CONTROL_SENTINEL_VALUE

logger = logging.getLogger(__name__)


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


def treatment_name_is_control(x):
    return pandas.isna(x) or not x


def treatment_dose_is_control(x):
    return pandas.isna(x) or x == 0 or not x


def encode_treatment_arrays_to_0_indexed_ids(
    treatment_name_arr: ArrayType,
    treatment_dose_arr: ArrayType,
    name_is_control: Callable[[Any], bool] = treatment_name_is_control,
    dose_is_control: Callable[[Any], bool] = treatment_dose_is_control,
    sentinel_value: int = -1,
):
    is_control = np.array([name_is_control(x) for x in treatment_name_arr]) | np.array(
        [dose_is_control(x) for x in treatment_dose_arr]
    )

    name_dose_arr = np.vstack([treatment_name_arr, treatment_dose_arr]).T

    df = pandas.DataFrame(name_dose_arr, columns=["name", "dose"])

    df["is_control"] = is_control

    df = df.drop_duplicates()

    df_no_controls = df[~df.is_control][["name", "dose"]]

    df_no_controls = df_no_controls.sort_values(by=df_no_controls.columns.tolist())

    df_no_controls = df_no_controls.reset_index(drop=True)

    new_arr = df_no_controls.to_numpy()

    mapping = dict(zip([tuple(x) for x in new_arr], df_no_controls.index))

    df_controls = df[df.is_control][["name", "dose"]]
    df_controls_arr = df_controls.to_numpy()

    for row in df_controls_arr:
        mapping[tuple(row)] = sentinel_value

    return np.array([mapping[tuple(x)] for x in name_dose_arr])


def encode_1d_array_to_0_indexed_ids(arr: ArrayType):
    unique_values = np.unique(arr)

    mapping = dict(zip(unique_values, np.arange(len(unique_values))))

    return np.array([mapping[x] for x in arr])


class DatasetSubset:
    def __init__(self, dataset: "Dataset", selection_vector: ArrayType):
        self.dataset = dataset

        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if selection_vector.shape[0] != dataset.n_experiments:
            raise ValueError(
                "selection_vector must have same number of rows as dataset"
            )

        self.selection_vector = selection_vector

    @property
    def n_experiments(self):
        return self.selection_vector.sum()

    @property
    def plate_ids(self):
        return self.dataset.plate_ids[self.selection_vector]

    @property
    def sample_ids(self):
        return self.dataset.sample_ids[self.selection_vector]

    @property
    def treatment_ids(self):
        return self.dataset.treatment_ids[self.selection_vector]


class Dataset:
    def __init__(
        self,
        treatment_names: ArrayType,
        treatment_doses: ArrayType,
        observations: ArrayType,
        sample_names: ArrayType,
        plate_names: ArrayType,
    ):
        if (
            len(
                set(
                    [
                        x.shape[0]
                        for x in [
                            treatment_names,
                            treatment_doses,
                            observations,
                            sample_names,
                            plate_names,
                        ]
                    ]
                )
            )
            != 1
        ):
            raise ValueError("All arrays must have the same number of experiments")

        if treatment_names.shape != treatment_doses.shape:
            raise ValueError(
                "treatment_names, treatment_doses must have the same shape, "
                "but got shapes {}, {}".format(
                    treatment_names.shape, treatment_doses.shape
                )
            )

        if not np.issubdtype(observations.dtype, float):
            raise ValueError("observations must be floats")

        dose_class_combos = []
        for i in range(treatment_names.shape[1]):
            dose_class_combos.append(
                np.vstack([treatment_names[:, i], treatment_doses[:, i]]).T
            )
        all_dose_class_combos = np.concatenate(dose_class_combos)
        all_dose_class_combos_encoded = encode_treatment_arrays_to_0_indexed_ids(
            treatment_name_arr=all_dose_class_combos[:, 0],
            treatment_dose_arr=all_dose_class_combos[:, 1],
            sentinel_value=CONTROL_SENTINEL_VALUE,
        )

        self.treatment_ids = np.vstack(
            np.split(all_dose_class_combos_encoded, treatment_names.shape[1])
        ).T
        self.sample_ids = encode_1d_array_to_0_indexed_ids(sample_names)
        self.plate_ids = encode_1d_array_to_0_indexed_ids(plate_names)
        self.observations = observations
        self.sample_names = sample_names
        self.plate_names = plate_names
        self.treatment_names = treatment_names
        self.treatment_doses = treatment_doses

    def get_plate(self, plate_id: int) -> DatasetSubset:
        return DatasetSubset(self, self.plate_ids == plate_id)

    @property
    def n_experiments(self):
        return self.observations.shape[0]

    @property
    def unique_plate_ids(self):
        return np.unique(self.plate_ids)

    @property
    def unique_sample_ids(self):
        return np.unique(self.sample_ids)

    @property
    def unique_treatments(self):
        return np.unique(self.treatment_ids)

    @property
    def n_treatments(self):
        return self.treatment_ids.shape[1]

    def save_h5(self, fn):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset("treatment_names", data=self.treatment_names)
            f.create_dataset("treatment_doses", data=self.treatment_names)
            f.create_dataset("treatment_ids", data=self.treatment_ids)
            f.create_dataset("observations", data=self.observations)
            f.create_dataset("sample_ids", data=self.sample_ids)
            f.create_dataset("sample_names", data=self.sample_names)
            f.create_dataset("plate_ids", data=self.plate_ids)
            f.create_dataset("plate_names", data=self.plate_names)

    @staticmethod
    def load_h5(path):
        """Load saved data from h5 archive"""
        with h5py.File(path, "r") as f:
            return Dataset(
                treatment_names=f["treatment_names"][:],
                treatment_doses=f["treatment_doses"][:],
                observations=f["observations"][:],
                sample_names=f["sample_names"][:],
                plate_names=f["plate_names"][:],
            )


def randomly_subsample_dataset(
    dataset: Dataset,
    treatment_class_fraction: Optional[float] = None,
    treatment_fraction: Optional[float] = None,
    sample_fraction: Optional[float] = None,
    rng: Optional[np.random.BitGenerator] = None,
):
    if rng is None:
        logger.warning(
            "No random number generator provided to randomly_subsample_dataset, using default. "
            "This will not be reproducible"
        )
        rng = np.random.default_rng()

    to_keep_vector = np.ones(dataset.n_experiments, dtype=bool)

    # Select all values where the treatment class is in one of the randomly select proportions
    if treatment_class_fraction is not None:
        choices = np.unique(dataset.treatment_classes.flatten())
        n_to_keep = int(treatment_class_fraction * choices.size)
        logger.info(
            f"Randomly selecting {n_to_keep} treatment classes out of {choices.size} total"
        )

        treatment_classes_to_keep = rng.choice(choices, size=n_to_keep, replace=False)
        to_keep_vector = to_keep_vector & np.all(
            np.isin(dataset.treatment_classes, treatment_classes_to_keep), axis=1
        )

    if treatment_fraction is not None:
        choices = np.unique(dataset.treatments.flatten())
        choices = choices[choices != CONTROL_SENTINEL_VALUE]
        n_to_keep = int(treatment_fraction * choices.size)
        logger.info(
            f"Randomly selecting {n_to_keep} treatments out of {choices.size} total"
        )

        treatments_to_keep = rng.choice(choices, size=n_to_keep, replace=False)
        #  We don't want to drop controls, so we add it back in
        treatments_to_keep = np.concatenate(
            [CONTROL_SENTINEL_VALUE], treatments_to_keep
        )
        to_keep_vector = to_keep_vector & np.all(
            np.isin(dataset.treatments, treatments_to_keep), axis=1
        )

    if sample_fraction is not None:
        choices = np.unique(dataset.sample_ids.flatten())
        n_to_keep = int(sample_fraction * choices.size)
        logger.info(
            f"Randomly selecting {n_to_keep} samples out of {choices.size} total"
        )

        samples_to_keep = rng.choice(choices, size=n_to_keep, replace=False)
        to_keep_vector = to_keep_vector & np.isin(dataset.sample_ids, samples_to_keep)

    if np.sum(to_keep_vector) == 0:
        raise ValueError("No experiments left after subsampling")

    logger.info(
        f"Keeping {np.sum(to_keep_vector)} experiments out of {dataset.n_experiments} total"
    )

    # Create two new dataset objects, one with the experiments to keep, and one with the experiments to drop
    dataset_of_kept_experiments = Dataset(
        treatments=dataset.treatments[to_keep_vector, :],
        observations=dataset.observations[to_keep_vector],
        sample_names=dataset.sample_ids[to_keep_vector],
        plate_names=dataset.plate_ids[to_keep_vector],
        treatment_names=dataset.treatment_classes[to_keep_vector, :],
        treatment_doses=dataset.treatment_doses[to_keep_vector, :],
    )

    dataset_of_dropped_experiments = Dataset(
        treatments=dataset.treatments[~to_keep_vector, :],
        observations=dataset.observations[~to_keep_vector],
        sample_names=dataset.sample_ids[~to_keep_vector],
        plate_names=dataset.plate_ids[~to_keep_vector],
        treatment_names=dataset.treatment_classes[~to_keep_vector, :],
        treatment_doses=dataset.treatment_doses[~to_keep_vector, :],
    )

    return dataset_of_kept_experiments, dataset_of_dropped_experiments
