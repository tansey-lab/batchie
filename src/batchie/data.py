import logging
from abc import ABC, abstractmethod
from typing import Optional

import h5py
import numpy as np
import pandas
from batchie.common import ArrayType, FloatingPointType, CONTROL_SENTINEL_VALUE

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


def encode_treatment_arrays_to_0_indexed_ids(
    treatment_name_arr: ArrayType,
    treatment_dose_arr: ArrayType,
    control_treatment_name: str = "",
    sentinel_value: int = -1,
):
    is_control = np.array(
        [x == control_treatment_name for x in treatment_name_arr]
    ) | np.array([x <= 0 for x in treatment_dose_arr])

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


def create_single_treatment_effect_map(
    sample_ids: ArrayType,
    treatment_ids: ArrayType,
    observation: ArrayType,
):
    if treatment_ids.shape[1] < 2:
        raise ValueError(
            "Experiment must have more than one treatment to get single treatment effects"
        )

    # find observations where all but one treatment ids are control
    single_treatment_mask = np.sum(treatment_ids == CONTROL_SENTINEL_VALUE, axis=1) == (
        treatment_ids.shape[1] - 1
    )
    single_treatment_observations = observation[single_treatment_mask]
    single_treatment_treatments = np.sort(
        treatment_ids[single_treatment_mask, :], axis=1
    )[:, -1]
    single_treatment_sample_ids = sample_ids[single_treatment_mask]

    result: dict[tuple[int, int], float] = {}

    for current_sample_id in np.unique(sample_ids):
        for current_treatment_id in np.unique(treatment_ids.flatten()):
            if current_treatment_id == CONTROL_SENTINEL_VALUE:
                result[(current_sample_id, current_treatment_id)] = 1.0
                continue

            mask = (single_treatment_treatments == current_treatment_id) & (
                single_treatment_sample_ids == current_sample_id
            )

            if not np.any(mask):
                continue

            single_effect = np.mean(single_treatment_observations[mask])
            result[(current_sample_id, current_treatment_id)] = single_effect

    return result


def create_single_treatment_effect_array(
    sample_ids: ArrayType,
    treatment_ids: ArrayType,
    observation: ArrayType,
):
    single_treatment_effect_map = create_single_treatment_effect_map(
        sample_ids=sample_ids,
        treatment_ids=treatment_ids,
        observation=observation,
    )

    result = np.ones_like(treatment_ids, dtype=float)

    for idx, (current_sample_id, current_treatment_ids) in enumerate(
        zip(
            sample_ids,
            treatment_ids,
        )
    ):
        for treatment_idx, current_treatment_id in enumerate(current_treatment_ids):
            result[idx, treatment_idx] = single_treatment_effect_map[
                (current_sample_id, current_treatment_id)
            ]

    return result


class ExperimentBase(ABC):
    @property
    @abstractmethod
    def plate_ids(self):
        return NotImplemented

    @property
    @abstractmethod
    def sample_ids(self):
        return NotImplemented

    @property
    @abstractmethod
    def treatment_ids(self):
        return NotImplemented

    @property
    @abstractmethod
    def observations(self):
        return NotImplemented

    @property
    @abstractmethod
    def single_treatment_effects(self) -> Optional[ArrayType]:
        return NotImplemented

    @property
    @abstractmethod
    def observation_mask(self):
        return NotImplemented

    @property
    def is_observed(self) -> bool:
        return np.all(self.observation_mask)

    @property
    def size(self):
        return self.treatment_ids.shape[0]

    @property
    def unique_plate_ids(self):
        return np.unique(self.plate_ids)

    @property
    def unique_sample_ids(self):
        return np.unique(self.sample_ids)

    @property
    def unique_treatments(self):
        return np.setdiff1d(np.unique(self.treatment_ids), [CONTROL_SENTINEL_VALUE])

    @property
    def n_unique_samples(self):
        return len(self.unique_sample_ids)

    @property
    def n_unique_treatments(self):
        return len(self.unique_treatments)

    @property
    def treatment_arity(self):
        return self.treatment_ids.shape[1]

    @property
    def n_plates(self):
        return self.unique_plate_ids.shape[0]

    @abstractmethod
    def combine(self, other):
        raise NotImplementedError


class ExperimentSubset(ExperimentBase):
    def __init__(self, experiment: "Experiment", selection_vector: ArrayType):
        self.dataset = experiment

        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if selection_vector.shape[0] != experiment.size:
            raise ValueError(
                "selection_vector must have same number of rows as dataset"
            )

        self.selection_vector = selection_vector

    @property
    def plate_ids(self):
        return self.dataset.plate_ids[self.selection_vector]

    @property
    def sample_ids(self):
        return self.dataset.sample_ids[self.selection_vector]

    @property
    def treatment_ids(self):
        return self.dataset.treatment_ids[self.selection_vector]

    @property
    def observations(self):
        return self.dataset.observations[self.selection_vector]

    @property
    def single_treatment_effects(self) -> Optional[ArrayType]:
        if self.dataset.single_treatment_effects is None:
            return None
        return self.dataset.single_treatment_effects[self.selection_vector]

    @property
    def observation_mask(self):
        return self.dataset.observation_mask[self.selection_vector]

    def invert(self):
        return Plate(self.dataset, ~self.selection_vector)

    def combine(self, other):
        if other.dataset is not self.dataset:
            raise ValueError("Cannot combine two subsets of different datasets")
        return Plate(self.dataset, self.selection_vector | other.selection_vector)

    @classmethod
    def concat(cls, experiment_subsets: list):
        selection_vector = None

        if len(experiment_subsets) == 1:
            return experiment_subsets[0]
        elif len(experiment_subsets) == 0:
            raise ValueError("Cannot concat empty list of experiment subsets")

        for experiment_subset in experiment_subsets:
            if experiment_subset.dataset is not experiment_subsets[0].dataset:
                raise ValueError("Cannot concat subsets of different datasets")

            if selection_vector is None:
                selection_vector = experiment_subset.selection_vector
            else:
                selection_vector = selection_vector | experiment_subset.selection_vector

        return Plate(experiment_subsets[0].dataset, selection_vector)

    def to_dataset(self):
        return Experiment(
            treatment_names=self.dataset.treatment_names[self.selection_vector].copy(),
            treatment_doses=self.dataset.treatment_doses[self.selection_vector].copy(),
            observations=self.dataset.observations[self.selection_vector].copy(),
            sample_names=self.dataset.sample_names[self.selection_vector].copy(),
            plate_names=self.dataset.plate_names[self.selection_vector].copy(),
            control_treatment_name=self.dataset.control_treatment_name,
        )


class Plate(ExperimentSubset):
    @property
    def plate_id(self):
        unique_plate_ids = self.unique_plate_ids
        if len(unique_plate_ids) != 1:
            raise ValueError(
                "Cannot retrieve a plate id from an experiment subset that contains more than one plate"
            )

        return unique_plate_ids[0]

    @property
    def plate_name(self):
        return self.dataset.plate_names[self.selection_vector][0]


class Experiment(ExperimentBase):
    def __init__(
        self,
        treatment_names: ArrayType,
        treatment_doses: ArrayType,
        sample_names: ArrayType,
        plate_names: ArrayType,
        observations: Optional[ArrayType] = None,
        observation_mask: Optional[ArrayType] = None,
        control_treatment_name="",
    ):
        self.control_treatment_name = control_treatment_name
        if (
            len(
                set(
                    [
                        x.shape[0]
                        for x in [
                            treatment_names,
                            treatment_doses,
                            sample_names,
                            plate_names,
                        ]
                    ]
                )
            )
            != 1
        ):
            raise ValueError("All arrays must have the same number of experiments")

        n_experiment_dimension = treatment_names.shape[0]

        if len(treatment_names.shape) != 2:
            raise ValueError("treatment_names must be a column vector")
        if len(treatment_doses.shape) != 2:
            raise ValueError("treatment_doses must be a column vector")

        if treatment_names.shape != treatment_doses.shape:
            raise ValueError(
                "treatment_names, treatment_doses must have the same shape, "
                "but got shapes {}, {}".format(
                    treatment_names.shape, treatment_doses.shape
                )
            )

        if observations is None and observation_mask is not None:
            raise ValueError("observation_mask cannot be provided without observations")

        if observations is not None:
            if observations.shape != (n_experiment_dimension,):
                raise ValueError(
                    "Expected observations to have shape {} but got {}".format(
                        (n_experiment_dimension,), observations.shape
                    )
                )

            if not np.issubdtype(observations.dtype, FloatingPointType):
                raise ValueError("observations must be floats")
            if observation_mask is None:
                observation_mask = np.ones((n_experiment_dimension,), dtype=bool)

        else:
            observation_mask = np.zeros((n_experiment_dimension,), dtype=bool)
            observations = np.zeros((n_experiment_dimension,), dtype=FloatingPointType)

        if not np.issubdtype(treatment_doses.dtype, FloatingPointType):
            raise ValueError("treatment_doses must be floats")

        if not np.issubdtype(treatment_names.dtype, str):
            raise ValueError("treatment_names must be strings")

        if not np.issubdtype(plate_names.dtype, str):
            raise ValueError("plate_names must be strings")

        if not np.issubdtype(sample_names.dtype, str):
            raise ValueError("sample_names must be strings")

        # group observation mask by plate name, check that all groups are either all true or all false
        plate_names_unique = np.unique(plate_names)
        for plate_name in plate_names_unique:
            plate_mask = plate_names == plate_name
            if not np.all(
                observation_mask[plate_mask] == observation_mask[plate_mask][0]
            ):
                raise ValueError(
                    f"Plate {plate_name} has a mixture of observed and not observed outcomes."
                )

        treatment_arity = treatment_names.shape[1]

        dose_class_combos = []
        for i in range(treatment_arity):
            dose_class_combos.append((treatment_names[:, i], treatment_doses[:, i]))
        all_dose_names = np.concatenate([x[0] for x in dose_class_combos])
        all_drug_names = np.concatenate([x[1] for x in dose_class_combos])

        all_dose_class_combos_encoded = encode_treatment_arrays_to_0_indexed_ids(
            treatment_name_arr=all_dose_names,
            treatment_dose_arr=all_drug_names,
            control_treatment_name=self.control_treatment_name,
            sentinel_value=CONTROL_SENTINEL_VALUE,
        )

        self._treatment_ids = np.vstack(
            np.split(all_dose_class_combos_encoded, treatment_names.shape[1])
        ).T
        self._sample_ids = encode_1d_array_to_0_indexed_ids(sample_names)
        self._plate_ids = encode_1d_array_to_0_indexed_ids(plate_names)
        self._observations = observations
        self._observation_mask = observation_mask
        self.sample_names = sample_names
        self.plate_names = plate_names
        self.treatment_names = treatment_names
        self.treatment_doses = treatment_doses

    @property
    def plate_ids(self):
        return self._plate_ids

    @property
    def sample_ids(self):
        return self._sample_ids

    @property
    def treatment_ids(self):
        return self._treatment_ids

    @property
    def observations(self):
        return self._observations

    @property
    def single_treatment_effects(self) -> Optional[ArrayType]:
        try:
            return create_single_treatment_effect_array(
                sample_ids=self.sample_ids,
                treatment_ids=self.treatment_ids,
                observation=self.observations,
            )
        except KeyError:
            logger.warning("Could not create single treatment effects array.")
            return None

    @property
    def observation_mask(self):
        return self._observation_mask

    def get_plate(self, plate_id: int) -> Plate:
        return Plate(self, self.plate_ids == plate_id)

    def set_observed(self, selection_mask: ArrayType, observations: ArrayType):
        if not np.issubdtype(selection_mask.dtype, bool):
            raise ValueError("selection_mask must be bool")
        if not np.issubdtype(observations.dtype, FloatingPointType):
            raise ValueError("observations must be float")

        self._observations[selection_mask] = observations
        self._observation_mask[selection_mask] = True

    @property
    def plates(self):
        return [self.get_plate(x) for x in self.unique_plate_ids]

    def subset(self, selection_vector: ArrayType):
        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if not selection_vector.size == self.size:
            raise ValueError("selection_vector must have same length as dataset")

        return Experiment(
            treatment_names=self.treatment_names[selection_vector],
            treatment_doses=self.treatment_doses[selection_vector],
            observations=self.observations[selection_vector],
            sample_names=self.sample_names[selection_vector],
            plate_names=self.plate_names[selection_vector],
            control_treatment_name=self.control_treatment_name,
        )

    def subset_unobserved(self) -> Optional["Experiment"]:
        if np.any(~self.observation_mask):
            return self.subset(~self.observation_mask)

    def subset_observed(self) -> Optional["Experiment"]:
        if np.any(self.observation_mask):
            return self.subset(self.observation_mask)

    def combine(self, other):
        if not isinstance(other, Experiment):
            raise ValueError("other must be an Experiment")
        if other.control_treatment_name != self.control_treatment_name:
            raise ValueError(
                "Cannot combine experiments with different control treatment names"
            )

        return Experiment(
            treatment_names=np.concatenate(
                [self.treatment_names, other.treatment_names]
            ),
            treatment_doses=np.concatenate(
                [self.treatment_doses, other.treatment_doses]
            ),
            observations=np.concatenate([self.observations, other.observations]),
            observation_mask=np.concatenate(
                [self.observation_mask, other.observation_mask]
            ),
            sample_names=np.concatenate([self.sample_names, other.sample_names]),
            plate_names=np.concatenate([self.plate_names, other.plate_names]),
            control_treatment_name=self.control_treatment_name,
        )

    def save_h5(self, fn):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset(
                "treatment_names", data=np.char.encode(self.treatment_names)
            )
            f.create_dataset("treatment_doses", data=self.treatment_doses)
            f.create_dataset("treatment_ids", data=self.treatment_ids)
            f.create_dataset("observations", data=self.observations)
            f.create_dataset("observation_mask", data=self.observation_mask)
            f.create_dataset("sample_ids", data=self.sample_ids)
            f.create_dataset("sample_names", data=np.char.encode(self.sample_names))
            f.create_dataset("plate_ids", data=self.plate_ids)
            f.create_dataset("plate_names", data=np.char.encode(self.plate_names))
            f.attrs["control_treatment_name"] = self.control_treatment_name

    @staticmethod
    def load_h5(path):
        """Load saved data from h5 archive"""
        with h5py.File(path, "r") as f:
            return Experiment(
                treatment_names=np.char.decode(f["treatment_names"][:], "utf-8"),
                treatment_doses=f["treatment_doses"][:],
                observations=f["observations"][:],
                observation_mask=f["observation_mask"][:],
                sample_names=np.char.decode(f["sample_names"][:], "utf-8"),
                plate_names=np.char.decode(f["plate_names"][:], "utf-8"),
                control_treatment_name=f.attrs["control_treatment_name"],
            )


def randomly_subsample_dataset(
    dataset: Experiment,
    treatment_class_fraction: Optional[float] = None,
    treatment_fraction: Optional[float] = None,
    sample_fraction: Optional[float] = None,
    rng: Optional[np.random.BitGenerator] = None,
) -> (Plate, Plate):
    if rng is None:
        logger.warning(
            "No random number generator provided to randomly_subsample_dataset, using default. "
            "This will not be reproducible"
        )
        rng = np.random.default_rng()

    to_keep_vector = np.ones(dataset.size, dtype=bool)

    # Select all values where the treatment class is in one of the randomly select proportions
    if treatment_class_fraction is not None:
        choices = np.unique(dataset.treatment_ids.flatten())
        n_to_keep = int(treatment_class_fraction * choices.size)
        logger.info(
            f"Randomly selecting {n_to_keep} treatment classes out of {choices.size} total"
        )

        treatment_classes_to_keep = rng.choice(choices, size=n_to_keep, replace=False)
        to_keep_vector = to_keep_vector & np.all(
            np.isin(dataset.treatment_ids, treatment_classes_to_keep), axis=1
        )

    if treatment_fraction is not None:
        choices = np.unique(dataset.treatment_ids.flatten())
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
            np.isin(dataset.treatment_ids, treatments_to_keep), axis=1
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
        f"Keeping {np.sum(to_keep_vector)} experiments out of {dataset.size} total"
    )

    # Create two new dataset objects, one with the experiments to keep, and one with the experiments to drop
    dataset_of_kept_experiments = Plate(
        experiment=dataset, selection_vector=to_keep_vector
    )

    dataset_of_dropped_experiments = Plate(
        experiment=dataset, selection_vector=~to_keep_vector
    )

    return dataset_of_kept_experiments, dataset_of_dropped_experiments


def filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
    dataset: Experiment,
) -> Experiment:
    if dataset.treatment_arity < 2:
        raise ValueError("Dataset must have at least 2 treatments")

    treatment_ids: ArrayType = dataset.treatment_ids

    treatment_selection_vector = np.all(
        ~((treatment_ids == CONTROL_SENTINEL_VALUE).reshape(treatment_ids.shape)),
        axis=1,
    )

    treatments_to_select = np.unique(
        treatment_ids[treatment_selection_vector].flatten()
    )
    treatments_to_select_plus_controls = np.concatenate(
        [treatments_to_select, [CONTROL_SENTINEL_VALUE]]
    )

    logger.info(
        "Filtering {}/{} treatments that appear in at least one combo".format(
            treatments_to_select.size, dataset.treatment_arity
        )
    )

    experiment_selection_vector = np.all(
        np.in1d(treatment_ids, treatments_to_select_plus_controls).reshape(
            treatment_ids.shape
        ),
        axis=1,
    )

    return dataset.subset(experiment_selection_vector)
