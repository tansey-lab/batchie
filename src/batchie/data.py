import logging
from abc import ABC, abstractmethod
from typing import Optional

import h5py
import numpy as np
import pandas
from batchie.common import ArrayType, FloatingPointType, CONTROL_SENTINEL_VALUE

logger = logging.getLogger(__name__)


def numpy_array_is_0_indexed_integers(arr: ArrayType):
    """
    Test numpy array arr contains only integers between 0 and n-1 with no gaps,
    where n is the number of unique values in arr.

    If the array contains :py:const:`batchie.common.CONTROL_SENTINEL_VALUE`,
    then we test that the array contains only integers between 0 and n-2, and the
    sentinel value.


    :param arr: numpy array
    :return: bool
    """
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
):
    """
    Encode treatment names and doses (which are arrays of string)
    to 0-indexed integers, where the control treatment is always mapped to
    :py:const:`batchie.common.CONTROL_SENTINEL_VALUE`

    :param treatment_name_arr: array of treatment names
    :param treatment_dose_arr: array of treatment doses
    :param control_treatment_name: The string value of the control treatment
    """
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
        mapping[tuple(row)] = CONTROL_SENTINEL_VALUE

    return np.array([mapping[tuple(x)] for x in name_dose_arr])


def encode_1d_array_to_0_indexed_ids(arr: ArrayType):
    """
    Encode a 1d array of strings to 0-indexed integers.

    :param arr: 1d array of strings
    :return: integer array containing only values between 0 and n-1,
    where n is the number of unique values in arr
    """
    unique_values = np.unique(arr)

    mapping = dict(zip(unique_values, np.arange(len(unique_values))))

    return np.array([mapping[x] for x in arr])


def create_single_treatment_effect_map(
    sample_ids: ArrayType,
    treatment_ids: ArrayType,
    observation: ArrayType,
):
    """
    Create a map from (sample_id, treatment_id) to single observation (a scalar).

    :param sample_ids: 1d array of sample ids
    :param treatment_ids: 1d array of treatment ids
    :param observation: 1d array of observations
    """
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
    """
    Create a n_observation x n_treatment array where each entry is the single treatment effect
    for the corresponding sample and treatment ids in the input arrays.

    :param sample_ids: 1d array of sample ids
    :param treatment_ids: 2d array of treatment ids
    :param observation: 1d array of observations
    """
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
    """
    Base class for the principal data structure in batchie.

    An :py:class:`batchie.data.Experiment` is a collection of experimental conditions,
    and optionally observations for those of those conditions.
    The conditions are defined by a set of treatment names and doses, and a set of sample names.
    Observations are scalar floating point numbers, with one scalar per condition.

    :py:class:`batchie.data.Experiment` class also defines the concept of a plate, which is a grouping
    of experimental conditions. The terminology plate comes from the world of high throughput biological
    screening, where plastic plates with 96, 384, or 1536 individual wells are used to hold distinct
    biochemical reactions. In batchie, this concept is abstracted to the concept of a plate being
    the discrete unit of experimental conditions that can be observed at one time.
    We also abstract away the concept of the plate having to be a fixed size each time.
    """

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
        """
        Return True if all observations are available, False otherwise

        :return: bool
        """
        return np.all(self.observation_mask)

    @property
    def size(self):
        """
        Return the number of experimental conditions contained
        in the experiment.

        :return: int, number of experimental conditions
        """
        return self.treatment_ids.shape[0]

    @property
    def unique_plate_ids(self):
        """
        Return the unique plate ids in the experiment.

        :return: 1d array of unique plate ids
        """
        return np.unique(self.plate_ids)

    @property
    def unique_sample_ids(self):
        """
        Return the unique sample ids in the experiment.

        :return: 1d array of unique sample ids
        """
        return np.unique(self.sample_ids)

    @property
    def unique_treatments(self):
        """
        Return the unique treatments in the experiment (excludes "control"
        treatments).

        :return: 2d array of unique treatments
        """
        return np.setdiff1d(np.unique(self.treatment_ids), [CONTROL_SENTINEL_VALUE])

    @property
    def n_unique_samples(self):
        """
        Return the number of unique samples in the experiment.

        :return: int, number of unique samples
        """
        return len(self.unique_sample_ids)

    @property
    def n_unique_treatments(self):
        """
        Return the number of unique treatments in the experiment.

        :return: int, number of unique treatments
        """
        return len(self.unique_treatments)

    @property
    def treatment_arity(self):
        """
        Return the number of treatments per experimental condition.

        :return: int, number of treatments per experimental condition
        """
        return self.treatment_ids.shape[1]

    @property
    def n_plates(self):
        """
        Return the number of plates in the experiment.

        :return: int, number of plates
        """
        return self.unique_plate_ids.shape[0]

    @abstractmethod
    def combine(self, other):
        raise NotImplementedError


class ExperimentSubset(ExperimentBase):
    """
    A subset of an :py:class:`batchie.data.Experiment` defined by a boolean selection vector.

    This class is not meant to be instantiated directly, but rather is returned by the
    :py:meth:`batchie.data.Experiment.subset` method.
    """

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
        """
        Return the inverse of this subset,
        i.e. the subset of the experiment that is not contained in this subset.

        :return: :py:class:`batchie.data.ExperimentSubset`
        """
        return Plate(self.dataset, ~self.selection_vector)

    def combine(self, other):
        """
        Union this subset with another subset of the same experiment.

        :param other: :py:class:`batchie.data.ExperimentSubset`
        :return: Unioned :py:class:`batchie.data.ExperimentSubset`
        """
        if other.dataset is not self.dataset:
            raise ValueError("Cannot combine two subsets of different datasets")
        return Plate(self.dataset, self.selection_vector | other.selection_vector)

    @classmethod
    def concat(cls, experiment_subsets: list):
        """
        Concatenate a list of experiment subsets into a single experiment subset.

        :param experiment_subsets: list of :py:class:`batchie.data.ExperimentSubset`
        :return: Unioned :py:class:`batchie.data.ExperimentSubset`
        """
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

    def to_experiment(self):
        """
        Promote this subset to an :py:class:`batchie.data.Experiment`.

        :return: :py:class:`batchie.data.Experiment`
        """
        return Experiment(
            treatment_names=self.dataset.treatment_names[self.selection_vector].copy(),
            treatment_doses=self.dataset.treatment_doses[self.selection_vector].copy(),
            observations=self.dataset.observations[self.selection_vector].copy(),
            sample_names=self.dataset.sample_names[self.selection_vector].copy(),
            plate_names=self.dataset.plate_names[self.selection_vector].copy(),
            control_treatment_name=self.dataset.control_treatment_name,
        )


class Plate(ExperimentSubset):
    """
    A subset of an :py:class:`batchie.data.Experiment` defined by a boolean selection vector

    This class is not meant to be instantiated directly, but rather is returned by the
    :py:meth:`batchie.data.Experiment.get_plate` method.

    The difference between a :py:class:`batchie.data.Plate` and an :py:class:`batchie.data.ExperimentSubset`
    is that a :py:class:`batchie.data.Plate` is guaranteed to contain only one unique plate id.
    """

    @property
    def plate_id(self):
        """
        Return the plate id of this plate.

        :return: int, plate id
        """
        unique_plate_ids = self.unique_plate_ids
        if len(unique_plate_ids) != 1:
            raise ValueError(
                "Cannot retrieve a plate id from an experiment subset that contains more than one plate"
            )

        return unique_plate_ids[0]

    @property
    def plate_name(self):
        """
        Return the original plate name of this plate.

        :return: str, plate name
        """
        return self.dataset.plate_names[self.selection_vector][0]


class Experiment(ExperimentBase):
    """
    The principal data structure in batchie.

    An :py:class:`batchie.data.Experiment` is a collection of experimental conditions that represents
    the entire search space of a high throughput experiment. Some parts of the search space may be
    observed, and some parts may not be observed. Anything not enumerated as an experimental
    condition in this top level class will be "invisible" to batchie.

    An :py:class:`batchie.data.Experiment` can be subset into :py:class:`batchie.data.Plate`s
    or :py:class:`batchie.data.ExperimentSubset` of multiple plates. :py:class:`batchie.data.Experiment`
    is the only data class that can be subdivided.
    """

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
        """
        Return the array of plate ids in the experiment.

        Plate ids are always 0 indexed integers from
        0 to :py:meth:`batchie.data.ExperimentBase.n_unique_plates` - 1 with no gaps.

        :return: 1d array of plate ids
        """
        return self._plate_ids

    @property
    def sample_ids(self):
        """
        Return the array of sample ids in the experiment.

        Sample ids are always 0 indexed integers from
        0 to :py:meth:`batchie.data.ExperimentBase.n_unique_samples` - 1 with no gaps.

        :return: 1d array of sample ids
        """
        return self._sample_ids

    @property
    def treatment_ids(self):
        """
        Return the array of treatment ids in the experiment.

        Treatment ids are always 0 indexed integers from
        0 to :py:meth:`batchie.data.ExperimentBase.n_unique_treatments` - 1 with no gaps.

        :return: 2d array of treatment ids
        """
        return self._treatment_ids

    @property
    def observations(self):
        """
        Return the array of observations in the experiment.

        We do not use any NaN values in our arrays, the observation value
        for a condition set where :py:meth:`batchie.data.Experiment.observation_mask` is False
        is undefined. Its up to the user to decide how to handle this.

        :return: 1d array of observations
        """
        return self._observations

    @property
    def single_treatment_effects(self) -> Optional[ArrayType]:
        """
        Return the array of single treatment effects in the experiment.

        :return: 2d array of single treatment effects
        """
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
        """
        Return the array of observation masks in the experiment.
        If the array is true, it means the condition is observed,
        if false it is unobserved.

        :return: 1d array of observation masks
        """
        return self._observation_mask

    def get_plate(self, plate_id: int) -> Plate:
        """
        Return a :py:class:`batchie.data.Plate` defined by a plate id.

        :param plate_id: int, plate id
        :return: A :py:class:`batchie.data.Plate`
        """
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
        """
        Return a list of all :py:class:`batchie.data.Plate`s in the experiment.

        :return: list of :py:class:`batchie.data.Plate`s
        """
        return [self.get_plate(x) for x in self.unique_plate_ids]

    def subset(self, selection_vector: ArrayType) -> ExperimentSubset:
        """
        Return a :py:class:`batchie.data.ExperimentSubset` defined by a boolean selection vector.

        :param selection_vector: 1d array of bools
        :return: :py:class:`batchie.data.ExperimentSubset`
        """
        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if not selection_vector.size == self.size:
            raise ValueError("selection_vector must have same length as dataset")

        return ExperimentSubset(
            self,
            selection_vector,
        )

    def subset_unobserved(self) -> Optional[ExperimentSubset]:
        """
        Return a :py:class:`batchie.data.ExperimentSubset` containing all
        conditions that are not observed. Returns none if
        :py:meth:`batchie.data.Experiment.is_observed` is True.

        :return: :py:class:`batchie.data.ExperimentSubset`
        """
        if np.any(~self.observation_mask):
            return self.subset(~self.observation_mask)

    def subset_observed(self) -> Optional[ExperimentSubset]:
        """
        Return a :py:class:`batchie.data.ExperimentSubset` containing all
        conditions that are observed. Returns none if
        all conditions are unobserved.

        :return: :py:class:`batchie.data.ExperimentSubset`
        """
        if np.any(self.observation_mask):
            return self.subset(self.observation_mask)

    def combine(self, other):
        """
        Union this experiment with another experiment.

        Warning: treatment, sample, and plate ids are not guaranteed to be
        the same in the resulting new experiment instance.

        :param other: :py:class:`batchie.data.Experiment`
        :return: Unioned :py:class:`batchie.data.Experiment`
        """
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
        """
        Save experiment to h5 archive.

        :param fn: str, path to h5 archive
        """
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
        """
        Load experiment from h5 archive.

        :param path: str, path to h5 archive
        """
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


def filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
    dataset: Experiment,
) -> Experiment:
    """
    Utility function to filter down an :py:class:`batchie.data.Experiment` to only
    the treatments that appear in at least one combo.

    :param dataset: an :py:class:`batchie.data.Experiment`
    :return: A filtered :py:class:`batchie.data.Experiment`
    """
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
