import logging
from abc import ABC, abstractmethod
from typing import Optional

import h5py
import numpy as np
import pandas

from batchie.common import (
    ArrayType,
    FloatingPointType,
    CONTROL_SENTINEL_VALUE,
    select_unique_zipped_numpy_arrays,
)

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
    with pandas.option_context("mode.copy_on_write", True):
        df = pandas.DataFrame({"name": treatment_name_arr, "dose": treatment_dose_arr})

        df_unique = df.drop_duplicates()

        dose_is_zero = df_unique["dose"] <= 0

        treatment_is_control = df_unique["name"] == control_treatment_name

        is_control = dose_is_zero | treatment_is_control

        df_unique["is_control"] = is_control

        df_unique = df_unique.reset_index(drop=False)

        df_unique["new_index"] = df_unique.index - df_unique.is_control.cumsum()

        selection = df_unique.index[df_unique.is_control]

        df_unique.loc[selection, "new_index"] = CONTROL_SENTINEL_VALUE

        del df_unique["index"]
        del df_unique["is_control"]

        joined = df.merge(df_unique, on=["name", "dose"], how="left")

        return joined.new_index.values


def encode_1d_array_to_0_indexed_ids(arr: ArrayType):
    """
    Encode a 1d array of strings to 0-indexed integers.

    :param arr: 1d array of strings
    :return: integer array containing only values between 0 and n-1,
    where n is the number of unique values in arr
    """
    with pandas.option_context("mode.copy_on_write", True):
        df = pandas.DataFrame({"val": arr})

        df_unique = df.drop_duplicates()
        df_unique = df_unique.reset_index(drop=False)
        df_unique = df_unique.rename(columns={"index": "new_index"})
        joined = df.merge(df_unique, on=["val"], how="left")
        return joined.new_index.values


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


class ScreenBase(ABC):
    """
    Base class for the principal data structure in batchie.

    An :py:class:`batchie.data.Screen` is a collection of experimental conditions,
    and optionally observations for those of those conditions.
    The conditions are defined by a set of treatment names and doses, and a set of sample names.
    Observations are scalar floating point numbers, with one scalar per condition.

    :py:class:`batchie.data.Screen` class also defines the concept of a plate, which is a grouping
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
        Return the unique plate ids in the screen.

        :return: 1d array of unique plate ids
        """
        return np.unique(self.plate_ids)

    @property
    def unique_sample_ids(self):
        """
        Return the unique sample ids in the screen.

        :return: 1d array of unique sample ids
        """
        return np.unique(self.sample_ids)

    @property
    def unique_treatments(self):
        """
        Return the unique treatments in the screen (excludes "control"
        treatments).

        :return: 2d array of unique treatments
        """
        return np.setdiff1d(np.unique(self.treatment_ids), [CONTROL_SENTINEL_VALUE])

    @property
    def n_unique_samples(self):
        """
        Return the number of unique samples in the screen.

        :return: int, number of unique samples
        """
        return len(self.unique_sample_ids)

    @property
    def n_unique_treatments(self):
        """
        Return the number of unique treatments in the screen.

        :return: int, number of unique treatments
        """
        return len(self.unique_treatments)

    @property
    def treatment_arity(self):
        """
        Return the number of treatments per experiment.

        :return: int, number of treatments per experiment
        """
        return self.treatment_ids.shape[1]

    @property
    def n_plates(self):
        """
        Return the number of plates in the screen.

        :return: int, number of plates
        """
        return self.unique_plate_ids.shape[0]

    @abstractmethod
    def combine(self, other):
        raise NotImplementedError


class ScreenSubset(ScreenBase):
    """
    A subset of an :py:class:`batchie.data.Screen` defined by a boolean selection vector.

    This class is not meant to be instantiated directly, but rather is returned by the
    :py:meth:`batchie.data.Screen.subset` method.
    """

    def __init__(self, screen: "Screen", selection_vector: ArrayType):
        self.screen = screen

        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if selection_vector.shape[0] != screen.size:
            raise ValueError(
                "selection_vector must have same number of rows as dataset"
            )

        self.selection_vector = selection_vector

    @property
    def plate_ids(self):
        return self.screen.plate_ids[self.selection_vector]

    @property
    def sample_ids(self):
        return self.screen.sample_ids[self.selection_vector]

    @property
    def treatment_ids(self):
        return self.screen.treatment_ids[self.selection_vector]

    @property
    def observations(self):
        return self.screen.observations[self.selection_vector]

    @property
    def single_treatment_effects(self) -> Optional[ArrayType]:
        if self.screen.single_treatment_effects is None:
            return None
        return self.screen.single_treatment_effects[self.selection_vector]

    @property
    def observation_mask(self):
        return self.screen.observation_mask[self.selection_vector]

    def invert(self):
        """
        Return the inverse of this subset,
        i.e. the subset of the screen that is not contained in this subset.

        :return: :py:class:`batchie.data.ScreenSubset`
        """
        return Plate(self.screen, ~self.selection_vector)

    def combine(self, other):
        """
        Union this subset with another subset of the same screen.

        :param other: :py:class:`batchie.data.ScreenSubset`
        :return: Unioned :py:class:`batchie.data.ScreenSubset`
        """
        if other.screen is not self.screen:
            raise ValueError("Cannot combine two subsets of different datasets")
        return Plate(self.screen, self.selection_vector | other.selection_vector)

    @classmethod
    def concat(cls, screen_subsets: list):
        """
        Concatenate a list of :py:class:`batchie.data.ScreenSubset`s into a single :py:class:`batchie.data.ScreenSubset`.

        :param screen_subsets: list of :py:class:`batchie.data.ScreenSubset`
        :return: Unioned :py:class:`batchie.data.ScreenSubset`
        """
        selection_vector = None

        if len(screen_subsets) == 1:
            return screen_subsets[0]
        elif len(screen_subsets) == 0:
            raise ValueError("Cannot concat empty list")

        for screen_subset in screen_subsets:
            if screen_subset.screen is not screen_subsets[0].screen:
                raise ValueError("Cannot concat subsets of different screens")

            if selection_vector is None:
                selection_vector = screen_subset.selection_vector
            else:
                selection_vector = selection_vector | screen_subset.selection_vector

        return Plate(screen_subsets[0].screen, selection_vector)

    def to_screen(self):
        """
        Promote this subset to an :py:class:`batchie.data.Screen`.

        :return: :py:class:`batchie.data.Screen`
        """
        return Screen(
            treatment_names=self.screen.treatment_names[self.selection_vector].copy(),
            treatment_doses=self.screen.treatment_doses[self.selection_vector].copy(),
            observations=self.screen.observations[self.selection_vector].copy(),
            observation_mask=self.screen.observation_mask[self.selection_vector].copy(),
            sample_names=self.screen.sample_names[self.selection_vector].copy(),
            plate_names=self.screen.plate_names[self.selection_vector].copy(),
            control_treatment_name=self.screen.control_treatment_name,
        )

    def subset(self, selection_vector):
        """
        Return a new :py:class:`batchie.data.ScreenSubset` defined by a boolean selection vector.

        :param selection_vector: 1d array of bools
        :return: :py:class:`batchie.data.ScreenSubset`
        """
        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if not selection_vector.size == self.size:
            raise ValueError("selection_vector must have same length as dataset")

        original_selection_vector = self.selection_vector.copy()

        indexes = np.where(original_selection_vector)[0]

        original_selection_vector[indexes] = selection_vector

        return ScreenSubset(
            self.screen,
            original_selection_vector,
        )


class Plate(ScreenSubset):
    """
    A subset of an :py:class:`batchie.data.Screen` defined by a boolean selection vector

    This class is not meant to be instantiated directly, but rather is returned by the
    :py:class:`batchie.data.Screen.get_plate` method.

    The difference between a :py:class:`batchie.data.Plate` and an :py:class:`batchie.data.ScreenSubset`
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
        return self.screen.plate_names[self.selection_vector][0]

    def merge(self, other):
        """
        Merge this plate with another plate, mutate the parent :py:class:`batchie.data.Screen` in place.

        :param other: :py:class:`batchie.data.Plate`
        """
        if other.screen is not self.screen:
            raise ValueError("Cannot merge two plates from different screens")
        self.selection_vector = self.selection_vector | other.selection_vector
        self.screen.plate_names[self.selection_vector] = self.plate_name
        self.screen._plate_ids = encode_1d_array_to_0_indexed_ids(
            self.screen.plate_names
        )
        return self

    def __lt__(self, other):
        return self.size < other.size


class Screen(ScreenBase):
    """
    The principal data structure in batchie.

    An :py:class:`batchie.data.Screen` is a collection of experiments.
    Some of the experiments may be observed and some may not be observed.
    Anything not enumerated as an experimental condition in this top level
    class will be "invisible" to batchie.

    An :py:class:`batchie.data.Screen` can be subset into :py:class:`batchie.data.Plate`s
    or :py:class:`batchie.data.ScreenSubset` of multiple plates. :py:class:`batchie.data.Screen`
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
        treatment_ids: Optional[ArrayType] = None,
        sample_ids: Optional[ArrayType] = None,
        plate_ids: Optional[ArrayType] = None,
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

        if treatment_ids is not None:
            if treatment_ids.shape != treatment_names.shape:
                raise ValueError(
                    "treatment_ids must have same shape as treatment_names"
                )
            if not np.issubdtype(treatment_ids.dtype, int):
                raise ValueError("treatment_ids must be ints")
            self._treatment_ids = treatment_ids
        else:
            self._treatment_ids = np.vstack(
                np.split(all_dose_class_combos_encoded, treatment_names.shape[1])
            ).T

        if sample_ids is not None:
            if sample_ids.shape != sample_names.shape:
                raise ValueError("sample_ids must have same shape as sample_names")
            if not np.issubdtype(sample_ids.dtype, int):
                raise ValueError("sample_ids must be ints")
            self._sample_ids = sample_ids
        else:
            self._sample_ids = encode_1d_array_to_0_indexed_ids(sample_names)

        if plate_ids is not None:
            if plate_ids.shape != plate_names.shape:
                raise ValueError("plate_ids must have same shape as plate_names")
            if not np.issubdtype(plate_ids.dtype, int):
                raise ValueError("plate_ids must be ints")
            self._plate_ids = plate_ids
        else:
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
        Return the array of plate ids in the screen.

        Plate ids are always 0 indexed integers from
        0 to :py:class:`batchie.data.ScreenBase.n_unique_plates` - 1 with no gaps.

        :return: 1d array of plate ids
        """
        return self._plate_ids

    @property
    def sample_ids(self):
        """
        Return the array of sample ids in the screen.

        Sample ids are always 0 indexed integers from
        0 to :py:class:`batchie.data.ScreenBase.n_unique_samples` - 1 with no gaps.

        :return: 1d array of sample ids
        """
        return self._sample_ids

    @property
    def treatment_ids(self):
        """
        Return the array of treatment ids in the screen.

        Treatment ids are always 0 indexed integers from
        0 to :py:class:`batchie.data.ScreenBase.n_unique_treatments` - 1 with no gaps.

        :return: 2d array of treatment ids
        """
        return self._treatment_ids

    @property
    def observations(self):
        """
        Return the array of observations in the screen.

        We do not use any NaN values in our arrays, the observation value
        for a condition set where :py:class:`batchie.data.Screen.observation_mask` is False
        is undefined. Its up to the user to decide how to handle this.

        :return: 1d array of observations
        """
        return self._observations

    @property
    def single_treatment_effects(self) -> Optional[ArrayType]:
        """
        Return the array of single treatment effects in the screen.

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
        Return the array of observation masks in the screen.
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
        Return a list of all :py:class:`batchie.data.Plate`s in the screen.

        :return: list of :py:class:`batchie.data.Plate`s
        """
        return [self.get_plate(x) for x in self.unique_plate_ids]

    def subset(self, selection_vector: ArrayType) -> ScreenSubset:
        """
        Return a :py:class:`batchie.data.ScreenSubset` defined by a boolean selection vector.

        :param selection_vector: 1d array of bools
        :return: :py:class:`batchie.data.ScreenSubset`
        """
        if not np.issubdtype(selection_vector.dtype, bool):
            raise ValueError("selection_vector must be bool")

        if not selection_vector.size == self.size:
            raise ValueError("selection_vector must have same length as dataset")

        return ScreenSubset(
            self,
            selection_vector,
        )

    def subset_unobserved(self) -> Optional[ScreenSubset]:
        """
        Return a :py:class:`batchie.data.ScreenSubset` containing all
        conditions that are not observed. Returns none if
        :py:class:`batchie.data.Screen.is_observed` is True.

        :return: :py:class:`batchie.data.ScreenSubset`
        """
        if np.any(~self.observation_mask):
            return self.subset(~self.observation_mask)

    def subset_observed(self) -> Optional[ScreenSubset]:
        """
        Return a :py:class:`batchie.data.ScreenSubset` containing all
        conditions that are observed. Returns none if
        all conditions are unobserved.

        :return: :py:class:`batchie.data.ScreenSubset`
        """
        if np.any(self.observation_mask):
            return self.subset(self.observation_mask)

    def combine(self, other):
        """
        Union this screen with another screen.

        Warning: treatment, sample, and plate ids are not guaranteed to be
        the same in the resulting new screen instance.

        :param other: :py:class:`batchie.data.Screen`
        :return: Unioned :py:class:`batchie.data.Screen`
        """
        if not isinstance(other, Screen):
            raise ValueError("other must be a Screen instance")
        if other.control_treatment_name != self.control_treatment_name:
            raise ValueError(
                "Cannot combine screens with different control treatment names"
            )

        return Screen(
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

    @classmethod
    def concat(cls, screens: list["Screen"]):
        """
        Concatenate a list of :py:class:`batchie.data.Screen`s into a single :py:class:`batchie.data.Screen`.

        :param screens: list of :py:class:`batchie.data.Screen`
        :return: Unioned :py:class:`batchie.data.Screen`
        """
        if len(screens) == 1:
            return screens[0]
        elif len(screens) == 0:
            raise ValueError("Cannot concat empty list")

        result = screens[0]

        for screen in screens[1:]:
            result = result.combine(screen)

        return result

    def save_h5(self, fn):
        """
        Save screen to h5 archive.

        :param fn: str, path to h5 archive
        """
        with h5py.File(fn, "w") as f:
            f.create_dataset(
                "treatment_names",
                data=np.char.encode(self.treatment_names),
                compression="gzip",
            )
            f.create_dataset(
                "treatment_doses", data=self.treatment_doses, compression="gzip"
            )
            f.create_dataset(
                "treatment_ids", data=self.treatment_ids, compression="gzip"
            )
            f.create_dataset("observations", data=self.observations, compression="gzip")
            f.create_dataset(
                "observation_mask", data=self.observation_mask, compression="gzip"
            )
            f.create_dataset("sample_ids", data=self.sample_ids, compression="gzip")
            f.create_dataset(
                "sample_names",
                data=np.char.encode(self.sample_names),
                compression="gzip",
            )
            f.create_dataset("plate_ids", data=self.plate_ids, compression="gzip")
            f.create_dataset(
                "plate_names", data=np.char.encode(self.plate_names), compression="gzip"
            )
            f.attrs["control_treatment_name"] = self.control_treatment_name

    @staticmethod
    def load_h5(path):
        """
        Load screen from h5 archive.

        :param path: str, path to h5 archive
        """
        with h5py.File(path, "r") as f:
            return Screen(
                treatment_names=np.char.decode(f["treatment_names"][:], "utf-8"),
                treatment_doses=f["treatment_doses"][:],
                observations=f["observations"][:],
                observation_mask=f["observation_mask"][:],
                sample_names=np.char.decode(f["sample_names"][:], "utf-8"),
                plate_names=np.char.decode(f["plate_names"][:], "utf-8"),
                control_treatment_name=f.attrs["control_treatment_name"],
            )


def filter_dataset_to_unique_treatments(screen: Screen | ScreenSubset):
    """
    Ensure that the dataset only has one experiment per treatment and sample condition
    by arbitrarily dropping duplicates.

    :param screen: an :py:class:`batchie.data.ScreenSubset`
    :return: A :py:class:`batchie.data.ScreenSubset` with the same or
        smaller number of experiments compared to the input.
    """
    arrs = [screen.sample_ids]

    for i in range(screen.treatment_arity):
        arrs.append(screen.treatment_ids[:, i])

    mask = select_unique_zipped_numpy_arrays(arrs)

    return screen.subset(mask)


def filter_dataset_to_treatments_that_appear_in_at_least_one_combo(
    screen: Screen,
) -> Screen:
    """
    Utility function to filter down an :py:class:`batchie.data.Screen` to only
    the treatments that appear in at least one combo.

    :param screen: an :py:class:`batchie.data.Screen`
    :return: A filtered :py:class:`batchie.data.Screen`
    """
    if screen.treatment_arity < 2:
        raise ValueError("Dataset must have at least 2 treatments")

    treatment_ids: ArrayType = screen.treatment_ids

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
        "Selecting {} treatments that appear in at least one combo of arity {}".format(
            treatments_to_select.size, screen.treatment_arity
        )
    )

    filtered_treatment_names = np.unique(
        screen.treatment_names[~treatment_selection_vector].flatten()
    )

    logger.info(
        "Filtering {} treatments that do not appear in any combo: ({})".format(
            len(filtered_treatment_names), ", ".join(filtered_treatment_names)
        )
    )

    screen_selection_vector = np.all(
        np.in1d(treatment_ids, treatments_to_select_plus_controls).reshape(
            treatment_ids.shape
        ),
        axis=1,
    )

    return screen.subset(screen_selection_vector).to_screen()
