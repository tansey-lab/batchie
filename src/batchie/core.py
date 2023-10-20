from abc import ABC, abstractmethod

import h5py
import numpy as np
from batchie.common import ArrayType
from batchie.data import ExperimentBase, Plate


class BayesianModelParams:
    """
    This class represents a snapshot of all random variables
    in the model. Should be implemented by dataclass or similar.
    """

    pass


class ModelParamsHolder:
    def __init__(self, n_samples: int, *args, **kwargs):
        self._cursor = 0
        self.n_samples = n_samples

    def _save_sample(
        self, sample: BayesianModelParams, variance: float, sample_index: int
    ):
        raise NotImplementedError

    def get_sample(self, step_index: int) -> BayesianModelParams:
        raise NotImplementedError

    def get_variance(self, step_index: int) -> float:
        raise NotImplementedError

    def save_h5(self, fn: str):
        raise NotImplementedError

    @staticmethod
    def load_h5(path: str):
        raise NotImplementedError

    def add_sample(self, sample: BayesianModelParams, variance: float):
        # test if we are at the end of the chain
        if self._cursor >= self.n_samples:
            raise ValueError("Cannot add more samples to the results object")

        self._save_sample(sample, variance, self._cursor)
        self._cursor += 1

    def __iter__(self):
        for i in range(self.n_samples):
            yield self.get_sample(i)

    @property
    def is_complete(self):
        return self._cursor == self.n_samples

    @classmethod
    def concat(cls, instances: list):
        """
        Combine multiple instances of SamplesHolder into one.
        """
        if len(instances) == 0:
            raise ValueError("Cannot concatenate an empty list of SamplesHolders")
        if len(instances) == 1:
            return instances[0]

        n_samples = sum([x.n_samples for x in instances])
        combined = cls(n_samples)
        for instance in instances:
            for sample_index, sample in enumerate(instance):
                combined.add_sample(
                    sample,
                    instance.get_variance(sample_index),
                )
        return combined


class BayesianModel(ABC):
    """
    This class represents a Bayesian model.

    A Bayesian model has internal state. Each BayesianModel should have a companion BayesianModelSample
    class which represents the models internal state in a serializable way,

    The internal state of the model can be set explicitly via BayesianModel#set_model_state
    or it can be advanced via BayesianModel#mcmc_step.

    A BayesianModel can have data added to it via BayesianModel#add_observations.
    If data is present, the model should use that data somehow when BayesianModel#mcmc_step is called.
    BayesianModel#n_obs should report the number of datapoints that have been added to the model.

    A BayesianModel can be used to predict an outcome given a set of inputs via BayesianModel#predict.
    """

    def __init__(
        self,
        n_unique_treatments: int,
        n_unique_samples: int,
    ):
        self.n_unique_treatments = n_unique_treatments
        self.n_unique_samples = n_unique_samples

    @abstractmethod
    def reset_model(self):
        raise NotImplementedError

    @abstractmethod
    def set_model_state(self, parameters: BayesianModelParams):
        raise NotImplementedError

    @abstractmethod
    def get_model_state(self) -> BayesianModelParams:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: ExperimentBase) -> ArrayType:
        raise NotImplementedError

    @abstractmethod
    def variance(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def set_rng(self, rng: np.random.Generator):
        raise NotImplementedError

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        raise NotImplementedError

    @abstractmethod
    def add_observations(self, data: ExperimentBase):
        raise NotImplementedError

    @abstractmethod
    def n_obs(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_results_holder(self, n_samples: int) -> ModelParamsHolder:
        """
        Return a SamplesHolder class that goes with this model
        """
        raise NotImplementedError


class Metric:
    def __init__(self, model: BayesianModel):
        self.model = model

    def evaluate(self, sample: BayesianModelParams) -> float:
        raise NotImplementedError

    def evaluate_all(self, results_holder: ModelParamsHolder) -> ArrayType:
        return np.array([self.evaluate(x) for x in results_holder])


class DistanceMetric:
    def distance(self, a: ArrayType, b: ArrayType) -> float:
        raise NotImplementedError


class DistanceMatrix:
    """
    Class which can represent part or a whole pairwise distance matrix.

    The distance matrix is stored in a sparse format, but can be converted to a dense format if all values
    are present.

    Several partial DistanceMatrix classes can be combined. This is useful for parallelization of the distance
    matrix computation.
    """

    def __init__(self, size, chunk_size=100):
        self.size = size
        self.chunk_size = chunk_size
        self.current_index = 0
        self.row_indices = np.zeros(chunk_size, dtype=int)
        self.col_indices = np.zeros(chunk_size, dtype=int)
        self.values = np.zeros(chunk_size, dtype=float)

    def _expand_storage(self):
        self.row_indices = np.concatenate(
            (self.row_indices, np.zeros(self.chunk_size, dtype=int))
        )
        self.col_indices = np.concatenate(
            (self.col_indices, np.zeros(self.chunk_size, dtype=int))
        )
        self.values = np.concatenate(
            (self.values, np.zeros(self.chunk_size, dtype=float))
        )

    def add_value(self, i, j, value):
        if i >= self.size or j >= self.size:
            raise ValueError("Indices are out of bounds")
        if self.current_index + 1 > len(self.values):
            self._expand_storage()

        self.row_indices[self.current_index] = i
        self.col_indices[self.current_index] = j
        self.values[self.current_index] = value
        self.current_index += 1

    def is_complete(self):
        return self.current_index == (self.size * self.size)

    def to_dense(self):
        if not self.is_complete():
            raise ValueError("The distance matrix is not complete")
        dense = np.zeros((self.size, self.size))
        for i in range(self.current_index):
            dense[self.row_indices[i], self.col_indices[i]] = self.values[i]
        return dense

    def save(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("row_indices", data=self.row_indices[: self.current_index])
            f.create_dataset("col_indices", data=self.col_indices[: self.current_index])
            f.create_dataset("values", data=self.values[: self.current_index])
            f.create_dataset("size", data=np.array([self.size]))

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, "r") as f:
            row_indices = f["row_indices"][:]
            col_indices = f["col_indices"][:]
            values = f["values"][:]
            size = f["size"][0]
            instance = cls(
                size, chunk_size=len(values)
            )  # assuming all values are filled, otherwise adjust chunk_size accordingly
            instance.row_indices[: len(values)] = row_indices
            instance.col_indices[: len(values)] = col_indices
            instance.values[: len(values)] = values
            instance.current_index = len(values)
        return instance

    def combine(self, other):
        if self.size != other.size:
            raise ValueError(
                "The matrices must be of the same size to be composed together"
            )

        composed = DistanceMatrix(self.size, self.chunk_size)
        composed.row_indices[: self.current_index] = self.row_indices[
            : self.current_index
        ]
        composed.col_indices[: self.current_index] = self.col_indices[
            : self.current_index
        ]
        composed.values[: self.current_index] = self.values[: self.current_index]
        composed.current_index = self.current_index

        for i in range(other.current_index):
            row, col, value = (
                other.row_indices[i],
                other.col_indices[i],
                other.values[i],
            )
            if (row, col) not in zip(
                composed.row_indices[: composed.current_index],
                composed.col_indices[: composed.current_index],
            ):
                composed.add_value(row, col, value)

        return composed

    @classmethod
    def concat(cls, matrices: list):
        if len(matrices) == 1:
            return matrices[0]
        elif len(matrices) == 0:
            raise ValueError("Cannot concat empty list of matrices")

        accumulator = matrices[0]
        for matrix in matrices[1:]:
            if accumulator.size != matrix.size:
                raise ValueError("Cannot concat matrices of different sizes")
            accumulator = accumulator.combine(matrix)
        return accumulator


class Scorer:
    """
    This class represents a scoring function for plates, which are potential sets of experiments.
    """

    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: DistanceMatrix,
        samples: ModelParamsHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        raise NotImplementedError


class PlatePolicy:
    """
    Given an Experiment, which is a set of potential "plates" or ExperimentSubsets,
    implementations of this class will determine which set of plates is eligible
    for the next round.
    """

    def filter_eligible_plates(
        self,
        observed_plates: list[Plate],
        unobserved_plates: list[Plate],
        rng: np.random.Generator,
    ) -> list[Plate]:
        raise NotImplementedError
