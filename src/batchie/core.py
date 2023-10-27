from abc import ABC, abstractmethod

import json
import numpy as np
from batchie.common import ArrayType
from batchie.data import ExperimentBase, Plate, Experiment


class Theta:
    """
    This class represents parameters for a BayesianModel.
    Should be implemented by dataclass or similar.
    """

    pass


class ThetaHolder(ABC):
    """
    This class represents multiple parameter sets for a BayesianModel.
    """

    def __init__(self, n_thetas: int, *args, **kwargs):
        self._cursor = 0
        self._n_thetas = n_thetas

    @abstractmethod
    def _save_theta(self, theta: Theta, variance: float, sample_index: int):
        raise NotImplementedError

    @abstractmethod
    def get_theta(self, step_index: int) -> Theta:
        raise NotImplementedError

    @abstractmethod
    def get_variance(self, step_index: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def save_h5(self, fn: str):
        raise NotImplementedError

    @abstractmethod
    def combine(self, other):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_h5(path: str):
        raise NotImplementedError

    def add_theta(self, theta: Theta, variance: float):
        # test if we are at the end of the chain
        if self._cursor >= self.n_thetas:
            raise ValueError("Cannot add more samples to the results object")

        self._save_theta(theta, variance, self._cursor)
        self._cursor += 1

    @property
    def n_thetas(self):
        return self._n_thetas

    def __iter__(self):
        for i in range(self.n_thetas):
            yield self.get_theta(i)

    @property
    def is_complete(self):
        return self._cursor == self.n_thetas

    @classmethod
    def concat(cls, instances: list):
        """
        Combine multiple instances of SamplesHolder into one.
        """
        if len(instances) == 0:
            raise ValueError("Cannot concatenate an empty list of SamplesHolders")
        if len(instances) == 1:
            return instances[0]

        first = instances[0]

        for instance in instances[1:]:
            if type(instance) != type(first):
                raise ValueError("Cannot concatenate different types of SamplesHolders")

            first = first.combine(instance)

        return first


class BayesianModel(ABC):
    """
    This class represents a Bayesian model.

    A Bayesian model has internal state. Each BayesianModel should have a companion Theta and ThetaHolder
    class which represents the models internal state in a serializable way.

    The internal state of the model can be set explicitly via BayesianModel#set_model_state.
    or it can be advanced via BayesianModel#step.

    A BayesianModel can have data added to it via BayesianModel#add_observations.
    If data is present, the model should use that data somehow when BayesianModel#mcmc_step is called.
    BayesianModel#n_obs should report the number of datapoints that have been added to the model.

    A BayesianModel can be used to predict an outcome via BayesianModel#predict.
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
    def set_model_state(self, parameters: Theta):
        raise NotImplementedError

    @abstractmethod
    def get_model_state(self) -> Theta:
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
    def get_results_holder(self, n_samples: int) -> ThetaHolder:
        """
        Return a SamplesHolder class that goes with this model
        """
        raise NotImplementedError


class Metric:
    def __init__(self, model: BayesianModel):
        self.model = model

    def evaluate(self, sample: Theta) -> float:
        raise NotImplementedError

    def evaluate_all(self, results_holder: ThetaHolder) -> ArrayType:
        return np.array([self.evaluate(x) for x in results_holder])


class DistanceMetric:
    """
    This class represents a symmetric distance metric between two arrays
    of model predictions.
    """

    def distance(self, a: ArrayType, b: ArrayType) -> float:
        raise NotImplementedError


class DistanceMatrix(ABC):
    @abstractmethod
    def add_value(self, i, j, value):
        raise NotImplementedError

    @abstractmethod
    def to_dense(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filename):
        raise NotImplementedError


class Scorer:
    """
    This class represents a scoring function for Plates.
    """

    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: DistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        raise NotImplementedError


class PlatePolicy:
    """
    Given an Experiment, which is a set of potential Plates,
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


class ExperimentTracker:
    """
    This class tracks the state of an active learning experiment.
    """

    def __init__(
        self, plate_ids_selected: list[list[int]], losses: list[float], seed: int
    ):
        self.plate_ids_selected = plate_ids_selected
        self.losses = losses
        self.seed = seed

    def save(self, fn):
        # write to json
        with open(fn, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, fn):
        with open(fn, "r") as f:
            data = json.load(f)
        return cls(**data)


class InitialRetrospectivePlateGenerator:
    def generate_and_unmask_initial_plate(
        self, experiment: Experiment, rng: np.random.BitGenerator
    ) -> Experiment:
        raise NotImplementedError


class RetrospectivePlateGenerator:
    def generate_plates(
        self, experiment: Experiment, rng: np.random.BitGenerator
    ) -> Experiment:
        raise NotImplementedError
