import json
import logging
from abc import ABC, abstractmethod
import numpy as np

from batchie.common import ArrayType, FloatingPointType
from batchie.data import ScreenBase, Plate, Screen, ScreenSubset

logger = logging.getLogger(__name__)


class Theta:
    """
    This class represents the set of parameters for a BayesianModel.
    Should be implemented by a dataclass or similarly serializable class.
    """

    pass


class ThetaHolder(ABC):
    """
    This class represents a container for multiple parameter sets for a BayesianModel.
    This class provides methods to save these parameter sets to an H5 file.
    """

    def __init__(self, n_thetas: int, *args, **kwargs):
        self._cursor = 0
        self._n_thetas = n_thetas

    @abstractmethod
    def _save_theta(self, theta: Theta, sample_index: int):
        raise NotImplementedError

    @abstractmethod
    def get_theta(self, step_index: int) -> Theta:
        """
        Returns the parameter set at the given index.

        :param step_index: The index of the parameter set to return.
        """
        raise NotImplementedError

    @abstractmethod
    def save_h5(self, fn: str):
        """
        Save the parameter sets to an H5 file.

        :param fn: The filename to save to.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, other):
        """
        Combine these parameters sets with another container of parameter sets.

        :param other: Another ThetaHolder instance.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_h5(path: str):
        """
        Load a ThetaHolder from an H5 file.

        :param path: The path to the H5 file.
        """
        raise NotImplementedError

    def add_theta(self, theta: Theta):
        """
        Add a new parameter set to the container.

        :param theta: The parameter set to add.
        """
        # test if we are at the end of the chain
        if self._cursor >= self.n_thetas:
            raise ValueError("Cannot add more samples to the results object")

        self._save_theta(theta, self._cursor)
        self._cursor += 1

    @property
    def n_thetas(self):
        """
        :return: The number of parameter sets in the container.
        """
        return self._n_thetas

    def __iter__(self):
        for i in range(self.n_thetas):
            yield self.get_theta(i)

    @property
    def is_complete(self):
        """
        :return: True if the container is full, False otherwise.
        """
        return self._cursor == self.n_thetas

    @classmethod
    def concat(cls, instances: list):
        """
        Combine multiple instances of SamplesHolder into one.

        :param instances: A list of ThetaHolder instances.
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

    A Bayesian model has internal state. Each :py:class:`batchie.core.BayesianModel` should have a companion
    :py:class:`batchie.core.Theta` and :py:class:`batchie.core.ThetaHolder`
    class which represents the models internal state in a serializable way.

    The internal state of the model can be set explicitly via :py:meth:`batchie.core.BayesianModel.set_model_state`.
    or it can be advanced via :py:meth:`batchie.core.BayesianModel.step`.

    A :py:class:`batchie.core.BayesianModel` can have data added to it via
    :py:meth:`batchie.core.BayesianModel.add_observations`.
    If data is present, the model should use that data somehow when BayesianModel#step is called.
    :py:meth:`batchie.core.BayesianModel.n_obs` should report the number of
    datapoints that have been added to the model.

    A :py:class:`batchie.core.BayesianModel` can be used to predict the outcome of an Experiment
    via :py:meth:`batchie.core.BayesianModel.predict`.

    A :py:class:`batchie.core.BayesianModel` must report its variance via
    :py:meth:`batchie.core.BayesianModel.variance`.
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
        """
        Reset the internal state of the model to its initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_model_state(self, parameters: Theta):
        """
        Set the internal state of the model to the given parameter set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_state(self) -> Theta:
        """
        Get the internal state of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: ScreenBase) -> ArrayType:
        """
        Predict the outcome of an :py:class:`batchie.data.ExperimentBase`.

        :return: An array of predictions for each item in the Experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def set_rng(self, rng: np.random.Generator):
        """
        Set the PRNG for this model instance.

        :param rng: The PRNG to use.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """
        Return the PRNG for this model instance.

        :return: The PRNG for this model instance.
        """
        raise NotImplementedError

    def add_observations(self, data: ScreenBase):
        """
        Add observations to the model.

        :param data: The data to add.
        """
        if not data.observation_mask.all():
            raise ValueError("Cannot add data with masked observations")

        self._add_observations(data)

    @abstractmethod
    def _add_observations(self, data: ScreenBase):
        """
        Add observations to the model.

        :param data: The data to add.
        """
        raise NotImplementedError

    @abstractmethod
    def n_obs(self) -> int:
        """
        Return the number of observations that have been added to the model.

        :return: Integer number of observations
        """
        raise NotImplementedError

    @abstractmethod
    def get_results_holder(self, n_samples: int) -> ThetaHolder:
        """
        Construct a :py:meth:`batchie.core.ThetaHolder` instance that is compatible
        with this mode and has capacity n_samples.

        :param n_samples: The number of samples to allocate space for.
        """
        raise NotImplementedError


class MCMCModel:
    """
    This class subclasses BayesianModel and implements :py:meth:`batchie.core.MCMCModel.step`
    """

    @abstractmethod
    def step(self):
        """
        Advance the internal state of the model by one step.

        In the case of an MCMC model, this would mean taking one more MCMC step. Other types
        of models should implement accordingly.

        """
        raise NotImplementedError


class VIModel:
    """
    This class subclasses BayesianModel and implements :py:meth:`batchie.core.VIModel.sample`
    """

    @abstractmethod
    def sample(self, num_samples: int) -> list[Theta]:
        """
        Returns a list of Theta samples. Length of the list should be num_samples.

        """
        raise NotImplementedError


class HomoscedasticModel:
    @abstractmethod
    def variance(self, data: ScreenBase) -> FloatingPointType:
        """
        Return the variance of the model.

        :return: A single floating point number representing the variance of this models predictions.
        """
        raise NotImplementedError


class HeteroscedasticModel:
    @abstractmethod
    def variance(self, data: ScreenBase) -> ArrayType:
        """
        Return the variance of the model.

        :return: An array containing the variance of the prediction for each experiment in the screen.
        """
        raise NotImplementedError


class Metric:
    def __init__(self, model: BayesianModel):
        self.model = model

    def evaluate(self, sample: Theta) -> FloatingPointType:
        """
        Evaluate the metric on a single parameter set.

        :param sample: The parameter set to evaluate.
        :return: The value of the metric.
        """
        raise NotImplementedError

    def evaluate_all(self, results_holder: ThetaHolder) -> ArrayType:
        """
        Evaluate the metric on all parameter sets in the results_holder.

        :param results_holder: The parameter sets to evaluate.
        :return: An array of metric values.
        """
        return np.array([self.evaluate(x) for x in results_holder])


class DistanceMetric:
    """
    This class represents a symmetric distance metric between two arrays
    of model predictions.
    """

    def distance(self, a: ArrayType, b: ArrayType) -> float:
        """
        Calculate the distance between two arrays of model predictions.

        :param a: The first array of model predictions.
        :param b: The second array of model predictions.
        :return: The distance between the two arrays.
        """
        raise NotImplementedError


class DistanceMatrix(ABC):
    @abstractmethod
    def add_value(self, i, j, value):
        """
        Add a value to the distance matrix.

        :param i: The row index.
        :param j: The column index.
        :param value: The value to add.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dense(self):
        """
        Return a dense representation of the distance matrix.

        :return: A dense representation of the distance matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        """
        Save the distance matrix to a file.

        :param filename: The filename to save to.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filename):
        """
        Load a distance matrix from a file.

        :param filename: The filename to load from.
        """
        raise NotImplementedError


class Scorer:
    """
    This class represents a scoring function for :py:class:`batchie.data.Plate`
    instances.

    The score should represent how desirable it is to observe the given plate, with
    a lower score being more desirable.
    """

    def score(
        self,
        model: BayesianModel,
        plates: dict[int, ScreenSubset],
        distance_matrix: DistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
        progress_bar: bool,
    ) -> dict[int, FloatingPointType]:
        raise NotImplementedError


class PlatePolicy:
    """
    Given a :py:class:`batchie.data.Screen`, which is a set of potential :py:class:`batchie.data.Plate`s,
    implementations of this class will determine which set of :py:class:`batchie.data.Plate`s is eligible
    for the next round.
    """

    def filter_eligible_plates(
        self,
        batch_plates: list[Plate],
        unobserved_plates: list[Plate],
        rng: np.random.Generator,
    ) -> list[Plate]:
        raise NotImplementedError


class SimulationTracker:
    """
    This class tracks the state of a retrospective active learning simulation.
    It will record the plates that were revealed at each step and the total loss of the predictor
    trained on the plates revealed up until that point.
    """

    def __init__(
        self, plate_ids_selected: list[list[int]], losses: list[float], seed: int
    ):
        """
        :param plate_ids_selected: A list of lists of plate ids that were revealed at each step.
        :param losses: A list of losses of the predictor trained on the plates revealed up until each step.
        :param seed: The seed used to generate the simulation.
        """
        self.plate_ids_selected = plate_ids_selected
        self.losses = losses
        self.seed = seed

    def save(self, fn):
        """
        Save this instance to a JSON file.

        :param fn: The filename to save to.
        """
        # write to json
        with open(fn, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, fn):
        """
        Load this instance from a JSON file.

        :param fn: The filename to load from.
        """
        with open(fn, "r") as f:
            data = json.load(f)
        return cls(**data)


class InitialRetrospectivePlateGenerator(ABC):
    """
    When running a retrospective active learning simulation, results are sensitive to the initial
    plate which is revealed. For this reason users might want to implement a special routine for revealing
    the initial plate separate from the subsequent plates.
    """

    def generate_and_unmask_initial_plate(
        self, screen: Screen, rng: np.random.BitGenerator
    ) -> Screen:
        """
        Generate and unmask the initial plate.

        :param screen: A fully observed :py:class:`batchie.data.Screen`
        :param rng: The PRNG to use.
        :return: The same :py:class:`batchie.data.Screen` with the initial plate observed, and all other plates
        masked.
        """
        if not screen.is_observed:
            raise ValueError(
                "The experiment used for retrospective analysis must be fully observed"
            )

        return self._generate_and_unmask_initial_plate(screen, rng)

    @abstractmethod
    def _generate_and_unmask_initial_plate(
        self, screen: Screen, rng: np.random.BitGenerator
    ):
        raise NotImplementedError


class RetrospectivePlateGenerator(ABC):
    """
    When running a retrospective active learning simulation, the user might want to reorganize the dataset
    into different plates then were originally run. This class will generate these plate groupings from the individual
    observations in the retrospective dataset.
    """

    def generate_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
        """
        Generate plates from the remaining unobserved experiments in the input screen.

        :param screen: A partially observed :py:class:`batchie.data.Screen`
        :param rng: The PRNG to use.
        """
        unobserved_subset = screen.subset_unobserved()
        observed_subset = screen.subset_observed()

        if unobserved_subset is None:
            logger.warning("No unobserved data found, returning original experiment")
            return screen

        new_unobserved_subset = self._generate_plates(
            unobserved_subset.to_screen(), rng
        )

        if observed_subset is None:
            return new_unobserved_subset
        else:
            combined_screen = new_unobserved_subset.combine(observed_subset.to_screen())
            return combined_screen

    @abstractmethod
    def _generate_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
        raise NotImplementedError


class RetrospectivePlateSmoother(ABC):
    """
    After plates have been generated for a retrospective simulation using a
    :py:class:`batchie.core.RetrospectivePlateGenerator`,
    those plates may be of very uneven sizes, which is not desirable. Implementations of this class
    should aim to merge plates together and/or drop experiments until plate sizes are more even. We call
    this process "plate smoothing".
    """

    def smooth_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
        """
        Smooth the plates in the screen.

        :param screen: A partially observed :py:class:`batchie.data.Screen`
        :param rng: The PRNG to use.
        """
        unobserved_subset = screen.subset_unobserved()
        observed_subset = screen.subset_observed()

        if unobserved_subset is None:
            logger.warning("No unobserved data found, returning original experiment")
            return screen

        new_unobserved_subset = self._smooth_plates(unobserved_subset.to_screen(), rng)

        if observed_subset is None:
            return new_unobserved_subset
        else:
            return new_unobserved_subset.combine(observed_subset.to_screen())

    @abstractmethod
    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
        raise NotImplementedError


class ScoresHolder(ABC):
    """
    This class represents a set of scores for a set of plates.
    """

    def get_score(self, plate_id: int) -> float:
        """
        Get the score for a given plate.

        :param plate_id: The plate id to get the score for.
        :return: The score for the given plate.
        """
        raise NotImplementedError

    def add_score(self, plate_id: int, score: float):
        """
        Add a score for a given plate.

        :param plate_id: The plate id to add the score for.
        :param score: The score to add.
        """
        raise NotImplementedError

    def plate_id_with_minimum_score(self, eligible_plate_ids: list[int] = None) -> int:
        """
        Get the plate id with the minimum score.

        :param eligible_plate_ids: The set of plates to consider.
        :return: The plate id with the minimum score.
        """
        raise NotImplementedError
