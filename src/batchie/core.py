import json
import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from numbers import Number
import h5py
from batchie.common import ArrayType, FloatingPointType
from batchie.data import ScreenBase, Plate, Screen, ScreenSubset, ExperimentSpace
import importlib


logger = logging.getLogger(__name__)


class Theta:
    """
    This class represents the set of parameters for a BayesianModel.
    Should be implemented by a dataclass or similarly serializable class.
    """

    @abstractmethod
    def predict_viability(self, data: ScreenBase) -> ArrayType:
        """
        Predict the conditional mean of an :py:class:`batchie.data.ExperimentBase` in viability space.

        :return: An array of means for each item in the Experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_conditional_mean(self, data: ScreenBase) -> ArrayType:
        """
        Predict the conditional mean of an :py:class:`batchie.data.ExperimentBase` in modeling space.

        :return: An array of means for each item in the Experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_conditional_variance(self, data: ScreenBase) -> ArrayType:
        """
        Predict the conditional variance of an :py:class:`batchie.data.ExperimentBase`.

        :return: An array of variances for each item in the Experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def private_parameters_dict(self) -> dict[str, ArrayType]:
        """
        The private parameters of a :py:class:`batchie.core.Theta`.

        :return: a dictionary mapping class variables to arrays/numerical values.
        """
        raise NotImplementedError

    def shared_parameters_dict(self) -> dict[str, ArrayType]:
        """
        The shared parameters of a :py:class:`batchie.core.Theta`.

        :return: a dictionary mapping class variables to arrays.
        """
        return {}

    @classmethod
    @abstractmethod
    def from_dicts(cls, private_params: dict, shared_params: dict):
        """
        Instantiate :py:class:`batchie.core.Theta` from dictionary

        :return: a dictionary mapping class variables to arrays/numerical values.
        """
        raise NotImplementedError

    def equals(self, other):
        print("fsdsdfsd")
        if not isinstance(other, type(self)):
            return False

        for d1, d2 in [
            (self.private_parameters_dict(), other.private_parameters_dict()),
            (self.shared_parameters_dict(), other.shared_parameters_dict()),
        ]:
            for k, v in d1.items():
                if k not in d2:
                    return False
                if isinstance(v, Number):
                    if v != d2[k]:
                        return False
                elif isinstance(v, ArrayType):
                    if not np.array_equal(v, d2[k]):
                        return False
                else:
                    if v != d2[k]:
                        return False
        return True


class ThetaHolder(ABC):
    """
    This class represents a container for multiple parameter sets for a BayesianModel.
    This class provides methods to save these parameter sets to an H5 file.
    """

    def __init__(self, n_thetas: int, *args, **kwargs):
        self._n_thetas: int = n_thetas
        self.thetas: list[Theta] = []

    def get_theta(self, step_index: int) -> Theta:
        """
        Returns the parameter set at the given index.

        :param step_index: The index of the parameter set to return.
        """
        if (step_index > (len(self.thetas) - 1)) or (step_index < 0):
            raise ValueError("step_index out of bounds")

        return self.thetas[step_index]

    def combine(self, other):
        """
        Combine these parameters sets with another container of parameter sets.

        :param other: Another ThetaHolder instance.
        """
        if type(self) != type(other):
            raise ValueError("Cannot combine with different type")

        n_thetas = self.n_thetas + other.n_thetas
        result = ThetaHolder(n_thetas)
        result.thetas = self.thetas + other.thetas
        return result

    def save_h5(self, fn: str):
        """
        Save the parameter sets to an H5 file.

        :param fn: The filename to save to.
        """

        if len(self.thetas) == 0:
            raise ValueError("Cannot save an empty ThetaHolder")

        ## Only get shared parameters of first theta
        shared_params = self.thetas[0].shared_parameters_dict()

        ## Only get the class of first theta
        theta_class = self.thetas[0].__class__.__name__
        theta_module = self.thetas[0].__class__.__module__

        with h5py.File(fn, "w") as f:
            f.attrs.create("n_thetas", self.n_thetas)
            f.attrs.create("theta_class", theta_class)
            f.attrs.create("theta_module", theta_module)

            shared_grp = f.create_group("shared_params")
            for key, val in shared_params.items():
                if isinstance(val, ArrayType):
                    shared_grp.create_dataset(key, data=val, compression="gzip")
                else:
                    shared_grp.attrs.create(key, val)

            private_grp = f.create_group("private_params")
            for i, theta in enumerate(self.thetas):
                i_grp = private_grp.create_group(str(i))
                private_params = theta.private_parameters_dict()
                for key, val in private_params.items():
                    if isinstance(val, ArrayType):
                        i_grp.create_dataset(key, data=val, compression="gzip")
                    else:
                        i_grp.attrs.create(key, val)

    @staticmethod
    def load_h5(path: str):
        """
        Load a ThetaHolder from an H5 file.

        :param path: The path to the H5 file.
        """
        with h5py.File(path, "r") as f:
            n_thetas = f.attrs["n_thetas"]
            result = ThetaHolder(n_thetas=n_thetas)

            theta_class = f.attrs["theta_class"]
            theta_module = f.attrs["theta_module"]
            ThetaClass = getattr(importlib.import_module(theta_module), theta_class)

            ## Unpack shared parameters
            shared_params = {}
            shared_grp = f["shared_params"]
            shared_params.update(shared_grp.attrs.items())
            for key in shared_grp.keys():
                shared_params[key] = shared_grp[key][:]

            private_grp = f["private_params"]
            theta_keys = sorted(list(private_grp.keys()), key=int)
            for theta_key in theta_keys:
                i_grp = private_grp[theta_key]
                private_params = {}
                private_params.update(i_grp.attrs.items())

                for key in i_grp.keys():
                    private_params[key] = i_grp[key][:]

                theta = ThetaClass.from_dicts(
                    private_params=private_params, shared_params=shared_params
                )
                result.add_theta(theta)

        return result

    def add_theta(self, theta: Theta):
        """
        Add a new parameter set to the container.

        :param theta: The parameter set to add.
        """
        # test if we are at the end of the chain
        if len(self.thetas) >= self.n_thetas:
            raise ValueError("Cannot add more samples to the results object")

        self.thetas.append(theta)

    @property
    def n_thetas(self):
        """
        :return: The number of parameter sets in the container.
        """
        return self._n_thetas

    def __iter__(self):
        for theta in self.thetas:
            yield theta

    @property
    def is_complete(self):
        """
        :return: True if the container is full, False otherwise.
        """
        return len(self.thetas) == self.n_thetas

    @classmethod
    def concat(cls, instances: list):
        """
        Combine multiple instances of ThetaHolder into one.

        :param instances: A list of ThetaHolder instances.
        """
        if len(instances) == 0:
            raise ValueError("Cannot concatenate an empty list of ThetaHolder")
        if len(instances) == 1:
            return instances[0]

        first = instances[0]

        for instance in instances[1:]:
            if type(instance) != type(first):
                raise ValueError("Cannot concatenate different types of ThetaHolder")

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

    def __init__(self, experiment_space: ExperimentSpace):
        self.experiment_space = experiment_space

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
    def reset_model(self):
        """
        Reset the internal state of the model to its initial state.
        """
        raise NotImplementedError


class MCMCModel:
    """
    This class subclasses BayesianModel and implements :py:meth:`batchie.core.MCMCModel.step`
    """

    @abstractmethod
    def get_model_state(self) -> Theta:
        """
        Get the internal state of the model.
        """
        raise NotImplementedError

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
