from batchie.data import Data, Dataset
import numpy as np
from batchie.common import ArrayType


class BayesianModelSample:
    """
    This class represents a snapshot of all random variables
    in the model. Should be implemented by dataclass or similar.
    """

    pass


class BayesianModel:
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

    def reset_model(self):
        raise NotImplementedError

    def set_model_state(self, parameters: BayesianModelSample):
        raise NotImplementedError

    def get_model_state(self) -> BayesianModelSample:
        raise NotImplementedError

    def predict(self, data: Data) -> ArrayType:
        raise NotImplementedError

    def mcmc_step(self):
        raise NotImplementedError

    def add_observations(self, data: Data):
        raise NotImplementedError

    def n_obs(self) -> int:
        raise NotImplementedError


class ResultsHolder:
    def __init__(self, n_mcmc_steps: int, *args, **kwargs):
        self._cursor = 0
        self.n_mcmc_steps = n_mcmc_steps

    def add_mcmc_sample(self, sample: BayesianModelSample):
        # test if we are at the end of the chain
        if self._cursor >= self.n_mcmc_steps:
            raise ValueError("Cannot add more samples to the results object")

        self._save_mcmc_sample(sample)

    def _save_mcmc_sample(self, sample: BayesianModelSample):
        raise NotImplementedError

    def get_mcmc_sample(self, step_index: int) -> BayesianModelSample:
        raise NotImplementedError

    def save_h5(self, fn: str):
        raise NotImplementedError

    def __iter__(self):
        for i in range(self.n_mcmc_steps):
            yield self.get_mcmc_sample(i)

    @staticmethod
    def load_h5(path: str):
        raise NotImplementedError

    @property
    def is_complete(self):
        return self._cursor == self.n_mcmc_steps


class Metric:
    def __init__(self, model: BayesianModel):
        self.model = model

    def evaluate(self, sample: BayesianModelSample) -> float:
        raise NotImplementedError

    def evaluate_all(self, results_holder: ResultsHolder) -> ArrayType:
        return np.array([self.evaluate(x) for x in results_holder])


class DistanceMetric:
    def __init__(self, model: BayesianModel):
        self.model = model

    def distance(self, a: BayesianModelSample, b: BayesianModelSample) -> float:
        raise NotImplementedError

    def square_form(self, samples: list[BayesianModelSample]) -> ArrayType:
        dists = np.zeros((len(samples), len(samples)), dtype=np.float32)

        for idx1, m1 in enumerate(samples):
            for idx2, m2 in enumerate(samples[idx1 + 1 :]):
                d = self.distance(m1, m2)
                dists[idx1, idx2] = d
                dists[idx2, idx1] = d
        return dists

    def one_v_rest(
        self, sample: BayesianModelSample, others: list[BayesianModelSample]
    ) -> ArrayType:
        dists = np.zeros(len(others), dtype=np.float32)
        for idx, other in enumerate(others):
            dists[idx] = self.distance(sample, other)
        return dists


class Scorer:
    """
    This class represents a scoring function for plates, which are potential sets of experiments.
    """

    def _score(
        self,
        data: Data,
        model: BayesianModel,
        results: ResultsHolder,
        rng: np.random.Generator,
    ):
        raise NotImplementedError

    def score(
        self,
        dataset: Dataset,
        model: BayesianModel,
        results: ResultsHolder,
        rng: np.random.Generator,
    ) -> dict:
        result = {}
        for plate_id, plate in dataset.plates.items():
            result[plate_id] = self._score(
                data=plate, rng=rng, model=model, results=results
            )
        return result
