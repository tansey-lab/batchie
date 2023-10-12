from batchie.data import Data


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

    def predict(self, data: Data):
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

    @staticmethod
    def load_h5(path: str):
        raise NotImplementedError

    @property
    def is_complete(self):
        return self._cursor == self.n_mcmc_steps
