from batchie.data import Dataset


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

    def predict(self, data: Dataset):
        raise NotImplementedError

    def mcmc_step(self):
        raise NotImplementedError

    def add_observations(self, data: Dataset):
        raise NotImplementedError

    def n_obs(self) -> int:
        raise NotImplementedError


class ResultsHolder:
    def add_mcmc_sample(self, sample: BayesianModelSample):
        raise NotImplementedError

    def get_mcmc_sample(self, step_index: int) -> BayesianModelSample:
        raise NotImplementedError

    def save_h5(self, fn: str):
        raise NotImplementedError

    @staticmethod
    def load_h5(path: str):
        raise NotImplementedError
