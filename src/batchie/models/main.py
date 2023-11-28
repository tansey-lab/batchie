import h5py
import numpy as np

from batchie.common import FloatingPointType, ArrayType
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import ScreenBase


class ModelEvaluation:
    def __init__(
        self, predictions: ArrayType, observations: ArrayType, chain_ids: ArrayType
    ):
        if not np.issubdtype(predictions.dtype, FloatingPointType):
            raise ValueError("predictions must be floats")

        if not np.issubdtype(observations.dtype, FloatingPointType):
            raise ValueError("observations must be floats")

        if not np.issubdtype(chain_ids.dtype, int):
            raise ValueError("chain_ids must be ints")

        if not predictions.shape[0] == observations.shape[0]:
            raise ValueError(
                "predictions and observations must have the same number of samples"
            )

        if not len(predictions.shape) == 2:
            raise ValueError(
                "Predictions must be a matrix (one prediction per theta per experiment)"
            )

        if not chain_ids.shape[0] == predictions.shape[1]:
            raise ValueError("chain_ids must have one entry per theta")

        self.predictions = predictions
        self.observations = observations
        self.chain_ids = chain_ids

    def save_h5(self, fn):
        with h5py.File(fn, "w") as f:
            f.create_dataset("predictions", data=self.predictions, compression="gzip")
            f.create_dataset("observations", data=self.observations, compression="gzip")
            f.create_dataset("chain_ids", data=self.chain_ids, compression="gzip")

    @classmethod
    def load_h5(cls, fn):
        with h5py.File(fn, "r") as f:
            predictions = f["predictions"][:]
            observations = f["observations"][:]
            chain_ids = f["chain_ids"][:]
            return cls(
                predictions=predictions, observations=observations, chain_ids=chain_ids
            )


def predict_all(model: BayesianModel, screen: ScreenBase, thetas: ThetaHolder):
    """
    Predict the experiment data using the model parameterized with each theta in thetas.

    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_samples, n_experiments) containing the
             predictions for each model / experiment combination
    """
    result = np.zeros((thetas.n_thetas, screen.size), dtype=FloatingPointType)
    for theta_index in range(thetas.n_thetas):
        model.set_model_state(thetas.get_theta(theta_index))
        result[theta_index, :] = model.predict(screen)
    return result


def predict_avg(model: BayesianModel, screen: ScreenBase, thetas: ThetaHolder):
    """
    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_experiments,) containing the
             mean predictions for each experiment combination
    """
    result = np.zeros((screen.size,), dtype=FloatingPointType)

    for theta_index in range(thetas.n_thetas):
        model.set_model_state(thetas.get_theta(theta_index))
        result = result + model.predict(screen)

    return result / thetas.n_thetas
