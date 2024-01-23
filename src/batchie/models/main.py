import h5py
import numpy as np
import math

from itertools import combinations

import pandas

from batchie.common import FloatingPointType, ArrayType
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import ScreenBase, Screen


class ModelEvaluation:
    def __init__(
        self,
        predictions: ArrayType,
        observations: ArrayType,
        chain_ids: ArrayType,
        sample_names: ArrayType,
    ):
        if not np.issubdtype(predictions.dtype, FloatingPointType):
            raise ValueError("predictions must be floats")

        if not np.issubdtype(observations.dtype, FloatingPointType):
            raise ValueError("observations must be floats")

        if not np.issubdtype(chain_ids.dtype, int):
            raise ValueError("chain_ids must be ints")

        if not np.issubdtype(sample_names.dtype, str):
            raise ValueError("sample_names must be str")

        if not predictions.shape[0] == observations.shape[0]:
            raise ValueError(
                "predictions and observations must have the same number of samples"
            )

        if not sample_names.shape[0] == observations.shape[0]:
            raise ValueError("sample_names and observations must be the same size")

        if not len(predictions.shape) == 2:
            raise ValueError(
                "Predictions must be a matrix (one prediction per theta per experiment)"
            )

        if not chain_ids.shape[0] == predictions.shape[1]:
            raise ValueError("chain_ids must have one entry per theta")

        self.predictions = predictions
        self.observations = observations
        self.chain_ids = chain_ids
        self.sample_names = sample_names

    def rmse(self):
        """
        Calculate root mean square error
        """
        return ((self.predictions - self.observations) ** 2).mean()

    @property
    def mean_predictions(self):
        return self.predictions.mean(axis=1)

    def save_h5(self, fn):
        with h5py.File(fn, "w") as f:
            f.create_dataset("predictions", data=self.predictions, compression="gzip")
            f.create_dataset("observations", data=self.observations, compression="gzip")
            f.create_dataset("chain_ids", data=self.chain_ids, compression="gzip")
            f.create_dataset(
                "sample_names",
                data=np.char.encode(self.sample_names),
                compression="gzip",
            )

    @classmethod
    def load_h5(cls, fn):
        with h5py.File(fn, "r") as f:
            predictions = f["predictions"][:]
            observations = f["observations"][:]
            chain_ids = f["chain_ids"][:]
            sample_names = np.char.decode(f["sample_names"][:], "utf-8")
            return cls(
                predictions=predictions,
                observations=observations,
                chain_ids=chain_ids,
                sample_names=sample_names,
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

        if np.isnan(result[theta_index, :]).any():
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

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

        sub_result = model.predict(screen)

        if np.isnan(sub_result).any():
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

        result = result + sub_result

    return result / thetas.n_thetas


def combination_count(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def generate_full_combinatoric_space(sample_id: int, screen: ScreenBase):
    """
    Generate an artificial screen object that represents all combinations of
    treatments
    """

    if combination_count(screen.n_unique_treatments, screen.treatment_arity) > 1e7:
        raise ValueError("The treatment space is too large for this method")

    combos = combinations(screen.unique_treatments, screen.treatment_arity)

    treatment_ids = np.array(list(combos))

    sample_ids = np.array([sample_id] * treatment_ids.shape[0])
    plate_names = np.array(["1"] * treatment_ids.shape[0])

    return Screen(
        treatment_names=treatment_ids.astype(str),
        sample_names=sample_ids.astype(str),
        treatment_doses=np.ones_like(treatment_ids).astype(FloatingPointType),
        plate_names=plate_names,
        treatment_ids=treatment_ids,
        sample_ids=sample_ids,
    )


def correlation_matrix(model: BayesianModel, screen: ScreenBase, thetas: ThetaHolder):
    """
    Predict over the entire space of treatment combinations for each unique sample in
    screen and create a correlation matrix between samples based on how similar they are
    in that prediction space.


    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_samples, n_samples) containing the
             correlation in predictions between each samples
    """
    predictions = []
    index = []
    id_to_name = dict(zip(screen.sample_ids, screen.sample_names))

    for sample_id in screen.unique_sample_ids:
        combinatoric_space = generate_full_combinatoric_space(sample_id, screen)

        predictions.append(predict_avg(model, combinatoric_space, thetas))
        index.append(id_to_name[sample_id])

    predictions = np.stack(predictions)

    mu = np.mean(predictions, axis=0, keepdims=True)
    X = predictions - mu
    X_ = X / np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
    corr = np.einsum("ik, jk->ij", X_, X_)

    return pandas.DataFrame(corr, index=index, columns=index)
