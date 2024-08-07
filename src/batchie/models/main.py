import math
from itertools import combinations

import h5py
import numpy as np
import pandas

from batchie.common import FloatingPointType, ArrayType
from batchie.core import (
    ThetaHolder,
)
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

        self._predictions = predictions
        self._observations = observations
        self._chain_ids = chain_ids
        self._sample_names = sample_names

    def mse(self):
        """
        Overall mean squared error
        """
        return ((self.predictions - self.observations[:, None]) ** 2).mean()

    def mse_variance(self):
        """
        Calculate variance of mean squared error
        """
        return np.var(
            ((self.predictions - self.observations[:, None]) ** 2).mean(axis=1)
        )

    def inter_chain_mse_variance(self):
        """
        Calculate variance of mean square error between chains
        """
        mses = []

        for chain_id in np.unique(self.chain_ids):

            selection_vector = self.chain_ids == chain_id
            chain_mse = (
                (self.predictions[:, selection_vector] - self.observations[:, None])
                ** 2
            ).mean()
            mses.append(chain_mse)

        return np.var(np.array(mses))

    @property
    def mean_predictions(self):
        return self.predictions.mean(axis=1)

    @property
    def predictions(self) -> ArrayType:
        """
        The predictions made by the model, shape (n_experiments, n_thetas)
        """
        return self._predictions

    @property
    def observations(self) -> ArrayType:
        """
        Observed data, shape (n_experiments,)
        """
        return self._observations

    @property
    def chain_ids(self) -> ArrayType:
        """
        The chain id for each theta, shape (n_thetas,)
        """
        return self._chain_ids

    @property
    def sample_names(self) -> ArrayType:
        """
        The sample name for each prediction, shape (n_experiments,)
        """
        return self._sample_names

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


def predict_viability_all(screen: ScreenBase, thetas: ThetaHolder):
    """
    Predict the experiment data using the model parameterized with each theta in thetas.

    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_samples, n_experiments) containing the
             viability predictions for each model / experiment combination
    """
    result = np.zeros((thetas.n_thetas, screen.size), dtype=FloatingPointType)
    for theta_index in range(thetas.n_thetas):
        theta = thetas.get_theta(theta_index)
        result[theta_index, :] = theta.predict_viability(screen)

        if np.isnan(result[theta_index, :]).any():
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

    return result


def predict_mean_all(screen: ScreenBase, thetas: ThetaHolder):
    """
    Predict the experiment data using the model parameterized with each theta in thetas.

    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_samples, n_experiments) containing the
             mean predictions for each model / experiment combination
    """
    result = np.zeros((thetas.n_thetas, screen.size), dtype=FloatingPointType)
    for theta_index in range(thetas.n_thetas):
        theta = thetas.get_theta(theta_index)
        result[theta_index, :] = theta.predict_conditional_mean(screen)

        if np.isnan(result[theta_index, :]).any():
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

    return result


def predict_variance_all(screen: ScreenBase, thetas: ThetaHolder):
    """
    Get the variance for the experiment data using the model parameterized with each theta in thetas.

    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_thetas, n_experiments) containing the
             prediction variances for each model / experiment combination.
    """

    results = []
    for theta_index in range(thetas.n_thetas):
        theta = thetas.get_theta(theta_index)
        result = theta.predict_conditional_variance(screen)

        if not result.size == screen.size:
            raise ValueError(
                "The variance array returned by the model is not the same size as the screen"
            )

        results.append(result)
        if np.any(np.isnan(result)):
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

    result = np.stack(results, dtype=FloatingPointType)
    return result


def predict_mean_avg(screen: ScreenBase, thetas: ThetaHolder):
    """
    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_experiments,) containing the
             mean predictions for each experiment combination
    """
    result = np.zeros((screen.size,), dtype=FloatingPointType)

    for theta_index in range(thetas.n_thetas):
        theta = thetas.get_theta(theta_index)

        sub_result = theta.predict_conditional_mean(screen)

        if np.isnan(sub_result).any():
            raise ValueError(
                "NaN predictions were created, please check screen and theta values"
            )

        result = result + sub_result

    return result / thetas.n_thetas


def predict_viability_avg(screen: ScreenBase, thetas: ThetaHolder):
    """
    :param model: The model to use for prediction
    :param screen: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_experiments,) containing the
             viability predictions for each experiment combination
    """
    result = np.zeros((screen.size,), dtype=FloatingPointType)

    for theta_index in range(thetas.n_thetas):
        theta = thetas.get_theta(theta_index)

        sub_result = theta.predict_viability(screen)

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

    if combination_count(screen.treatment_space_size, screen.treatment_arity) > 1e7:
        raise ValueError("The treatment space is too large for this method")

    all_treatments = zip(screen.treatment_mapping[0], screen.treatment_mapping[1])

    combos = np.array(
        list(combinations(all_treatments, screen.treatment_arity)), dtype=object
    )

    treatment_names = combos[:, :, 0]
    treatment_doses = combos[:, :, 1]

    plate_names = np.array(["1"] * combos.shape[0])

    sample_name = dict(zip(screen.sample_mapping[1], screen.sample_mapping[0]))[
        sample_id
    ]

    sample_ids = np.array([sample_name] * combos.shape[0])

    return Screen(
        treatment_names=treatment_names.astype(str),
        sample_names=sample_ids.astype(str),
        treatment_doses=treatment_doses.astype(FloatingPointType),
        plate_names=plate_names,
        sample_mapping=screen.sample_mapping,
        treatment_mapping=screen.treatment_mapping,
    )


def correlation_matrix(screen: ScreenBase, thetas: ThetaHolder):
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

        predictions.append(predict_viability_avg(combinatoric_space, thetas))
        index.append(id_to_name[sample_id])

    predictions = np.stack(predictions)

    mu = np.mean(predictions, axis=0, keepdims=True)
    X = predictions - mu
    X_ = X / np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
    corr = np.einsum("ik, jk->ij", X_, X_)

    return pandas.DataFrame(corr, index=index, columns=index)
