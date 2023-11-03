import numpy as np
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import ExperimentBase
from batchie.common import FloatingPointType


def predict_all(model: BayesianModel, experiment: ExperimentBase, thetas: ThetaHolder):
    """
    :param model: The model to use for prediction
    :param experiment: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_samples, n_experiments) containing the
             predictions for each model / experiment combination
    """
    result = np.zeros((thetas.n_thetas, experiment.size), dtype=FloatingPointType)
    for theta_index in range(thetas.n_thetas):
        model.set_model_state(thetas.get_theta(theta_index))
        result[theta_index, :] = model.predict(experiment)
    return result


def predict_avg(model: BayesianModel, experiment: ExperimentBase, thetas: ThetaHolder):
    """
    :param model: The model to use for prediction
    :param experiment: The data to predict
    :param thetas: The set of model parameters to use for prediction
    :return: A matrix of shape (n_experiments,) containing the
             mean predictions for each experiment combination
    """
    result = np.zeros((experiment.size,), dtype=FloatingPointType)

    for theta_index in range(thetas.n_thetas):
        model.set_model_state(thetas.get_theta(theta_index))
        result = result + model.predict(experiment)

    return result / thetas.n_thetas
