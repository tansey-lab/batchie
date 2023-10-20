import numpy as np
from batchie.core import BayesianModel, SamplesHolder
from batchie.data import ExperimentBase
from batchie.common import FloatingPointType


def predict_all(
    model: BayesianModel, experiment: ExperimentBase, samples: SamplesHolder
):
    """
    :param model: The model to use for prediction
    :param experiment: The data to predict
    :param samples: The samples to use for prediction
    :return: A matrix of shape (n_samples, n_experiments) containing the
             predictions for each model / experiment combination
    """
    result = np.zeros((samples.n_samples, experiment.size), dtype=FloatingPointType)

    for theta_index in range(samples.n_samples):
        model.set_model_state(samples.get_sample(theta_index))
        result[theta_index, :] = model.predict(experiment)
    return result


def predict_avg(
    model: BayesianModel, experiment: ExperimentBase, samples: SamplesHolder
):
    """
    :param model: The model to use for prediction
    :param experiment: The data to predict
    :param samples: The samples to use for prediction
    :return: A matrix of shape (n_experiments,) containing the
             mean predictions for each experiment combination
    """
    result = np.zeros((experiment.size,), dtype=FloatingPointType)

    for theta_index in range(samples.n_samples):
        model.set_model_state(samples.get_sample(theta_index))
        result = result + model.predict(experiment)

    return result / samples.n_samples
