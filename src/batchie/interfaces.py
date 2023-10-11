import pickle
from typing import Union

import numpy as np

from batchie.common import ArrayType
from batchie.data import Dataset


class Predictor:
    def __init__(self, **kwargs):
        return

    def predict(self, data: Dataset, **kwargs) -> ArrayType:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def predict_plates(self, plates: dict[str, Plate], **kwargs) -> ArrayType:
        nplates = len(plates)
        preds = [self.predict(plate) for plate in plates.values()]
        max_len = max([len(x) for x in preds])
        predictions = np.zeros((nplates, max_len))
        for idx, pred in enumerate(preds):
            predictions[idx, : len(pred)] = pred
        return predictions


class BayesianModel:
    def reset_model(self):
        raise NotImplementedError

    def predictor(self) -> Predictor:
        raise NotImplementedError

    def mcmc_step(self):
        raise NotImplementedError

    def add_observations(self, data: Dataset):
        raise NotImplementedError


class ResultsHolder:
    def add_mcmc_sample(self, sample):
        raise NotImplementedError

    def get_mcmc_sample(self, step_index: int):
        raise NotImplementedError

    def save_h5(self, fn: str):
        raise NotImplementedError

    @staticmethod
    def load_h5(path: str):
        raise NotImplementedError
