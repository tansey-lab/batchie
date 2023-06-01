import numpy as np
from typing import Union
import pickle

from batchie.common import ArrayType


class Plate:
    def __init__(self, **kwargs):
        return

    def combine(self, plate_list: list["Plate"], **kwargs) -> "Plate":
        raise NotImplementedError


class Dataset:
    def __init__(self, **kwargs):
        return

    def available_plates(self, **kwargs) -> dict[str, Plate]:
        raise NotImplementedError

    def reveal_plate(self, plate_name: str, remove: bool = True, **kwargs) -> ArrayType:
        raise NotImplementedError


class Predictor:
    def __init__(self, **kwargs):
        return

    def predict(self, plate: Plate, **kwargs) -> ArrayType:
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


class PredictorHolder:
    def __init__(self, dataset: Dataset, pred_list: list[Predictor] = None, **kwargs):
        if pred_list is None:
            self.pred_list = []
        else:
            self.pred_list = pred_list

    def predict_plates(self, plates: dict) -> ArrayType:
        npreds = len(self.pred_list)
        assert npreds > 0, "Need at least one predictor to make predictions"
        p_predictions = []
        for pred in self.pred_list:
            p_predictions.append(pred.predict_plates(plates))

        nplates, dim = p_predictions[0].shape

        predictions = np.zeros((npreds, nplates, dim))
        for idx, p in enumerate(p_predictions):
            predictions[idx, :, :] = p
        return predictions

    def clear_predictors(self):
        self.pred_list = []

    def add_predictor(self, predictor: Union[Predictor, list[Predictor]]):
        if isinstance(predictor, list):
            for pred in predictor:
                self.pred_list.append(pred)
        else:
            self.pred_list.append(predictor)

    def predictor_list(self) -> list[Predictor]:
        return self.pred_list

    def save(self, fname: str):
        with open(fname, "wb") as io:
            pickle.dump(self.pred_list, io, pickle.HIGHEST_PROTOCOL)

    def load(self, fname: str):
        with open(fname, "rb") as io:
            self.pred_list = pickle.load(io)


def load_predictor_holder(dataset: Dataset, **kwargs):
    pred_holder = PredictorHolder(dataset)
    return pred_holder


class BayesianModel:
    def reset_model(self):
        raise NotImplementedError

    def predictor(self) -> Predictor:
        raise NotImplementedError

    def mcmc_step(self):
        raise NotImplementedError

    def update(self, plate: Plate, y: ArrayType, **kwargs):
        raise NotImplementedError

    def update_list(self, plates: list[Plate], ys: list[ArrayType]):
        for plate, y in zip(plates, ys):
            self.update(plate, y)
