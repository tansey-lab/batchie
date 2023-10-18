from batchie.core import DistanceMetric
from scipy.special import expit
from batchie.common import ArrayType
import numpy as np


class MSEDistance(DistanceMetric):
    def __init__(self, sigmoid: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.sigmoid = sigmoid

    def distance(self, a: ArrayType, b: ArrayType):
        if self.sigmoid:
            a = expit(a)
            b = expit(b)
        return np.mean((a - b) ** 2)