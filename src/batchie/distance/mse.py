import numpy as np
from batchie.common import ArrayType
from batchie.core import DistanceMetric
from scipy.special import expit


class MSEDistance(DistanceMetric):
    def __init__(self, sigmoid: bool = True):
        self.sigmoid = sigmoid

    def distance(self, a: ArrayType, b: ArrayType):
        if self.sigmoid:
            a = expit(a)
            b = expit(b)
        return np.mean((a - b) ** 2)
