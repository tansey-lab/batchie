import numpy as np

from batchie.data import Data
from batchie.core import Scorer


class RandomScorer(Scorer):
    def _score(self, data: Data, rng: np.random.Generator, **kwargs):
        return rng.random()
