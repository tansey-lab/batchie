import numpy as np

from batchie.data import Data
from batchie.interfaces import Scorer


class SizeScorer(Scorer):
    def _score(self, data: Data, rng: np.random.Generator):
        return data.n_experiments
