import numpy as np

from batchie.data import Data
from batchie.core import Scorer


class SizeScorer(Scorer):
    def _score(self, data: Data, **kwargs):
        return data.n_experiments
