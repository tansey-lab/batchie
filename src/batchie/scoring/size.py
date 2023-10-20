import numpy as np

from batchie.data import ExperimentBase
from batchie.core import Scorer


class SizeScorer(Scorer):
    def _score(self, data: ExperimentBase, **kwargs):
        return data.size
