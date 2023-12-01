import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    ThetaHolder,
    Plate,
)
from batchie.distance_calculation import ChunkedDistanceMatrix


class SizeScorer(Scorer):
    """
    A scorer that returns the number of conditions in the :py:class:`Plate` as the score.
    """

    def score(
        self,
        model: BayesianModel,
        plates: dict[int, Plate],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
        progress_bar: bool,
    ) -> dict[int, float]:
        scores = {k: plate.size for k, plate in plates.items()}
        return scores
