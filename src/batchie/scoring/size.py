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
        plates: list[Plate],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = {plate.plate_id: plate.size for plate in plates}
        return scores
