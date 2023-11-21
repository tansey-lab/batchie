import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    ThetaHolder,
    Plate,
)
from batchie.distance_calculation import ChunkedDistanceMatrix


class RandomScorer(Scorer):

    """
    A scorer that returns a random score for each plate, used for baseline comparison
    """

    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
        progress_bar: bool,
    ) -> dict[int, float]:
        scores = {plate.plate_id: rng.random() for plate in plates}
        return scores
