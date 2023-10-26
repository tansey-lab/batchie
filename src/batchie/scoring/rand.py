import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    ThetaHolder,
    Plate,
)
from batchie.distance_calculation import ChunkedDistanceMatrix


class RandomScorer(Scorer):
    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: ChunkedDistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = {plate.plate_id: rng.random() for plate in plates}
        return scores
