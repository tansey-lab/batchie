import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    DistanceMatrix,
    ThetaHolder,
    Plate,
)


class SizeScorer(Scorer):
    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: DistanceMatrix,
        samples: ThetaHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = {plate.plate_id: plate.size for plate in plates}
        return scores
