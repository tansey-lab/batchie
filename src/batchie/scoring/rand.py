import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    DistanceMatrix,
    ModelParamsHolder,
    Plate,
)


class RandomScorer(Scorer):
    def score(
        self,
        model: BayesianModel,
        plates: list[Plate],
        distance_matrix: DistanceMatrix,
        samples: ModelParamsHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = {plate.plate_id: rng.random() for plate in plates}
        return scores
