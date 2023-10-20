import numpy as np

from batchie.core import (
    Scorer,
    BayesianModel,
    DistanceMatrix,
    SamplesHolder,
    ExperimentSubset,
)


class RandomScorer(Scorer):
    def score(
        self,
        model: BayesianModel,
        plates: list[ExperimentSubset],
        distance_matrix: DistanceMatrix,
        samples: SamplesHolder,
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = {plate.plate_id: rng.random() for plate in plates}
        return scores
