from batchie.core import (
    Scorer,
    PlatePolicy,
    BayesianModel,
    SamplesHolder,
    DistanceMatrix,
)
from batchie.data import Experiment
from typing import Optional
import numpy as np


def select_next_batch(
    model: BayesianModel,
    scorer: Scorer,
    samples: SamplesHolder,
    experiment_space: Experiment,
    distance_matrix: DistanceMatrix,
    policy: Optional[PlatePolicy],
    batch_size: int = 1,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    observed_plates = [
        plate for plate_id, plate in experiment_space.plates if plate.is_observed
    ]

    unobserved_plates = [
        plate for plate_id, plate in experiment_space.plates if not plate.is_observed
    ]

    selected_unobserved_plates = []

    for i in range(batch_size):
        if policy is None:
            eligible_plates = unobserved_plates
        else:
            eligible_plates = policy.filter_eligible_plates(
                observed_plates=observed_plates,
                unobserved_plates=unobserved_plates,
                rng=rng,
            )

        scores = scorer.score(
            dataset=experiment_space,
            model=model,
            samples=samples,
            rng=rng,
            distance_matrix=distance_matrix,
        )
