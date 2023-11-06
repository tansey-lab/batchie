from typing import Optional

import numpy as np
import logging
from batchie.core import (
    Scorer,
    PlatePolicy,
    BayesianModel,
    ThetaHolder,
)
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.data import Screen, Plate


logger = logging.getLogger(__name__)


def select_next_batch(
    model: BayesianModel,
    scorer: Scorer,
    samples: ThetaHolder,
    experiment_space: Screen,
    distance_matrix: ChunkedDistanceMatrix,
    policy: Optional[PlatePolicy],
    batch_size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> list[Plate]:
    """
    Select the next batch of plates to observe, these are the :py:class:`Plate` objects t
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_plates = [plate for plate in experiment_space.plates if plate.is_observed]

    unobserved_plates = [
        plate for plate in experiment_space.plates if not plate.is_observed
    ]

    selected_plates = []

    for i in range(batch_size):
        if policy is None:
            eligible_plates = unobserved_plates
        else:
            eligible_plates = policy.filter_eligible_plates(
                observed_plates=observed_plates,
                unobserved_plates=unobserved_plates,
                rng=rng,
            )

        if not eligible_plates:
            logger.warning("No eligible plates remaining, exiting early.")
            break

        scores: dict[int, float] = scorer.score(
            plates=eligible_plates,
            model=model,
            samples=samples,
            rng=rng,
            distance_matrix=distance_matrix,
        )

        best_plate_id = max(scores, key=scores.get)

        # move best plate from unobserved to selected
        selected_plates.append(
            unobserved_plates.pop(
                next(
                    i
                    for i, plate in enumerate(unobserved_plates)
                    if plate.plate_id == best_plate_id
                )
            )
        )

    return selected_plates
