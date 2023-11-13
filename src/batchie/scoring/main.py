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
    screen: Screen,
    distance_matrix: ChunkedDistanceMatrix,
    policy: Optional[PlatePolicy],
    batch_size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> list[Plate]:
    """
    Select the next batch of :py:class:`batchie.data.Plate`s to observe

    :param model: The model to use for prediction
    :param scorer: The scorer to use for plate scoring
    :param samples: The set of model parameters to use for prediction
    :param screen: The screen which defines the set of plates to choose from
    :param distance_matrix: The distance matrix between model parameterizations
    :param policy: The policy to use for plate selection
    :param batch_size: The number of plates to select
    :param rng: PRNG to use for sampling
    :return: A list of plates to observe
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_plates = [plate for plate in screen.plates if plate.is_observed]

    unobserved_plates = [plate for plate in screen.plates if not plate.is_observed]

    selected_plates = []

    for i in range(batch_size):
        logger.info(f"Selecting plate {i+1} of {batch_size}")

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
