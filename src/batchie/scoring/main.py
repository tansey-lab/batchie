from typing import Optional

import numpy as np
import logging
import h5py

from batchie.common import FloatingPointType
from batchie.core import Scorer, PlatePolicy, BayesianModel, ThetaHolder, ScoresHolder
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.data import Screen, Plate

logger = logging.getLogger(__name__)


class ChunkedScoresHolder(ScoresHolder):
    def __init__(self, size: int):
        self.size = size
        self.scores = np.zeros(size, dtype=FloatingPointType)
        self.plate_ids = np.zeros(size, dtype=int)
        self.current_index = 0

    def add_score(self, plate_id: int, score: FloatingPointType):
        self.scores[self.current_index] = score
        self.plate_ids[self.current_index] = plate_id
        self.current_index += 1

    def get_score(self, plate_id: int) -> FloatingPointType:
        return self.scores[self.plate_ids == plate_id][0]

    def plate_id_with_minimum_score(self) -> int:
        return self.plate_ids[self.scores.argmin()]

    def combine(self, other: ScoresHolder):
        self.scores = np.concatenate((self.scores, other.scores))
        self.plate_ids = np.concatenate((self.plate_ids, other.plate_ids))
        self.current_index += len(other.scores)
        return self

    @classmethod
    def concat(cls, scores_list: list[ScoresHolder]):
        if not scores_list:
            raise ValueError(
                "Must provide at least one ChunkedScoresHolder to concatenate"
            )

        current = scores_list[0]
        for scores in scores_list[1:]:
            current = current.combine(scores)
        return current

    def save_h5(self, fn):
        with h5py.File(fn, "w") as f:
            f.create_dataset("scores", data=self.scores)
            f.create_dataset("plate_ids", data=self.plate_ids)
            f.attrs["current_index"] = self.current_index

    @classmethod
    def load_h5(cls, fn):
        with h5py.File(fn, "r") as f:
            scores = f["scores"][:]
            plate_ids = f["plate_ids"][:]
            current_index = f.attrs["current_index"]
        scores_holder = cls(len(scores))
        scores_holder.scores = scores
        scores_holder.plate_ids = plate_ids
        scores_holder.current_index = current_index
        return scores_holder


def select_next_plate(
    scores: ScoresHolder,
    screen: Screen,
    policy: Optional[PlatePolicy],
    rng: Optional[np.random.Generator] = None,
) -> Optional[Plate]:
    """
    Select the next :py:class:`batchie.data.Plate` to observe

    :param scores: The scores for each plate
    :param screen: The screen which defines the set of plates to choose from
    :param policy: The policy to use for plate selection
    :param rng: PRNG to use for sampling
    :return: A list of plates to observe
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_plates = [plate for plate in screen.plates if plate.is_observed]

    unobserved_plates = [plate for plate in screen.plates if not plate.is_observed]

    unobserved_plates = sorted(unobserved_plates, key=lambda p: p.plate_id)

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
        return

    best_plate_id = scores.plate_id_with_minimum_score()
    best_plate = screen.get_plate(best_plate_id)
    best_plate_name = best_plate.plate_name

    logger.info(
        "Best plate: {} (id: {}). Size: {}".format(
            best_plate_name, best_plate_id, best_plate.size
        )
    )

    return best_plate


def score_chunk(
    model: BayesianModel,
    scorer: Scorer,
    samples: ThetaHolder,
    screen: Screen,
    distance_matrix: ChunkedDistanceMatrix,
    rng: Optional[np.random.Generator] = None,
    progress_bar: bool = False,
    n_chunks: int = 1,
    chunk_index: int = 0,
) -> ChunkedScoresHolder:
    if rng is None:
        rng = np.random.default_rng()

    unobserved_plates = [plate for plate in screen.plates if not plate.is_observed]

    unobserved_plates = sorted(unobserved_plates, key=lambda p: p.plate_id)

    chunk_plates = np.array_split(unobserved_plates, n_chunks)[chunk_index].tolist()

    logger.info(
        f"Scoring chunk {chunk_index+1} of {n_chunks}, with {len(chunk_plates)} plates"
    )

    scores_holder = ChunkedScoresHolder(len(chunk_plates))

    scores: dict[int, float] = scorer.score(
        plates=chunk_plates,
        model=model,
        samples=samples,
        rng=rng,
        distance_matrix=distance_matrix,
        progress_bar=progress_bar,
    )

    for k, v in scores.items():
        scores_holder.add_score(k, v)

    return scores_holder
