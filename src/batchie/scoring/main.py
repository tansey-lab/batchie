import logging
from typing import Optional

import h5py
import numpy as np

from batchie.common import FloatingPointType
from batchie.core import Scorer, PlatePolicy, BayesianModel, ThetaHolder, ScoresHolder
from batchie.data import (
    Screen,
    Plate,
    ScreenSubset,
    filter_dataset_to_unique_treatments,
)
from batchie.distance_calculation import ChunkedDistanceMatrix

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
        return self.scores[self.plate_ids == plate_id].item()

    def plate_id_with_minimum_score(self, eligible_plate_ids: list[int] = None) -> int:
        if eligible_plate_ids is None:
            return self.plate_ids[self.scores.argmin()].item()

        mask = np.isin(self.plate_ids, eligible_plate_ids)
        return self.plate_ids[mask][self.scores[mask].argmin()].item()

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
    batch_plate_ids: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Plate]:
    """
    Select the next :py:class:`batchie.data.Plate` to observe

    :param scores: The scores for each plate
    :param screen: The screen which defines the set of plates to choose from
    :param policy: The policy to use for plate selection
    :param batch_plate_ids: The plates currently selected in the batch
    :param rng: PRNG to use for sampling
    :return: A list of plates to observe
    """
    if rng is None:
        rng = np.random.default_rng()

    if batch_plate_ids is None:
        batch_plate_ids = []

    batch_plates = [
        plate for plate in screen.plates if plate.plate_id in batch_plate_ids
    ]

    unobserved_plates_not_already_selected = [
        plate
        for plate in screen.plates
        if not plate.is_observed and plate.plate_id not in batch_plate_ids
    ]

    unobserved_plates_not_already_selected = sorted(
        unobserved_plates_not_already_selected, key=lambda p: p.plate_id
    )

    if policy is None:
        eligible_plates = unobserved_plates_not_already_selected
    else:
        eligible_plates = policy.filter_eligible_plates(
            batch_plates=batch_plates,
            unobserved_plates=unobserved_plates_not_already_selected,
            rng=rng,
        )

    if not eligible_plates:
        logger.warning("No eligible plates remaining, exiting early.")
        return

    eligible_plate_ids = [plate.plate_id for plate in eligible_plates]

    best_plate_id = scores.plate_id_with_minimum_score(eligible_plate_ids)
    best_plate = screen.get_plate(best_plate_id)
    best_plate_name = best_plate.plate_name

    logger.info(
        "Best plate: {} (id: {}). Size: {}".format(
            best_plate_name, best_plate_id, best_plate.size
        )
    )

    return best_plate


def score_chunk(
    scorer: Scorer,
    thetas: ThetaHolder,
    screen: Screen,
    distance_matrix: ChunkedDistanceMatrix,
    rng: Optional[np.random.Generator] = None,
    progress_bar: bool = False,
    n_chunks: int = 1,
    chunk_index: int = 0,
    batch_plate_ids: Optional[list[int]] = None,
) -> ChunkedScoresHolder:
    """
    Score a subset of all unobserved plates in a screen.

    :param scorer: The scorer to use for scoring
    :param thetas: The samples to use for scoring
    :param screen: The screen to score
    :param distance_matrix: The distance matrix to use for scoring
    :param rng: PRNG to use for sampling
    :param progress_bar: Whether to show a progress bar
    :param n_chunks: The number of chunks to split the unobserved plates into
    :param chunk_index: The index of the chunk to score
    :param batch_plate_ids: A list of plate ids that have already been selected in the batch
    :return: ChunkedScoresHolder containing the scores for each plate in the current chunk
    """
    if rng is None:
        rng = np.random.default_rng()

    unobserved_plates = [plate for plate in screen.plates if not plate.is_observed]

    if batch_plate_ids is not None:
        unobserved_plates = [
            plate
            for plate in unobserved_plates
            if plate.plate_id not in batch_plate_ids
        ]

    unobserved_plates = sorted(unobserved_plates, key=lambda p: p.plate_id)
    chunk_plates = np.array_split(unobserved_plates, n_chunks)[chunk_index].tolist()

    if batch_plate_ids:
        previously_selected_plates = [
            plate for plate in screen.plates if plate.plate_id in batch_plate_ids
        ]

        previously_selected_plates_combined = ScreenSubset.concat(
            previously_selected_plates
        )

        plates_to_score = {}
        for plate in chunk_plates:
            conditioned_plate = filter_dataset_to_unique_treatments(
                plate.combine(previously_selected_plates_combined)
            )
            plates_to_score[plate.plate_id] = conditioned_plate
    else:
        plates_to_score = {p.plate_id: p for p in chunk_plates}

    logger.info(
        f"Scoring chunk {chunk_index+1} of {n_chunks}, with {len(plates_to_score)} non-excluded plates"
    )

    scores_holder = ChunkedScoresHolder(len(plates_to_score))

    scores: dict[int, float] = scorer.score(
        plates=plates_to_score,
        distance_matrix=distance_matrix,
        samples=thetas,
        rng=rng,
        progress_bar=progress_bar,
    )

    for k, v in scores.items():
        scores_holder.add_score(k, v)

    return scores_holder
