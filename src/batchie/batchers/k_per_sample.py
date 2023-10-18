from batchie.core import Batcher
import numpy as np
from typing import Set
from batchie.data import Experiment, ExperimentSubset
from collections import defaultdict


class KPerSampleBatcher(Batcher):
    def __init__(self, k):
        self.k = k

    def next_batch(
        self,
        selected_plates: Set[int],
        experiment: Experiment,
        rng: np.random.Generator,
    ) -> list[ExperimentSubset]:
        for plate_id in experiment.unique_plate_ids:
            if len(experiment.get_plate(plate_id).unique_sample_ids) != 1:
                raise ValueError(
                    "KPerSampleBatcher only works if all plates in the experiment contain exactly one sample"
                )

        remaining_plates = sorted(
            list(set(experiment.unique_plate_ids).difference(selected_plates))
        )

        # For each cell line, count how many plates remain
        n_plates_per_sample = defaultdict(int)
        for plate_id in remaining_plates:
            plate = experiment.get_plate(plate_id)
            sample_id = plate.sample_ids[0]
            n_plates_per_sample[sample_id] += 1

        # If we are choosing a new cell line, must have at least K plates remaining
        sample_ids_with_insufficient_plates = set()
        for sample_id, v in n_plates_per_sample.items():
            if v < self.k:
                sample_ids_with_insufficient_plates.add(sample_id)

        # Calculate how many times each cell line has appeared in an already selected plate
        n_plates_already_selected_per_sample = defaultdict(int)
        for plate_id in selected_plates:
            plate = experiment.get_plate(plate_id)
            sample_id = plate.sample_ids[0]
            n_plates_already_selected_per_sample[sample_id] += 1

        # If there is a cell line that has been chosen at least once but less than K, we restrict to it
        sample_chosen = None
        for sample_id, v in n_plates_already_selected_per_sample.items():
            if v < self.k:
                sample_chosen = sample_id

        result = []
        if sample_chosen is not None:
            for plate_id in remaining_plates:
                if plate_id in selected_plates:
                    raise RuntimeError(
                        "Plate from remaining plates set should not appear in the selected plates set"
                    )
                plate = experiment.get_plate(plate_id)
                sample_id = plate.sample_ids[0]
                if sample_id == sample_chosen:
                    result.append(plate)
        else:
            ## If all pending cell lines have been selected K times, then we exclude all such plates
            ## We also exclude plates with insufficient remaining plates
            for plate_id in remaining_plates:
                if plate_id in selected_plates:
                    raise RuntimeError(
                        "Plate from remaining plates set should not appear in the selected plates set"
                    )
                plate = experiment.get_plate(plate_id)
                sample_id = plate.sample_ids[0]

                sample_id_has_not_yet_been_selected = (
                    sample_id not in n_plates_already_selected_per_sample
                )

                if (
                    sample_id not in sample_ids_with_insufficient_plates
                ) and sample_id_has_not_yet_been_selected:
                    result.append(plate)

        return result
