import heapq
import logging
from collections import defaultdict
from typing import Optional

import math
import numpy as np

from batchie.common import CONTROL_SENTINEL_VALUE
from batchie.core import (
    BayesianModel,
    ThetaHolder,
    InitialRetrospectivePlateGenerator,
    RetrospectivePlateGenerator,
    RetrospectivePlateSmoother,
)
from batchie.data import Screen, Plate
from batchie.models.main import predict_avg

logger = logging.getLogger(__name__)


class SparseCoverPlateGenerator(InitialRetrospectivePlateGenerator):
    def __init__(self, reveal_single_treatment_experiments: bool):
        self.reveal_single_treatment_experiments = reveal_single_treatment_experiments

    def _generate_and_unmask_initial_plate(
        self, screen: Screen, rng: np.random.BitGenerator
    ):
        """
        We want to make sure we have at least one observation for
        each cell line/drug dose combination in the first plate

        This is called a greedy cover algorithm

        We'll use this to construct the first plate, this initialization causes
        faster convergence of the algorithm.
        """
        covered_treatments = set()
        chosen_selection_indices = []

        for sample_id in screen.unique_sample_ids:
            experiments_with_at_least_one_treatment_not_in_covered_treatments = np.any(
                ~np.isin(screen.treatment_ids, list(covered_treatments)), axis=1
            )

            selection_vector = (
                screen.sample_ids == sample_id
            ) & experiments_with_at_least_one_treatment_not_in_covered_treatments
            if selection_vector.sum() > 0:
                selection_indices = np.arange(selection_vector.size)[selection_vector]
                chosen_selection_index = rng.choice(selection_indices, size=1)
                chosen_selection_indices.append(chosen_selection_index)

                covered_treatments.update(
                    set(screen.treatment_ids[chosen_selection_indices].flatten())
                )
            else:  ## randomly choose some index corresponding to c
                selection_vector = screen.sample_ids == sample_id
                selection_indices = np.arange(selection_vector.size)[selection_vector]
                chosen_selection_index = rng.choice(selection_indices, size=1)
                chosen_selection_indices.append(chosen_selection_index)
                covered_treatments.update(
                    set(screen.treatment_ids[chosen_selection_indices].flatten())
                )

        remaining_treatments = np.setdiff1d(
            screen.treatment_ids, list(covered_treatments)
        )

        while len(remaining_treatments) > 0:
            experiments_with_at_least_one_treatment_in_remaining_treatments = np.any(
                np.isin(screen.treatment_ids, list(remaining_treatments)), axis=1
            )
            selection_indices = np.arange(selection_vector.size)[
                experiments_with_at_least_one_treatment_in_remaining_treatments
            ]
            chosen_selection_index = rng.choice(selection_indices, 1)
            chosen_selection_indices.append(chosen_selection_index)
            covered_treatments.update(
                set(screen.treatment_ids[chosen_selection_indices].flatten())
            )
            remaining_treatments = np.setdiff1d(
                screen.treatment_ids, list(covered_treatments)
            )

        final_plate_selection_vector = np.isin(
            np.arange(screen.size), chosen_selection_indices
        )

        if self.reveal_single_treatment_experiments:
            single_drug_experiments = np.any(
                np.isin(screen.treatment_ids, [CONTROL_SENTINEL_VALUE]), axis=1
            )

            final_plate_selection_vector = (
                final_plate_selection_vector | single_drug_experiments
            )

        plate_names = np.array(["initial_plate"] * screen.size, dtype=str)

        plate_names[~final_plate_selection_vector] = "unobserved_plate"

        logger.info(
            "Created initial plate of size {}".format(
                final_plate_selection_vector.sum()
            )
        )

        observation_vector = screen.observations.copy()

        return Screen(
            treatment_names=screen.treatment_names,
            treatment_doses=screen.treatment_doses,
            observations=observation_vector,
            sample_names=screen.sample_names,
            plate_names=plate_names,
            control_treatment_name=screen.control_treatment_name,
            observation_mask=final_plate_selection_vector,
        )


class PairwisePlateGenerator(RetrospectivePlateGenerator):
    def __init__(self, subset_size: int, anchor_size: int):
        self.anchor_size = anchor_size
        self.subset_size = subset_size

    def _generate_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
        """
        Break up your drug doses into groups of size subset_size

        Each plate will contain all pairwise combinations of the two groups of drug doses
        restricted to a single cell line

        If you do this for GDCS you'll get weird plates because they have some "anchor drugs"
        that are used in almost every experiment. So you need to make sure that you have
        <anchor_size> anchor drugs in each group

        :param screen: The :py:class:`batchie.data.Screen` to generate plates for
        :param rng: The random number generator to use
        :return: A :py:class:`batchie.data.Screen` with the generated plates
        """
        combo_mask = ~np.any((screen.treatment_ids == CONTROL_SENTINEL_VALUE), axis=1)
        single_treatment_mask = ~combo_mask

        combo_treatment_screen = screen.subset(combo_mask).to_screen()

        if single_treatment_mask.any():
            single_treatment_screen = screen.subset(single_treatment_mask).to_screen()
        else:
            single_treatment_screen = None
        unique_treatments, unique_treatment_counts = np.unique(
            combo_treatment_screen.treatment_ids, return_counts=True
        )

        if self.anchor_size > 0:
            anchor_dds = unique_treatments[
                np.argsort(-unique_treatment_counts)[: self.anchor_size]
            ]
            n_anchor_groups = len(anchor_dds) // self.subset_size
            anchor_groups = np.array_split(rng.permutation(anchor_dds), n_anchor_groups)

            remain_dds = np.setdiff1d(unique_treatments, anchor_dds)
            n_remain_groups = len(remain_dds) // self.subset_size
            remain_groups = np.array_split(rng.permutation(remain_dds), n_remain_groups)
            groupings = anchor_groups + remain_groups
        else:
            n_groups = len(unique_treatments) // self.subset_size
            groupings = np.array_split(rng.permutation(unique_treatments), n_groups)

        num_groups = len(groupings)
        group_lookup = {}
        for group_id, treatment_ids_in_group in enumerate(groupings):
            for treatment_id_in_group in treatment_ids_in_group:
                group_lookup[treatment_id_in_group] = group_id
        group_lookup[CONTROL_SENTINEL_VALUE] = CONTROL_SENTINEL_VALUE

        treatment_group_ids = np.vectorize(group_lookup.get)(
            combo_treatment_screen.treatment_ids
        )

        # Assign the groups for control treatments
        n_control = np.sum(treatment_group_ids == CONTROL_SENTINEL_VALUE)
        treatment_group_ids[treatment_group_ids == CONTROL_SENTINEL_VALUE] = rng.choice(
            range(num_groups), size=n_control, replace=True
        )
        treatment_group_ids_sorted = np.sort(treatment_group_ids, axis=1)

        sample_id_col_vector = combo_treatment_screen.sample_ids[:, np.newaxis]

        grouping_tuples = np.hstack([sample_id_col_vector, treatment_group_ids_sorted])

        unique_grouping_tuples = np.unique(grouping_tuples, axis=0)

        new_plate_names = np.array([""] * combo_treatment_screen.size, dtype=object)

        for idx, unique_grouping_tuple in enumerate(unique_grouping_tuples):
            mask = (grouping_tuples == unique_grouping_tuple).all(axis=1)
            new_plate_names[mask] = f"generated_plate_{idx}"

        combo_screen_with_generated_plates = Screen(
            treatment_names=combo_treatment_screen.treatment_names,
            treatment_doses=combo_treatment_screen.treatment_doses,
            observations=combo_treatment_screen.observations,
            observation_mask=np.zeros(combo_treatment_screen.size, dtype=bool),
            sample_names=combo_treatment_screen.sample_names,
            plate_names=new_plate_names.astype(str),
            control_treatment_name=combo_treatment_screen.control_treatment_name,
        )

        if single_treatment_screen is None:
            return combo_screen_with_generated_plates

        new_plate_names = np.array([""] * single_treatment_screen.size, dtype=object)

        for sample_name in np.unique(single_treatment_screen.sample_names):
            n_to_assign = (single_treatment_screen.sample_names == sample_name).sum()

            eligible_plate_names = np.unique(
                combo_screen_with_generated_plates.plate_names[
                    combo_screen_with_generated_plates.sample_names == sample_name
                ]
            )
            if len(eligible_plate_names) == 0:
                raise ValueError(
                    "Single treatment experiments not corresponding to a treatment seen in any combo experiment"
                    "should be filtered before using this method"
                )

            assignments = rng.choice(
                eligible_plate_names, size=n_to_assign, replace=True
            )

            new_plate_names[
                single_treatment_screen.sample_names == sample_name
            ] = assignments

        single_screen_with_generated_plates = Screen(
            treatment_names=single_treatment_screen.treatment_names,
            treatment_doses=single_treatment_screen.treatment_doses,
            observations=single_treatment_screen.observations,
            observation_mask=np.zeros(single_treatment_screen.size, dtype=bool),
            sample_names=single_treatment_screen.sample_names,
            plate_names=new_plate_names.astype(str),
            control_treatment_name=single_treatment_screen.control_treatment_name,
        )

        return combo_screen_with_generated_plates.combine(
            single_screen_with_generated_plates
        )


class PlatePermutationPlateGenerator(RetrospectivePlateGenerator):
    """
    This generator will create new plates by permuting the plate labels.

    Plates can be excluded from permutation with the force_include_plate_names argument
    """

    def __init__(self, force_include_plate_names: Optional[list[str]] = None):
        self.force_include_plate_names = force_include_plate_names

    def _generate_plates(self, screen: Screen, rng: np.random.BitGenerator):
        if self.force_include_plate_names:
            selection_vector = ~np.isin(
                screen.plate_names, self.force_include_plate_names
            )
        else:
            selection_vector = np.ones(screen.size, dtype=bool)

        if np.any(~selection_vector):
            to_permute = screen.subset(selection_vector).to_screen()
            non_permuted = screen.subset(~selection_vector).to_screen()
        else:
            to_permute = screen.subset(selection_vector).to_screen()
            non_permuted = None

        new_plate_names = rng.permutation(to_permute.plate_names)

        permuted = Screen(
            treatment_names=to_permute.treatment_names,
            treatment_doses=to_permute.treatment_doses,
            observations=to_permute.observations,
            sample_names=to_permute.sample_names,
            plate_names=new_plate_names,
            control_treatment_name=to_permute.control_treatment_name,
            observation_mask=np.zeros(to_permute.size, dtype=bool),
        )

        if non_permuted is not None:
            return permuted.combine(non_permuted)
        else:
            return permuted


class SampleSegregatingPermutationPlateGenerator(RetrospectivePlateGenerator):
    """
    This generator will generate plates that only contain experiments for a single
    sample. If there are more than max_plate_size experiments for a single sample
    then the experiments will be split across multiple equal sized plates.
    """

    def __init__(self, max_plate_size: int):
        self.max_plate_size = max_plate_size

    def _generate_plates(self, screen: Screen, rng: np.random.BitGenerator):
        plate_indices = []
        for sample_id in screen.unique_sample_ids:
            sample_indices = np.arange(screen.size)[screen.sample_ids == sample_id]

            if len(sample_indices) > self.max_plate_size:
                n_plates = math.ceil(len(sample_indices) / float(self.max_plate_size))
                plates = np.array_split(rng.permutation(sample_indices), n_plates)
                for plate in plates:
                    plate_indices.append(plate)
        logger.info(
            "SampleSegregatingPermutationPlateGenerator created {} plates".format(
                len(plate_indices)
            )
        )

        plate_names = np.array([""] * screen.size, dtype=object)

        for idx, indices in enumerate(plate_indices):
            plate_names[indices] = f"generated_plate_{idx}"

        return Screen(
            treatment_names=screen.treatment_names.copy(),
            treatment_doses=screen.treatment_doses.copy(),
            observations=screen.observations.copy(),
            sample_names=screen.sample_names.copy(),
            plate_names=plate_names.astype(str),
            control_treatment_name=screen.control_treatment_name,
            observation_mask=screen.observation_mask.copy(),
        )


class MergeMinPlateSmoother(RetrospectivePlateSmoother):
    """
    Iteratively combine the smallest two plates for each sample until all plates are
    above a user specified size.
    """

    def __init__(self, min_size: int):
        self.min_size = min_size

    def _get_plate_sample_id(self, plate: Plate):
        if len(plate.unique_sample_ids) > 1:
            raise ValueError(
                "This method is only valid for one-sample-per-plate designs"
            )

        return plate.unique_sample_ids[0]

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        logger.info(
            "Applying MergeMinPlateSmoother with min_size={}".format(self.min_size)
        )
        current_screen = screen

        for sample_id in current_screen.unique_sample_ids:
            plate_heap = [
                p
                for p in current_screen.plates
                if self._get_plate_sample_id(p) == sample_id
            ]
            # create minheap of plate objects
            heapq.heapify(plate_heap)

            while True:
                if len(plate_heap) <= 1:
                    break

                smallest_plate = heapq.heappop(plate_heap)
                second_smallest_plate = heapq.heappop(plate_heap)

                if (smallest_plate.size + second_smallest_plate.size) > self.min_size:
                    break

                merged_plate = second_smallest_plate.merge(smallest_plate)

                heapq.heappush(plate_heap, merged_plate)

        return current_screen


class MergeTopBottomPlateSmoother(RetrospectivePlateSmoother):
    """
    Iteratively combine the largest and smallest plates for each sample.
    Runs for a user specified number of iterations.
    """

    def __init__(self, n_iterations: int):
        self.n_iterations = n_iterations

    def _get_plate_sample_id(self, plate: Plate):
        if len(plate.unique_sample_ids) > 1:
            raise ValueError(
                "This method is only valid for one-sample-per-plate designs"
            )

        return plate.unique_sample_ids[0]

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        logger.info(
            "Applying MergeTopBottomPlateSmoother with n_iterations={}".format(
                self.n_iterations
            )
        )
        current_screen = screen

        for sample_id in current_screen.unique_sample_ids:
            for i in range(self.n_iterations):
                plates = [
                    p
                    for p in current_screen.plates
                    if self._get_plate_sample_id(p) == sample_id
                ]

                if len(plates) <= 1:
                    break

                plates = sorted(plates, key=lambda x: x.size)
                halfway = math.floor(len(plates) / 2)

                for smaller_plate, bigger_plate in zip(
                    plates[:halfway], list(reversed(plates))[:halfway]
                ):
                    bigger_plate.merge(smaller_plate)

        return current_screen


class FixedSizeSmoother(RetrospectivePlateSmoother):
    """
    Filter all plates smaller than the given size and randomly truncate all plates larger
    than a fixed size to the given size.
    """

    def __init__(self, plate_size: int):
        self.plate_size = plate_size

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        logger.info(
            "Applying FixedSizeSmoother with plate_size={}".format(self.plate_size)
        )
        results = []

        for plate in screen.plates:
            if plate.size < self.plate_size:
                logger.info("Dropping plate of size {}".format(plate.size))
                continue
            elif plate.size == self.plate_size:
                results.append(plate)
            elif plate.size > self.plate_size:
                new_indices = rng.choice(
                    np.arange(screen.size)[plate.selection_vector],
                    self.plate_size,
                    replace=False,
                )
                new_selection_vector = np.isin(np.arange(screen.size), new_indices)

                results.append(Plate(screen, new_selection_vector))

        final_selection_vector = np.zeros(screen.size, dtype=bool)

        for plate in results:
            final_selection_vector = final_selection_vector | plate.selection_vector

        return screen.subset(final_selection_vector).to_screen()


class OptimalSizeSmoother(RetrospectivePlateSmoother):
    """
    The cost function for any particular plate size is the sum of two terms,
    the first term is the number of experiments you have to completely throw out
    because they are in plates below the threshold,
    the second term is the number of experiments that need to be trimmed out of plates
    that are over the threshold. This smoother optimizes this cost function
    and then drops all plates smaller than the optimal size and sub-samples all
    plates larger than the optimal size until all plates are the same size.
    """

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        logger.info("Applying OptimalSizeSmoother")
        # Find optimal plate size
        plate_sizes = np.sort(np.array([plate.size for plate in screen.plates]))
        i = np.argmax(plate_sizes * (len(plate_sizes) - np.arange(len(plate_sizes))))
        optimal_size = plate_sizes[i]

        logger.info("Optimal plate size is {}".format(optimal_size))

        results = []

        for plate in screen.plates:
            if plate.size < optimal_size:
                logger.info("Dropping plate of size {}".format(plate.size))
                continue
            elif plate.size == optimal_size:
                results.append(plate)
            elif plate.size > optimal_size:
                new_indices = rng.choice(
                    np.arange(screen.size)[plate.selection_vector],
                    optimal_size,
                    replace=False,
                )
                new_selection_vector = np.isin(np.arange(screen.size), new_indices)

                results.append(Plate(screen, new_selection_vector))

        final_selection_vector = np.zeros(screen.size, dtype=bool)

        for plate in results:
            final_selection_vector = final_selection_vector | plate.selection_vector

        return screen.subset(final_selection_vector).to_screen()


class NPlatePerCellLineSmoother(RetrospectivePlateSmoother):
    """
    Remove all experiments involving cell lines which have less than
    the user specified min_n_cell_line_plates
    """

    def __init__(self, min_n_cell_line_plates: int):
        self.min_n_cell_line_plates = min_n_cell_line_plates

    def _get_plate_sample_id(self, plate: Plate):
        if len(plate.unique_sample_ids) > 1:
            raise ValueError(
                "This method is only valid for one-sample-per-plate designs"
            )

        return plate.unique_sample_ids[0]

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        logger.info(
            "Applying NPlatePerCellLineSmoother with min_n_cell_line_plates={}".format(
                self.min_n_cell_line_plates
            )
        )
        plate_counts = defaultdict(lambda: 0)

        for plate in screen.plates:
            plate_counts[self._get_plate_sample_id(plate)] += 1

        for sample_id, plate_count in plate_counts.items():
            if plate_count < self.min_n_cell_line_plates:
                logger.info("Dropping all plates for sample {}".format(sample_id))
                screen = screen.subset(screen.sample_ids != sample_id).to_screen()

        return screen


class BatchieEnsemblePlateSmoother(RetrospectivePlateSmoother):
    """
    Apply the following smoothers in sequence to the input :py:class:`batchie.data.Screen`:

    :py:class:`MergeMinPlateSmoother`
    :py:class:`MergeTopBottomPlateSmoother`
    :py:class:`OptimalSizeSmoother`
    :py:class:`NPlatePerCellLineSmoother`
    """

    def __init__(self, min_size: int, n_iterations: int, min_n_cell_line_plates: int):
        self.min_size = min_size
        self.n_iterations = n_iterations
        self.min_n_cell_line_plates = min_n_cell_line_plates

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        screen = MergeMinPlateSmoother(min_size=self.min_size).smooth_plates(
            screen, rng
        )
        screen = MergeTopBottomPlateSmoother(
            n_iterations=self.n_iterations
        ).smooth_plates(screen, rng)
        screen = OptimalSizeSmoother().smooth_plates(screen, rng)
        screen = NPlatePerCellLineSmoother(
            min_n_cell_line_plates=self.min_n_cell_line_plates
        ).smooth_plates(screen, rng)
        return screen


def mask_screen(screen: Screen) -> Screen:
    return Screen(
        treatment_names=screen.treatment_names,
        treatment_doses=screen.treatment_doses,
        observations=screen.observations,
        sample_names=screen.sample_names,
        plate_names=screen.plate_names,
        control_treatment_name=screen.control_treatment_name,
        observation_mask=np.zeros(screen.size, dtype=bool),
    )


def unmask_screen(screen: Screen) -> Screen:
    return Screen(
        treatment_names=screen.treatment_names,
        treatment_doses=screen.treatment_doses,
        observations=screen.observations,
        sample_names=screen.sample_names,
        plate_names=screen.plate_names,
        control_treatment_name=screen.control_treatment_name,
        observation_mask=np.ones(screen.size, dtype=bool),
    )


def create_random_holdout(
    screen: Screen, fraction: float, rng: np.random.BitGenerator
) -> (Screen, Screen):
    """
    Create a random subset of a screen, of size fraction of the original screen.

    :param screen: The screen to create a holdout set for
    :param fraction: The fraction of the screen to hold out
    :return: A tuple of (training_screen, holdout_screen)
    """
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be between 0 and 1")

    selection_vector = np.zeros(screen.size, dtype=bool)

    indices = rng.choice(
        np.arange(screen.size),
        math.ceil(screen.size * fraction),
        replace=False,
    )
    selection_vector[indices] = True

    keep_screen = Screen(
        treatment_names=screen.treatment_names[~selection_vector],
        treatment_doses=screen.treatment_doses[~selection_vector],
        observations=screen.observations[~selection_vector],
        sample_names=screen.sample_names[~selection_vector],
        plate_names=screen.plate_names[~selection_vector],
        control_treatment_name=screen.control_treatment_name,
        observation_mask=screen.observation_mask[~selection_vector],
        sample_mapping=screen.sample_mapping,
        treatment_mapping=screen.treatment_mapping,
    )

    holdout_screen = Screen(
        treatment_names=screen.treatment_names[selection_vector],
        treatment_doses=screen.treatment_doses[selection_vector],
        observations=screen.observations[selection_vector],
        sample_names=screen.sample_names[selection_vector],
        plate_names=screen.plate_names[selection_vector],
        control_treatment_name=screen.control_treatment_name,
        observation_mask=np.ones(np.count_nonzero(selection_vector), dtype=bool),
        sample_mapping=screen.sample_mapping,
        treatment_mapping=screen.treatment_mapping,
    )

    return keep_screen, holdout_screen


def create_plate_balanced_holdout_set_among_masked_plates(
    screen: Screen, fraction: float, rng: np.random.BitGenerator
) -> (Screen, Screen):
    """
    Create a holdout set from a retrospective screen (where all data
    is observed but some plates are artificially masked) by sampling
    a fraction of each unobserved plate.

    :param screen: The screen to create a holdout set for
    :param fraction: The fraction of each unobserved plate to hold out
    :return: A tuple of (training_screen, holdout_screen)
    """
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be between 0 and 1")

    selection_vector = np.zeros(screen.size, dtype=bool)

    for plate in screen.plates:
        plate_indices = np.arange(screen.size)[plate.selection_vector]

        if plate.is_observed:
            continue

        n_sample = math.ceil(plate.size * fraction)

        downsampled_indices = rng.choice(
            plate_indices,
            n_sample,
            replace=False,
        )

        selection_vector[downsampled_indices] = True

    keep_screen = Screen(
        treatment_names=screen.treatment_names[~selection_vector],
        treatment_doses=screen.treatment_doses[~selection_vector],
        observations=screen.observations[~selection_vector],
        sample_names=screen.sample_names[~selection_vector],
        plate_names=screen.plate_names[~selection_vector],
        control_treatment_name=screen.control_treatment_name,
        observation_mask=screen.observation_mask[~selection_vector],
        treatment_mapping=screen.treatment_mapping,
        sample_mapping=screen.sample_mapping,
    )

    holdout_screen = Screen(
        treatment_names=screen.treatment_names[selection_vector],
        treatment_doses=screen.treatment_doses[selection_vector],
        observations=screen.observations[selection_vector],
        sample_names=screen.sample_names[selection_vector],
        plate_names=screen.plate_names[selection_vector],
        control_treatment_name=screen.control_treatment_name,
        observation_mask=np.ones(np.count_nonzero(selection_vector), dtype=bool),
        treatment_mapping=screen.treatment_mapping,
        sample_mapping=screen.sample_mapping,
    )

    return keep_screen, holdout_screen


def reveal_plates(
    screen: Screen,
    plate_ids: list[int],
) -> Screen:
    """
    Utility function to reveal observations in the masked screen from the observed screen.

    :param screen: A :py:class:`batchie.data.Screen` that is partially masked, but with real observations
        present in the internal observation array
    :param plate_ids: The plate ids to reveal
    """
    reveal_mask = np.isin(screen.plate_ids, plate_ids)

    revealed_values = screen.observations[reveal_mask]

    if np.all(revealed_values == 0):
        raise ValueError(
            "All revealed observations were 0 indicating a problem with your Screen, please check your data"
        )

    if np.any(np.isnan(revealed_values)):
        raise ValueError("NaN found in revealed observations, please check your data")

    return Screen(
        treatment_names=screen.treatment_names,
        treatment_doses=screen.treatment_doses,
        observations=screen.observations,
        sample_names=screen.sample_names,
        plate_names=screen.plate_names,
        control_treatment_name=screen.control_treatment_name,
        observation_mask=screen.observation_mask | reveal_mask,
    )


def calculate_mse(
    observed_screen: Screen,
    model: BayesianModel,
    thetas: ThetaHolder,
) -> float:
    """
    Calculate the mean squared error between the masked observations and the unmasked observations

    :param observed_screen: A :py:class:`Screen` that is fully observed
    :param model: The model to use for prediction
    :param thetas: The set of model parameters to use for prediction
    :return: The average mean squared error between predicted and observed values
    """
    preds = predict_avg(
        model=model,
        screen=observed_screen,
        thetas=thetas,
    )

    return np.mean((preds - observed_screen.observations) ** 2)
