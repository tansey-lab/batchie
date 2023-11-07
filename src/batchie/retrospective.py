from typing import Optional

import batchie.data
import numpy as np
import math
from batchie.common import CONTROL_SENTINEL_VALUE, FloatingPointType
from batchie.data import Screen, Plate
from batchie.core import (
    BayesianModel,
    ThetaHolder,
    InitialRetrospectivePlateGenerator,
    RetrospectivePlateGenerator,
    RetrospectivePlateSmoother,
)
from batchie.models.main import predict_avg
import logging

logger = logging.getLogger(__name__)


class SparseCoverPlateGenerator(InitialRetrospectivePlateGenerator):
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

        plate_names = np.array(["initial_plate"] * screen.size, dtype=str)

        plate_names[~final_plate_selection_vector] = "unobserved_plate"

        logger.info(
            "Created initial plate of size {}".format(
                final_plate_selection_vector.sum()
            )
        )

        observation_vector = screen.observations.copy()
        observation_vector[~final_plate_selection_vector] = 0.0

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

        unobserved_combo_only_plates_mask = combo_mask & (~screen.observation_mask)

        unobserved_combo_only_plates = screen.subset(
            unobserved_combo_only_plates_mask
        ).to_screen()

        if (~unobserved_combo_only_plates_mask).any():
            remainder = screen.subset(~unobserved_combo_only_plates_mask).to_screen()
        else:
            remainder = None

        unique_treatments, unique_treatment_counts = np.unique(
            unobserved_combo_only_plates.treatment_ids, return_counts=True
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
            unobserved_combo_only_plates.treatment_ids
        )

        # Assign the groups for control treatments
        n_control = np.sum(treatment_group_ids == CONTROL_SENTINEL_VALUE)
        treatment_group_ids[treatment_group_ids == CONTROL_SENTINEL_VALUE] = rng.choice(
            range(num_groups), size=n_control, replace=True
        )
        treatment_group_ids_sorted = np.sort(treatment_group_ids, axis=1)

        sample_id_col_vector = unobserved_combo_only_plates.sample_ids[:, np.newaxis]

        grouping_tuples = np.hstack([sample_id_col_vector, treatment_group_ids_sorted])

        unique_grouping_tuples = np.unique(grouping_tuples, axis=0)

        new_plate_names = np.array(
            [""] * unobserved_combo_only_plates.size, dtype=object
        )

        for idx, unique_grouping_tuple in enumerate(unique_grouping_tuples):
            mask = (grouping_tuples == unique_grouping_tuple).all(axis=1)
            new_plate_names[mask] = f"generated_plate_{idx}"

        unobserved_with_generated_plates = Screen(
            treatment_names=unobserved_combo_only_plates.treatment_names,
            treatment_doses=unobserved_combo_only_plates.treatment_doses,
            observations=unobserved_combo_only_plates.observations,
            observation_mask=np.zeros(unobserved_combo_only_plates.size, dtype=bool),
            sample_names=unobserved_combo_only_plates.sample_names,
            plate_names=new_plate_names.astype(str),
            control_treatment_name=unobserved_combo_only_plates.control_treatment_name,
        )

        if remainder:
            return unobserved_with_generated_plates.combine(remainder)
        else:
            return unobserved_with_generated_plates


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
    def __init__(self, min_size: int):
        self.min_size = min_size

    def _get_plate_sample_id(self, plate: Plate):
        if len(plate.unique_sample_ids) > 1:
            raise ValueError(
                "This method is only valid for one-sample-per-plate designs"
            )

        return plate.unique_sample_ids[0]

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        current_screen = screen

        for sample_id in current_screen.unique_sample_ids:
            while True:
                plates = [
                    p
                    for p in current_screen.plates
                    if self._get_plate_sample_id(p) == sample_id
                ]

                if len(plates) <= 1:
                    break

                plates = sorted(plates, key=lambda x: x.size)
                smallest_plate = plates[0]
                second_smallest_plate = plates[1]

                if (smallest_plate.size + second_smallest_plate.size) > self.min_size:
                    break

                new_plate_names = current_screen.plate_names.copy()

                new_plate_names[
                    new_plate_names == smallest_plate.plate_name
                ] = second_smallest_plate.plate_name

                # merge plates
                current_screen = Screen(
                    treatment_names=current_screen.treatment_names,
                    treatment_doses=current_screen.treatment_doses,
                    observations=current_screen.observations,
                    sample_names=current_screen.sample_names,
                    plate_names=new_plate_names,
                    control_treatment_name=current_screen.control_treatment_name,
                    observation_mask=current_screen.observation_mask,
                )

        return current_screen


class MergeTopBottomPlateSmoother(RetrospectivePlateSmoother):
    def __init__(self, n_iterations: int):
        self.n_iterations = n_iterations

    def _get_plate_sample_id(self, plate: Plate):
        if len(plate.unique_sample_ids) > 1:
            raise ValueError(
                "This method is only valid for one-sample-per-plate designs"
            )

        return plate.unique_sample_ids[0]

    def _smooth_plates(self, screen: Screen, rng: np.random.BitGenerator):
        current_screen = screen

        for sample_id in current_screen.unique_sample_ids:
            for i in range(self.n_iterations):
                new_plate_names = current_screen.plate_names.copy()
                plates = [
                    p
                    for p in current_screen.plates
                    if self._get_plate_sample_id(p) == sample_id
                ]

                if len(plates) <= 1:
                    break

                plates = sorted(plates, key=lambda x: x.size)
                for smaller_plate, bigger_plate in zip(plates, reversed(plates)):
                    new_plate_selection = (
                        smaller_plate.selection_vector | bigger_plate.selection_vector
                    )
                    new_plate_names[new_plate_selection] = bigger_plate.plate_name

                current_screen = Screen(
                    treatment_names=current_screen.treatment_names,
                    treatment_doses=current_screen.treatment_doses,
                    observations=current_screen.observations,
                    sample_names=current_screen.sample_names,
                    plate_names=new_plate_names,
                    control_treatment_name=current_screen.control_treatment_name,
                    observation_mask=current_screen.observation_mask,
                )

        return current_screen


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
        # Find optimal plate size
        plate_sizes = np.sort(np.array([plate.size for plate in screen.plates]))
        i = np.argmax(plate_sizes * (len(plate_sizes) - np.arange(len(plate_sizes))))
        optimal_size = plate_sizes[i]

        results = []

        for plate in screen.plates:
            if plate.size < optimal_size:
                logger.info("Dropping plate of size {}".format(plate.size()))
                continue
            elif plate.size == optimal_size:
                results.append(plate.to_screen())
            elif plate.size > optimal_size:
                # create a boolean selection vector of size plate.size() with threshold_size True values
                selection_vector = np.zeros(plate.size, dtype=bool)
                selection_vector[:optimal_size] = True
                selection_vector = rng.permutation(selection_vector)

                results.append(plate.to_screen().subset(selection_vector).to_screen())

        return Screen.concat(results)


def mask_screen(screen: Screen) -> Screen:
    return Screen(
        treatment_names=screen.treatment_names,
        treatment_doses=screen.treatment_doses,
        observations=np.zeros(screen.size, dtype=FloatingPointType),
        sample_names=screen.sample_names,
        plate_names=screen.plate_names,
        control_treatment_name=screen.control_treatment_name,
        observation_mask=np.zeros(screen.size, dtype=bool),
    )


def reveal_plates(
    observed_screen: Screen,
    masked_screen: Screen,
    plate_ids: list[int],
) -> Screen:
    """
    Utility function to reveal observations in the masked screen from the observed screen.

    :param observed_screen: A :py:class:`batchie.data.Screen` that is fully observed
    :param masked_screen: The same :py:class:`batchie.data.Screen` that is partially observed
    :param plate_ids: The plate ids to reveal
    """
    selection_mask = np.isin(masked_screen.plate_ids, plate_ids)

    masked_screen.set_observed(
        selection_mask,
        observed_screen.observations[selection_mask],
    )

    return masked_screen


def calculate_mse(
    observed_screen: Screen,
    masked_screen: Screen,
    model: BayesianModel,
    thetas: ThetaHolder,
) -> float:
    """
    Calculate the mean squared error between the masked observations and the unmasked observations

    :param observed_screen: A :py:class:`Screen` that is fully observed
    :param masked_screen: The same :py:class:`Screen` that is partially observed
    :param model: The model to use for prediction
    :param thetas: The set of model parameters to use for prediction
    :return: The average mean squared error between predicted and observed values
    """
    preds = predict_avg(
        model=model,
        screen=masked_screen,
        thetas=thetas,
    )

    masked_obs = observed_screen.observations[~masked_screen.observation_mask]

    prediction_of_masked_obs = preds[~masked_screen.observation_mask]

    return np.mean((masked_obs - prediction_of_masked_obs) ** 2)
