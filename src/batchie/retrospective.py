from typing import Optional

import batchie.data
import numpy as np
from batchie.common import CONTROL_SENTINEL_VALUE, FloatingPointType
from batchie.data import Screen, Plate
from batchie.core import (
    BayesianModel,
    ThetaHolder,
    InitialRetrospectivePlateGenerator,
    RetrospectivePlateGenerator,
)
from batchie.models.main import predict_avg
import logging

logger = logging.getLogger(__name__)


class SparseCoverPlateGenerator(InitialRetrospectivePlateGenerator):
    def generate_and_unmask_initial_plate(
        self, screen: Screen, rng: np.random.BitGenerator
    ):
        """
        We want to make sure we have at least one observation for
        each cell line/drug dose combination in the first plate

        This is called a greedy cover algorithm

        We'll use this to construct the first plate, this initialization causes
        faster convergence of the algorithm.
        """
        if not screen.is_observed:
            raise ValueError(
                "The experiment used for retrospective "
                "analysis must be fully observed"
            )

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

    def generate_plates(self, screen: Screen, rng: np.random.BitGenerator) -> Screen:
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
        if not (~screen.observation_mask).any():
            logger.warning("No unobserved data found, returning original experiment")
            return screen

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


class RandomPlateGenerator(RetrospectivePlateGenerator):
    def __init__(self, force_include_plate_names: Optional[list[str]] = None):
        self.force_include_plate_names = force_include_plate_names

    def generate_plates(self, screen: Screen, rng: np.random.BitGenerator):
        unobserved_experiment = screen.subset_unobserved().to_screen()

        if unobserved_experiment is None:
            logger.warning("No unobserved data found, returning original experiment")
            return screen

        if self.force_include_plate_names:
            selection_vector = ~np.isin(
                unobserved_experiment.plate_names, self.force_include_plate_names
            )
        else:
            selection_vector = np.ones(unobserved_experiment.size, dtype=bool)

        if np.any(~selection_vector):
            to_permute = unobserved_experiment.subset(selection_vector).to_screen()
            non_permuted = unobserved_experiment.subset(~selection_vector).to_screen()
        else:
            to_permute = unobserved_experiment.subset(selection_vector).to_screen()
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
            new_unobserved = permuted.combine(non_permuted)
        else:
            new_unobserved = permuted

        if screen.subset_observed():
            return screen.subset_observed().to_screen().combine(new_unobserved)
        else:
            return new_unobserved


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
