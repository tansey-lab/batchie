from typing import Optional

import numpy as np
from batchie.common import CONTROL_SENTINEL_VALUE, FloatingPointType
from batchie.data import Experiment, Plate
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
        self, experiment: Experiment, rng: np.random.BitGenerator
    ):
        """
        We want to make sure we have at least one observation for
        each cell line/drug dose combination in the first plate

        This is called a greedy cover algorithm

        We'll use this to construct the first plate, this initialization causes
        faster convergence of the algorithm.
        """
        if not experiment.is_observed:
            raise ValueError(
                "The experiment used for retrospective "
                "analysis must be fully observed"
            )

        covered_treatments = set()
        chosen_selection_indices = []

        for sample_id in experiment.unique_sample_ids:
            experiments_with_at_least_one_treatment_not_in_covered_treatments = np.any(
                ~np.isin(experiment.treatment_ids, list(covered_treatments)), axis=1
            )

            selection_vector = (
                experiment.sample_ids == sample_id
            ) & experiments_with_at_least_one_treatment_not_in_covered_treatments
            if selection_vector.sum() > 0:
                selection_indices = np.arange(selection_vector.size)[selection_vector]
                chosen_selection_index = rng.choice(selection_indices, size=1)
                chosen_selection_indices.append(chosen_selection_index)

                covered_treatments.update(
                    set(experiment.treatment_ids[chosen_selection_indices].flatten())
                )
            else:  ## randomly choose some index corresponding to c
                selection_vector = experiment.sample_ids == sample_id
                selection_indices = np.arange(selection_vector.size)[selection_vector]
                chosen_selection_index = rng.choice(selection_indices, size=1)
                chosen_selection_indices.append(chosen_selection_index)
                covered_treatments.update(
                    set(experiment.treatment_ids[chosen_selection_indices].flatten())
                )

        remaining_treatments = np.setdiff1d(
            experiment.treatment_ids, list(covered_treatments)
        )

        while len(remaining_treatments) > 0:
            experiments_with_at_least_one_treatment_in_remaining_treatments = np.any(
                np.isin(experiment.treatment_ids, list(remaining_treatments)), axis=1
            )
            selection_indices = np.arange(selection_vector.size)[
                experiments_with_at_least_one_treatment_in_remaining_treatments
            ]
            chosen_selection_index = rng.choice(selection_indices, 1)
            chosen_selection_indices.append(chosen_selection_index)
            covered_treatments.update(
                set(experiment.treatment_ids[chosen_selection_indices].flatten())
            )
            remaining_treatments = np.setdiff1d(
                experiment.treatment_ids, list(covered_treatments)
            )

        final_plate_selection_vector = np.isin(
            np.arange(experiment.size), chosen_selection_indices
        )

        plate_names = np.array(["initial_plate"] * experiment.size, dtype=str)

        plate_names[~final_plate_selection_vector] = "unobserved_plate"

        observation_vector = experiment.observations.copy()
        observation_vector[~final_plate_selection_vector] = 0.0

        return Experiment(
            treatment_names=experiment.treatment_names,
            treatment_doses=experiment.treatment_doses,
            observations=observation_vector,
            sample_names=experiment.sample_names,
            plate_names=plate_names,
            control_treatment_name=experiment.control_treatment_name,
            observation_mask=final_plate_selection_vector,
        )


class PairwisePlateGenerator(RetrospectivePlateGenerator):
    def __init__(self, subset_size: int, anchor_size: int):
        self.anchor_size = anchor_size
        self.subset_size = subset_size

    def generate_plates(
        self, experiment: Experiment, rng: np.random.BitGenerator
    ) -> Experiment:
        """
        Break up your drug doses into groups of size subset_size

        Each plate will contain all pairwise combinations of the two groups of drug doses
        restricted to a single cell line

        IF you do this for GDCS you'll get weird plates because they have some "anchor drugs"
        that are used in almost every experiment. So you need to make sure that you have
        <anchor_size> anchor drugs in each group
        """

        unobserved_data = experiment.subset_unobserved()

        if not unobserved_data:
            logger.warning("No unobserved data found, returning original experiment")
            return experiment

        combo_mask = ~np.any(
            (unobserved_data.treatment_ids == CONTROL_SENTINEL_VALUE), axis=1
        )
        unique_treatments, unique_treatment_counts = np.unique(
            unobserved_data.treatment_ids[combo_mask], return_counts=True
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
        for g, idxs in enumerate(groupings):
            for j in idxs:
                group_lookup[j] = g
        group_lookup[CONTROL_SENTINEL_VALUE] = CONTROL_SENTINEL_VALUE

        treatment_group_ids = np.vectorize(group_lookup.get)(
            unobserved_data.treatment_ids
        )

        # Assign the groups for control treatments
        n_control = np.sum(treatment_group_ids == CONTROL_SENTINEL_VALUE)
        treatment_group_ids[treatment_group_ids == CONTROL_SENTINEL_VALUE] = rng.choice(
            range(num_groups), size=n_control, replace=True
        )
        treatment_group_ids_sorted = np.sort(treatment_group_ids, axis=1)

        sample_id_col_vector = unobserved_data.sample_ids[:, np.newaxis]

        grouping_tuples = np.hstack([sample_id_col_vector, treatment_group_ids_sorted])

        unique_grouping_tuples = np.unique(grouping_tuples, axis=0)

        new_plate_names = np.array([""] * unobserved_data.size, dtype=str)

        for idx, unique_grouping_tuple in enumerate(unique_grouping_tuples):
            mask = (grouping_tuples == unique_grouping_tuple).all(axis=1)
            new_plate_names[mask] = f"generated_plate_{idx}"

        unobserved_with_generated_plates = Experiment(
            treatment_names=unobserved_data.treatment_names,
            treatment_doses=unobserved_data.treatment_doses,
            observations=unobserved_data.observations,
            observation_mask=np.zeros(unobserved_data.size, dtype=bool),
            sample_names=unobserved_data.sample_names,
            plate_names=new_plate_names,
            control_treatment_name=unobserved_data.control_treatment_name,
        )

        observed_subset = experiment.subset_observed()

        if observed_subset:
            return experiment.subset_observed().combine(
                unobserved_with_generated_plates
            )
        else:
            return unobserved_with_generated_plates


class RandomPlateGenerator(RetrospectivePlateGenerator):
    def __init__(self, force_include_plate_names: Optional[list[str]] = None):
        self.force_include_plate_names = force_include_plate_names

    def generate_plates(self, experiment: Experiment, rng: np.random.BitGenerator):
        unobserved_experiment = experiment.subset_unobserved().to_experiment()

        if unobserved_experiment is None:
            logger.warning("No unobserved data found, returning original experiment")
            return experiment

        if self.force_include_plate_names:
            selection_vector = ~np.isin(
                unobserved_experiment.plate_names, self.force_include_plate_names
            )
        else:
            selection_vector = np.ones(unobserved_experiment.size, dtype=bool)

        if np.any(~selection_vector):
            to_permute = unobserved_experiment.subset(selection_vector).to_experiment()
            non_permuted = unobserved_experiment.subset(
                ~selection_vector
            ).to_experiment()
        else:
            to_permute = unobserved_experiment.subset(selection_vector)
            non_permuted = None

        new_plate_names = rng.permutation(to_permute.plate_names)

        permuted = Experiment(
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

        if experiment.subset_observed():
            return experiment.subset_observed().combine(new_unobserved)
        else:
            return new_unobserved


def mask_experiment(experiment: Experiment) -> Experiment:
    return Experiment(
        treatment_names=experiment.treatment_names,
        treatment_doses=experiment.treatment_doses,
        observations=np.zeros(experiment.size, dtype=FloatingPointType),
        sample_names=experiment.sample_names,
        plate_names=experiment.plate_names,
        control_treatment_name=experiment.control_treatment_name,
        observation_mask=np.zeros(experiment.size, dtype=bool),
    )


def reveal_plates(
    full_experiment: Experiment,
    masked_experiment: Experiment,
    plate_ids: list[int],
) -> Experiment:
    """
    Set observations in the masked experiment from the full experiment
    """
    selection_mask = np.isin(masked_experiment.plate_ids, plate_ids)

    masked_experiment.set_observed(
        selection_mask,
        full_experiment.observations[selection_mask],
    )

    return masked_experiment


def calculate_mse(
    full_experiment: Experiment,
    masked_experiment: Experiment,
    model: BayesianModel,
    thetas: ThetaHolder,
) -> float:
    preds = predict_avg(
        model=model,
        experiment=masked_experiment,
        thetas=thetas,
    )

    masked_obs = full_experiment.observations[~masked_experiment.observation_mask]

    prediction_of_masked_obs = preds[~masked_experiment.observation_mask]

    return np.mean((masked_obs - prediction_of_masked_obs) ** 2)
