from batchie.common import ArrayType, CONTROL_SENTINEL_VALUE
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_single_treatment_effect_map(
    sample_ids: ArrayType,
    treatment_ids: ArrayType,
    observation: ArrayType,
):
    if treatment_ids.shape[1] < 2:
        raise ValueError(
            "Experiment must have more than one treatment to get single treatment effects"
        )

    # find observations where all but one treatment ids are control
    single_treatment_mask = np.sum(treatment_ids == CONTROL_SENTINEL_VALUE, axis=1) == (
        treatment_ids.shape[1] - 1
    )
    single_treatment_observations = observation[single_treatment_mask]
    single_treatment_treatments = np.sort(
        treatment_ids[single_treatment_mask, :], axis=1
    )[:, -1]
    single_treatment_sample_ids = sample_ids[single_treatment_mask]

    result: dict[tuple[int, int], float] = {}

    for current_sample_id in np.unique(sample_ids):
        for current_treatment_id in np.unique(treatment_ids.flatten()):
            if current_treatment_id == CONTROL_SENTINEL_VALUE:
                continue

            mask = (single_treatment_treatments == current_treatment_id) & (
                single_treatment_sample_ids == current_sample_id
            )

            if not np.any(mask):
                continue

            single_effect = np.mean(single_treatment_observations[mask])
            result[(current_sample_id, current_treatment_id)] = single_effect

    return result


def calculate_synergy(
    sample_ids: ArrayType,
    treatment_ids: ArrayType,
    observation: ArrayType,
    strict: bool = False,
):
    """
    Calculate synergy for a given set of observations, sample ids, and treatment ids.

    If single treatment observations for all of the treatments in a multi-treatment observation are not present,
    the observation is skipped. If strict is True, an error is raised instead.
    """
    if treatment_ids.shape[1] < 2:
        raise ValueError(
            "Experiment must have more than one treatment to calculate synergy"
        )

    if sample_ids.shape[0] != treatment_ids.shape[0]:
        raise ValueError("Sample and treatment ids must be the same length")

    if sample_ids.shape[0] != observation.shape[0]:
        raise ValueError("Sample and observation ids must be the same length")

    single_treatment_effect_map = create_single_treatment_effect_map(
        sample_ids=sample_ids, treatment_ids=treatment_ids, observation=observation
    )

    # find observations where all but one treatment ids are control
    single_treatment_mask = np.sum(treatment_ids == CONTROL_SENTINEL_VALUE, axis=1) == (
        treatment_ids.shape[1] - 1
    )

    multi_treatment_observation = observation[~single_treatment_mask]
    multi_treatment_treatments = treatment_ids[~single_treatment_mask, :]
    multi_treatment_sample_ids = sample_ids[~single_treatment_mask]

    result_synergy = []
    result_treatment_ids = []
    result_sample_ids = []

    for idx, (current_sample_id, current_treatment_ids, observation) in enumerate(
        zip(
            multi_treatment_sample_ids,
            multi_treatment_treatments,
            multi_treatment_observation,
        )
    ):
        current_treatment_ids = current_treatment_ids[
            current_treatment_ids != CONTROL_SENTINEL_VALUE
        ]
        single_effects = []

        for current_treatment_id in current_treatment_ids:
            if (
                current_sample_id,
                current_treatment_id,
            ) not in single_treatment_effect_map:
                if strict:
                    raise ValueError(
                        f"Sample {current_sample_id} has no control for treatment {current_treatment_id}"
                    )
                else:
                    logger.warning(
                        f"Sample {current_sample_id} has no control for treatment {current_treatment_id}"
                    )
                    continue

            single_effects.append(
                single_treatment_effect_map[(current_sample_id, current_treatment_id)]
            )

        if len(current_treatment_ids) != len(single_effects):
            continue

        synergy = np.prod(single_effects) - observation

        result_synergy.append(synergy)
        result_treatment_ids.append(current_treatment_ids)
        result_sample_ids.append(current_sample_id)

    return (
        np.array(result_sample_ids),
        np.array(result_treatment_ids),
        np.array(result_synergy),
    )
