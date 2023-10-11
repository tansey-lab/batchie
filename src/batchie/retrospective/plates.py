from batchie.data import Dataset, DatasetSubset
import numpy as np
from batchie.common import CONTROL_SENTINEL_VALUE


def create_sparse_cover_plate(
    dataset: Dataset, rng: np.random.BitGenerator
) -> DatasetSubset:
    """
    We want to make sure we have at least one observation for
    each cell line/drug dose combination in the first plate

    This is called a greedy cover algorithm

    We'll use this to construct the first plate, this initialization causes
    faster convergence of the algorithm.
    """
    covered_treatments = set()
    chosen_selection_indices = []

    for sample_id in dataset.unique_sample_ids:
        experiments_with_at_least_one_treatment_not_in_covered_treatments = np.any(
            ~np.isin(dataset.treatment_ids, list(covered_treatments)), axis=1
        )

        selection_vector = (
            dataset.sample_ids == sample_id
        ) & experiments_with_at_least_one_treatment_not_in_covered_treatments
        if selection_vector.sum() > 0:
            selection_indices = np.arange(selection_vector.size)[selection_vector]
            chosen_selection_index = rng.choice(selection_indices, size=1)
            chosen_selection_indices.append(chosen_selection_index)

            covered_treatments.update(
                set(dataset.treatment_ids[chosen_selection_indices].flatten())
            )
        else:  ## randomly choose some index corresponding to c
            selection_vector = dataset.sample_ids == sample_id
            selection_indices = np.arange(selection_vector.size)[selection_vector]
            chosen_selection_index = rng.choice(selection_indices, size=1)
            chosen_selection_indices.append(chosen_selection_index)
            covered_treatments.update(
                set(dataset.treatment_ids[chosen_selection_indices].flatten())
            )

    remaining_treatments = np.setdiff1d(dataset.treatment_ids, list(covered_treatments))

    while len(remaining_treatments) > 0:
        experiments_with_at_least_one_treatment_in_remaining_treatments = np.any(
            np.isin(dataset.treatment_ids, list(remaining_treatments)), axis=1
        )
        selection_indices = np.arange(selection_vector.size)[
            experiments_with_at_least_one_treatment_in_remaining_treatments
        ]
        chosen_selection_index = rng.choice(selection_indices)
        chosen_selection_indices.append(chosen_selection_index)
        covered_treatments.update(
            set(dataset.treatment_ids[chosen_selection_indices].flatten())
        )
        remaining_treatments = np.setdiff1d(
            dataset.treatment_ids, list(covered_treatments)
        )

    final_plate_selection_vector = np.isin(
        np.arange(dataset.n_experiments), chosen_selection_indices
    )

    return DatasetSubset(dataset, final_plate_selection_vector)


def create_sarcoma_plates(
    dataset: Dataset,
    subset_size: int,
    rng: np.random.BitGenerator,
    anchor_size: int = 0,
) -> list[DatasetSubset]:
    """
    Break up your drug doses into groups of size subset_size

    Each plate will contain all pairwise combinations of the two groups of drug doses
    restricted to a single cell line

    IF you do this for GDCS youll get weird plates because they have some "anchor drugs"
    that are used in almost every experiment. So you need to make sure that you have
    <anchor_size> anchor drugs in each group
    """
    combo_mask = ~np.any((dataset.treatment_ids == CONTROL_SENTINEL_VALUE), axis=1)
    unique_treatments, unique_treatment_counts = np.unique(
        dataset.treatment_ids[combo_mask], return_counts=True
    )

    if anchor_size > 0:
        anchor_dds = unique_treatments[
            np.argsort(-unique_treatment_counts)[:anchor_size]
        ]
        n_anchor_groups = len(anchor_dds) // subset_size
        anchor_groups = np.array_split(rng.permutation(anchor_dds), n_anchor_groups)

        remain_dds = np.setdiff1d(unique_treatments, anchor_dds)
        n_remain_groups = len(remain_dds) // subset_size
        remain_groups = np.array_split(rng.permutation(remain_dds), n_remain_groups)
        groupings = anchor_groups + remain_groups
    else:
        n_groups = len(unique_treatments) // subset_size
        groupings = np.array_split(rng.permutation(unique_treatments), n_groups)

    num_groups = len(groupings)
    group_lookup = {}
    for g, idxs in enumerate(groupings):
        for j in idxs:
            group_lookup[j] = g
    group_lookup[CONTROL_SENTINEL_VALUE] = CONTROL_SENTINEL_VALUE

    treatment_group_ids = np.vectorize(group_lookup.get)(dataset.treatment_ids)

    # Assign the groups for control treatments
    n_control = np.sum(treatment_group_ids == CONTROL_SENTINEL_VALUE)
    treatment_group_ids[treatment_group_ids == CONTROL_SENTINEL_VALUE] = rng.choice(
        range(num_groups), size=n_control, replace=True
    )
    treatment_group_ids_sorted = np.sort(treatment_group_ids, axis=1)

    sample_id_col_vector = dataset.sample_ids[:, np.newaxis]

    grouping_tuples = np.hstack([sample_id_col_vector, treatment_group_ids_sorted])

    unique_grouping_tuples = np.unique(grouping_tuples, axis=0)

    results = []
    for unique_grouping_tuple in unique_grouping_tuples:
        mask = (grouping_tuples == unique_grouping_tuple).all(axis=1)
        results.append(DatasetSubset(dataset, mask))

    return results
