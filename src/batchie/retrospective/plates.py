from batchie.data import Dataset, DatasetSubset
import numpy as np


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
            chosen_selection_index = np.random.choice(selection_indices, size=1)
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
        chosen_selection_index = np.random.choice(selection_indices)
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
