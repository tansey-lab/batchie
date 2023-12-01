import numpy as np

ArrayType = np.array
FloatingPointType = float

CONTROL_SENTINEL_VALUE = -1


def copy_array_with_control_treatments_set_to_zero(
    arr: ArrayType, treatment_array: ArrayType
):
    results = arr[treatment_array, ...]
    results[treatment_array == CONTROL_SENTINEL_VALUE, ...] = 0.0
    return results


def select_unique_zipped_numpy_arrays(arrs):
    """
    Returns a boolean array that selects unique combinations of several same length numpy arrays.

    :param arrs: Arrays of the same length.
    :returns: Boolean array indicating unique combinations.
    """
    # Check if the arrays are of the same length
    if len(set([len(x) for x in arrs])) > 1:
        raise ValueError("All arrays must be of the same length")

    # Combine the arrays into a single 2D array
    combined = np.vstack(arrs).T

    # Find the unique rows (combinations) and their indices
    _, unique_indices = np.unique(combined, axis=0, return_index=True)

    # Create a boolean array where True indicates a unique combination
    result = np.zeros(len(arrs[0]), dtype=bool)
    result[unique_indices] = True

    return result


N_UNIQUE_TREATMENTS = "n_unique_treatments"
N_UNIQUE_SAMPLES = "n_unique_samples"
SELECTED_PLATES_KEY = "selected_plates"
