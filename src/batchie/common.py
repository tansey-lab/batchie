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


N_UNIQUE_TREATMENTS = "n_unique_treatments"
N_UNIQUE_SAMPLES = "n_unique_samples"
SELECTED_PLATES_KEY = "selected_plates"
