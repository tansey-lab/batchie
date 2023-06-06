import numpy as np

ArrayType = np.array

CONTROL_SENTINEL_VALUE = -1


def copy_array_with_control_treatments_set_to_zero(
    arr: ArrayType, treatment_array: ArrayType
):
    arr = arr.copy()
    arr[treatment_array == CONTROL_SENTINEL_VALUE] = 0.0
    return arr
