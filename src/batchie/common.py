import numpy as np
from numpy.typing import ArrayLike

ArrayType = ArrayLike

CONTROL_SENTINEL_VALUE = -1


def copy_array_with_control_treatments_set_to_zero(
    arr: ArrayType, treatment_array: ArrayType
):
    results = []
    for idx in treatment_array:
        if idx != CONTROL_SENTINEL_VALUE:
            results.append(arr[idx])
        else:
            results.append(np.zeros_like(arr[0]))

    return np.array(results)
