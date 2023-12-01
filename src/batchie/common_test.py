import numpy as np
import numpy.testing

from batchie import common


def test_copy_array_with_control_treatments_set_to_zero():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    treatment_array = np.array([0, 1, -1])
    result = common.copy_array_with_control_treatments_set_to_zero(arr, treatment_array)

    numpy.testing.assert_array_equal(
        result, np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
    )

    arr = np.array([1, 2])
    treatment_array = np.array([0, 1, -1])
    result = common.copy_array_with_control_treatments_set_to_zero(arr, treatment_array)

    numpy.testing.assert_array_equal(result, np.array([1, 2, 0]))


def test_select_unique_zipped_numpy_arrays():
    arr1 = np.array([1, 2, 3, 1, 3])
    arr2 = np.array([4, 5, 6, 4, 6])
    arr3 = np.array([7, 8, 9, 7, 9])

    arrs = [arr1, arr2, arr3]

    result = common.select_unique_zipped_numpy_arrays(arrs)

    numpy.testing.assert_array_equal(result, np.array([True, True, True, False, False]))
