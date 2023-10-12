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
