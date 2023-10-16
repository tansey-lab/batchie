from batchie.distance import mse
import numpy as np
import pytest


@pytest.mark.parametrize("sigmoid", [True, False])
def test_distance_is_zero_for_same_data(sigmoid):
    mse_distance = mse.MSEDistance(sigmoid=sigmoid)
    assert mse_distance.distance(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0


@pytest.mark.parametrize("sigmoid", [True, False])
def test_distance_is_nonzero_for_same_data(sigmoid):
    mse_distance = mse.MSEDistance(sigmoid=sigmoid)
    assert mse_distance.distance(np.array([1, 2, 4]), np.array([1, 2, 3])) > 0
