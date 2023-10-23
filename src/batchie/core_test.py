import os
from dataclasses import dataclass

import numpy as np
import pytest
from batchie.common import ArrayType
from batchie.core import DistanceMatrix, ThetaHolder, Theta


@dataclass
class TestBayesianModelParamsImpl(Theta):
    W: ArrayType


class TestThetaHolderImpl(ThetaHolder):
    def __init__(
        self,
        n_mcmc_steps: int,
    ):
        super().__init__(n_mcmc_steps)
        self.n_mcmc_steps = n_mcmc_steps

        self.W = np.zeros(
            (self.n_mcmc_steps, 2, 2),
            dtype=np.float32,
        )

    def _save_theta(
        self, sample: TestBayesianModelParamsImpl, variance: float, sample_index: int
    ):
        self.W[sample_index] = sample.W

    def get_theta(self, step_index: int) -> TestBayesianModelParamsImpl:
        return TestBayesianModelParamsImpl(W=self.W[step_index])

    def get_variance(self, step_index: int) -> float:
        return 1.0

    def save_h5(self, fn: str):
        pass

    @staticmethod
    def load_h5(path: str):
        return TestThetaHolderImpl(1)


def test_samples_holder():
    holder = TestThetaHolderImpl(3)

    for i in range(3):
        sample = TestBayesianModelParamsImpl(W=np.ones((2, 2)) * i)
        holder.add_theta(sample, 1.0)

    assert holder.is_complete

    samples = list(holder)
    np.testing.assert_array_equal(samples[0].W, np.ones((2, 2)) * 0)

    np.testing.assert_array_equal(samples[1].W, np.ones((2, 2)))


def test_samples_holder_concat():
    holder = TestThetaHolderImpl.concat(
        [TestThetaHolderImpl(3), TestThetaHolderImpl(3)]
    )

    assert len(list(holder)) == 6


@pytest.fixture
def distance_matrix():
    dm = DistanceMatrix(5, chunk_size=10)
    dm.add_value(0, 1, 2.5)
    dm.add_value(2, 3, 1.5)
    return dm


def test_add_value():
    dm = DistanceMatrix(5, chunk_size=10)
    dm.add_value(0, 1, 2.5)
    assert dm.values[0] == 2.5

    with pytest.raises(ValueError):
        dm.add_value(5, 6, 1.0)  # Indices out of bounds


def test_is_complete():
    distance_matrix = DistanceMatrix(5, chunk_size=25)

    assert not distance_matrix.is_complete()

    # Fill up the matrix to make it complete
    for i in range(5):
        for j in range(5):
            distance_matrix.add_value(i, j, 1.0)

    assert distance_matrix.is_complete()


def test_to_dense():
    distance_matrix = DistanceMatrix(5, chunk_size=10)
    with pytest.raises(ValueError):
        distance_matrix.to_dense()  # Incomplete matrix

    # Fill up the matrix to make it complete
    for i in range(5):
        for j in range(5):
            distance_matrix.add_value(i, j, 1.0)

    dense = distance_matrix.to_dense()
    assert isinstance(dense, np.ndarray)
    assert dense[0, 1] == 1.0


def test_save_load(distance_matrix):
    filename = "test_distance_matrix.h5"
    distance_matrix.save(filename)
    assert os.path.exists(filename)

    loaded = DistanceMatrix.load(filename)
    assert np.array_equal(
        loaded.row_indices[: loaded.current_index],
        distance_matrix.row_indices[: distance_matrix.current_index],
    )
    assert np.array_equal(
        loaded.col_indices[: loaded.current_index],
        distance_matrix.col_indices[: distance_matrix.current_index],
    )
    assert np.array_equal(
        loaded.values[: loaded.current_index],
        distance_matrix.values[: distance_matrix.current_index],
    )

    os.remove(filename)  # Cleanup


def test_combine_and_concat():
    dm = DistanceMatrix(5, chunk_size=10)
    dm.add_value(0, 1, 2.5)
    dm.add_value(2, 3, 1.5)
    dm2 = DistanceMatrix(5, chunk_size=10)
    dm2.add_value(1, 2, 3.5)
    dm2.add_value(3, 4, 0.5)

    composed = dm.combine(dm2)

    assert composed.values[0] == 2.5
    assert composed.values[1] == 1.5
    assert composed.values[2] == 3.5
    assert composed.values[3] == 0.5

    with pytest.raises(ValueError):
        dm3 = DistanceMatrix(6, chunk_size=10)  # Different size
        dm2.combine(dm3)

    concatted = DistanceMatrix.concat([dm, dm2])

    assert concatted.values[0] == 2.5
    assert concatted.values[1] == 1.5
    assert concatted.values[2] == 3.5
    assert concatted.values[3] == 0.5
