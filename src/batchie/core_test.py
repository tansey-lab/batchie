import os
from dataclasses import dataclass

import numpy as np
import pytest
from batchie.common import ArrayType
from batchie.core import ThetaHolder, Theta
from batchie.distance_calculation import ChunkedDistanceMatrix


@dataclass
class TestBayesianModelParamsImpl(Theta):
    W: ArrayType


class TestThetaHolderImpl(ThetaHolder):
    def __init__(
        self,
        n_thetas: int,
    ):
        super().__init__(n_thetas)

        self.W = np.zeros(
            (self.n_thetas, 2, 2),
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

    def combine(self, other):
        if type(self) != type(other):
            raise ValueError("Cannot combine with different type")

        output = TestThetaHolderImpl(
            n_thetas=self.n_thetas + other.n_thetas,
        )

        for i in range(self.n_thetas):
            sample = self.get_theta(i)
            variance = self.get_variance(i)
            output.add_theta(sample, variance)

        for i in range(other.n_thetas):
            sample = other.get_theta(i)
            variance = other.get_variance(i)
            output.add_theta(sample, variance)

        return output

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
    dm = ChunkedDistanceMatrix(5)
    dm.add_value(1, 0, 2.5)
    dm.add_value(3, 2, 1.5)
    return dm


def test_add_value():
    dm = ChunkedDistanceMatrix(5)
    dm.add_value(1, 0, 2.5)
    assert dm.values[0] == 2.5

    with pytest.raises(ValueError):
        dm.add_value(5, 6, 1.0)  # Indices out of bounds


def test_is_complete():
    distance_matrix = ChunkedDistanceMatrix(5)

    assert not distance_matrix.is_complete()

    # Fill up the matrix to make it complete
    for i in range(5):
        for j in range(i):
            distance_matrix.add_value(i, j, 1.0)

    assert distance_matrix.is_complete()


def test_to_dense():
    distance_matrix = ChunkedDistanceMatrix(5)
    with pytest.raises(ValueError):
        distance_matrix.to_dense()  # Incomplete matrix

    # Fill up the matrix to make it complete
    for i in range(5):
        for j in range(i):
            distance_matrix.add_value(i, j, 1.0)

    dense = distance_matrix.to_dense()
    assert isinstance(dense, np.ndarray)
    assert dense[1, 0] == 1.0
    np.testing.assert_array_equal(dense, np.tril(dense))


def test_save_load(distance_matrix):
    filename = "test_distance_matrix.h5"
    distance_matrix.save(filename)
    assert os.path.exists(filename)

    loaded = ChunkedDistanceMatrix.load(filename)
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
