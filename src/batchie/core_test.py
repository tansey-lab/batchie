import os
from dataclasses import dataclass

import numpy as np
import pytest
from batchie.common import ArrayType
from batchie.core import ThetaHolder, Theta, RetrospectivePlateGenerator
from batchie.data import Screen
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
    assert dense[0, 1] == 1.0


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


def test_retrospective_plate_generator():
    class TestRetrospectivePlateGenerator(RetrospectivePlateGenerator):
        def _generate_plates(
            self, screen: Screen, rng: np.random.BitGenerator
        ) -> Screen:
            return screen

    partially_observed_screen = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1]),
        observation_mask=np.array(
            [False, False, False, False, False, False, False, True, True]
        ),
        sample_names=np.array(["a", "a", "b", "b", "b", "b", "b", "b", "c"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "a", "a", "b", "c", "c"], dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
            ],
            dtype=str,
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
    )

    result = TestRetrospectivePlateGenerator().generate_plates(
        partially_observed_screen, np.random.default_rng(0)
    )

    assert result.observation_mask.sum() == 2
    assert result.n_unique_samples == 3
    assert len(result.plates) == 3
    assert result.size == 9

    fully_observed_screen = Screen(
        observations=np.array([0.4, 0.1]),
        observation_mask=np.array([True, True]),
        sample_names=np.array(["b", "c"], dtype=str),
        plate_names=np.array(["c", "c"], dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "b"],
            ],
            dtype=str,
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
    )

    result = TestRetrospectivePlateGenerator().generate_plates(
        fully_observed_screen, np.random.default_rng(0)
    )

    assert result is fully_observed_screen

    fully_unobserved_screen = Screen(
        observations=np.array([0.4, 0.1]),
        observation_mask=np.array([False, False]),
        sample_names=np.array(["b", "c"], dtype=str),
        plate_names=np.array(["c", "c"], dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "b"],
            ],
            dtype=str,
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
    )

    result = TestRetrospectivePlateGenerator().generate_plates(
        fully_unobserved_screen, np.random.default_rng(0)
    )

    assert result.size == 2
