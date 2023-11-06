import h5py
import numpy as np

from batchie.core import (
    DistanceMetric,
    DistanceMatrix,
    ThetaHolder,
    BayesianModel,
)
from batchie.data import Screen
from itertools import islice
import collections


def consume(iterator, n):
    """
    Advance the iterator n-steps ahead. If n is none, consume entirely.

    :param iterator: The iterator to consume
    :param n: The number of steps to advance the iterator
    """
    collections.deque(islice(iterator, n), maxlen=0)


def lower_triangular_indices(n: int):
    """
    Iterate all the lower triangular indices of a square matrix with dimension n

    :param n: The dimension of the square matrix
    :return: A generator which yields the indices
    """
    for i in range(n):
        for j in range(i):
            yield i, j


def get_number_of_lower_triangular_indices(n: int):
    """
    Get the number of lower triangular indices of a square matrix with dimension n

    :param n: The dimension of the square matrix
    :return: The number of lower triangular indices
    """
    return n * (n - 1) // 2


def get_lower_triangular_indices_chunk(n: int, chunk_index: int, n_chunks: int):
    """
    Assuming we want to split the number of lower triangular indices of a square matrix with dimension n
    into roughly equal chunks, return the indices for the chunk with index chunk_index

    :param n: The dimension of the square matrix
    :param chunk_index: The index of the chunk to return
    :param n_chunks: The number of chunks to split the indices into
    :return: A list of indices
    """
    assert chunk_index < n_chunks

    n_indices = get_number_of_lower_triangular_indices(n)

    chunk_size = n_indices // n_chunks
    remainder = n_indices % n_chunks

    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size

    if chunk_index < remainder:
        start_index += chunk_index
        end_index += chunk_index + 1
    else:
        start_index += remainder
        end_index += remainder

    g = lower_triangular_indices(n)
    consume(g, start_index)

    return list(islice(g, end_index - start_index))


class ChunkedDistanceMatrix(DistanceMatrix):
    """
    Class which can represent part or a whole pairwise distance matrix.

    The distance matrix is stored in a sparse format, but can be converted to a dense format if all values
    are present.

    Several partial ChunkedDistanceMatrix classes can be combined. This is useful for parallelization of the distance
    matrix computation.
    """

    def __init__(self, size, n_chunks=1, chunk_index=0, chunk_size=None):
        self.size = size
        if chunk_size:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = len(
                get_lower_triangular_indices_chunk(
                    n=size, chunk_index=chunk_index, n_chunks=n_chunks
                )
            )
        self.current_index = 0
        self.row_indices = np.zeros(self.chunk_size, dtype=int)
        self.col_indices = np.zeros(self.chunk_size, dtype=int)
        self.values = np.zeros(self.chunk_size, dtype=float)

    def _expand_storage(self):
        self.row_indices = np.concatenate(
            (self.row_indices, np.zeros(self.chunk_size, dtype=int))
        )
        self.col_indices = np.concatenate(
            (self.col_indices, np.zeros(self.chunk_size, dtype=int))
        )
        self.values = np.concatenate(
            (self.values, np.zeros(self.chunk_size, dtype=float))
        )

    def add_value(self, i, j, value):
        if i >= self.size or j >= self.size:
            raise ValueError("Indices are out of bounds")

        if i < j:
            raise ValueError("Indices must be lower triangular")

        if self.current_index + 1 > len(self.values):
            self._expand_storage()

        if self.row_indices[self.current_index] != 0:
            raise ValueError("This distance has already been calculated")
        self.row_indices[self.current_index] = i

        if self.col_indices[self.current_index] != 0:
            raise ValueError("This distance has already been calculated")

        self.col_indices[self.current_index] = j
        if self.values[self.current_index] != 0:
            raise ValueError("This distance has already been calculated")
        self.values[self.current_index] = value
        self.current_index += 1

    def is_complete(self):
        return self.current_index == get_number_of_lower_triangular_indices(self.size)

    def to_dense(self):
        if not self.is_complete():
            raise ValueError("The distance matrix is not complete")
        dense = np.zeros((self.size, self.size))
        for i in range(self.current_index):
            dense[self.row_indices[i], self.col_indices[i]] = self.values[i]
            dense[self.col_indices[i], self.row_indices[i]] = self.values[i]
        return dense

    def save(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "row_indices",
                data=self.row_indices[: self.current_index],
                compression="gzip",
            )
            f.create_dataset(
                "col_indices",
                data=self.col_indices[: self.current_index],
                compression="gzip",
            )
            f.create_dataset(
                "values", data=self.values[: self.current_index], compression="gzip"
            )
            f.create_dataset("size", data=np.array([self.size]), compression="gzip")

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, "r") as f:
            row_indices = f["row_indices"][:]
            col_indices = f["col_indices"][:]
            values = f["values"][:]
            size = f["size"][0]
            instance = cls(
                size, chunk_size=len(values)
            )  # assuming all values are filled, otherwise adjust chunk_size accordingly
            instance.row_indices[: len(values)] = row_indices
            instance.col_indices[: len(values)] = col_indices
            instance.values[: len(values)] = values
            instance.current_index = len(values)
        return instance

    def combine(self, other):
        if self.size != other.size:
            raise ValueError(
                "The matrices must be of the same size to be composed together"
            )

        composed = ChunkedDistanceMatrix(self.size, chunk_size=self.current_index)
        composed.row_indices[: self.current_index] = self.row_indices[
            : self.current_index
        ]
        composed.col_indices[: self.current_index] = self.col_indices[
            : self.current_index
        ]
        composed.values[: self.current_index] = self.values[: self.current_index]
        composed.current_index = self.current_index

        for i in range(other.current_index):
            row, col, value = (
                other.row_indices[i],
                other.col_indices[i],
                other.values[i],
            )
            if (row, col) not in zip(
                composed.row_indices[: composed.current_index],
                composed.col_indices[: composed.current_index],
            ):
                composed.add_value(row, col, value)

        return composed

    @classmethod
    def concat(cls, matrices: list):
        if len(matrices) == 1:
            return matrices[0]
        elif len(matrices) == 0:
            raise ValueError("Cannot concat empty list of matrices")

        accumulator = matrices[0]
        for matrix in matrices[1:]:
            if accumulator.size != matrix.size:
                raise ValueError("Cannot concat matrices of different sizes")
            accumulator = accumulator.combine(matrix)
        return accumulator


def calculate_pairwise_distance_matrix_on_predictions(
    model: BayesianModel,
    thetas: ThetaHolder,
    distance_metric: DistanceMetric,
    data: Screen,
    chunk_index: int,
    n_chunks: int,
) -> ChunkedDistanceMatrix:
    """
    Calculate the pairwise distance matrix between predictions.

    For all pairs of thetas in the given :py:class:`ThetaHolder`, predictions will be made on the unobserved
    conditions in the given :py:class:`Experiment` and the distance between the predictions produced by the two
    theta values will be calculated and populated into a :py:class:`ChunkedDistanceMatrix` instance.

    If n_chunks > 1, then the distance matrix is split into n_chunks roughly equal chunks, and only the chunk with
    index chunk_index is calculated. This is useful for parallelization.

    :param model: The model to use for prediction
    :param thetas: The set of model parameters to use for prediction
    :param distance_metric: The distance metric to use
    :param data: The data to predict
    :param chunk_index: The index of the chunk to calculate
    :param n_chunks: The number of chunks to split the distance matrix into
    :return: A :py:class:`ChunkedDistanceMatrix` containing the pairwise distances
    """
    indices = get_lower_triangular_indices_chunk(
        n=thetas.n_thetas, chunk_index=chunk_index, n_chunks=n_chunks
    )

    result = ChunkedDistanceMatrix(
        size=thetas.n_thetas, chunk_index=chunk_index, n_chunks=n_chunks
    )

    for i, j in indices:
        sample_i = thetas.get_theta(i)
        model.set_model_state(sample_i)
        i_pred = model.predict(data)

        sample_j = thetas.get_theta(j)
        model.set_model_state(sample_j)
        j_pred = model.predict(data)

        value = distance_metric.distance(i_pred, j_pred)

        result.add_value(i, j, value)

    return result
