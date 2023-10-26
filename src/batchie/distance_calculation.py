from batchie.core import (
    DistanceMetric,
    ThetaHolder,
    BayesianModel,
    DistanceMatrix,
)
from batchie.data import Experiment
from itertools import islice
import collections


def consume(iterator, n):
    """
    Advance the iterator n-steps ahead. If n is none, consume entirely.
    """
    collections.deque(islice(iterator, n), maxlen=0)


def lower_triangular_indices(n: int):
    """
    Iterate all the lower triangular indices of a square matrix with dimension n
    """
    for i in range(n):
        for j in range(i):
            yield i, j


def get_number_of_lower_triangular_indices(n: int):
    """
    Get the number of lower triangular indices of a square matrix with dimension n
    """
    return n * (n - 1) // 2


def get_lower_triangular_indices_chunk(n: int, chunk_index: int, n_chunks: int):
    """
    Assuming we want to split the number of lower triangular indices of a square matrix with dimension n
    into roughly equal chunks, return the indices for the chunk with index chunk_index
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


def calculate_pairwise_distance_matrix_on_predictions(
    model: BayesianModel,
    thetas: ThetaHolder,
    distance_metric: DistanceMetric,
    data: Experiment,
    chunk_index: int,
    n_chunks: int,
):
    indices = get_lower_triangular_indices_chunk(
        n=thetas.n_thetas, chunk_index=chunk_index, n_chunks=n_chunks
    )

    result = DistanceMatrix(
        size=thetas.n_thetas,
        chunk_size=len(indices),
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
