from batchie.data import Experiment
from batchie.core import DistanceMetric, SamplesHolder, BayesianModel, DistanceMatrix
from batchie.common import ArrayType


def calculate_pairwise_distance_matrix_on_predictions(
    model: BayesianModel,
    samples: SamplesHolder,
    distance_metric: DistanceMetric,
    data: Experiment,
    chunk_indices: tuple[ArrayType, ArrayType],
):
    result = DistanceMatrix(
        size=samples.n_samples,
        chunk_size=len(chunk_indices[0]) * len(chunk_indices[1]),
    )

    for i in chunk_indices[0]:
        for j in chunk_indices[1]:
            sample_i = samples.get_sample(i)
            model.set_model_state(sample_i)
            i_pred = model.predict(data)

            sample_j = samples.get_sample(j)
            model.set_model_state(sample_j)
            j_pred = model.predict(data)

            value = distance_metric.distance(i_pred, j_pred)

            result.add_value(i, j, value)

    return result
