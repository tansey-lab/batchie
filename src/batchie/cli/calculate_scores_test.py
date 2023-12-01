import os
import shutil
import tempfile

import numpy as np
import pytest

from batchie.cli import calculate_scores
from batchie.data import Screen
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.scoring.main import ChunkedScoresHolder


@pytest.fixture
def test_dataset():
    return Screen(
        observations=np.array([0.1, 0.2, 0, 0, 0, 0]),
        observation_mask=np.array([True, True, False, False, False, False]),
        sample_names=np.array(["a", "a", "b", "b", "c", "c"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "c", "c"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]],
            dtype=str,
        ),
        treatment_doses=np.array(
            [[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0.1], [2.0, 1.0], [2.0, 1.0]]
        ),
    )


@pytest.fixture
def test_dist_matrix():
    distance_matrix = ChunkedDistanceMatrix(size=10)

    for i in range(10):
        for j in range(i):
            distance_matrix.add_value(i, j, i + j)
    return distance_matrix


def test_main(mocker, test_dataset, test_dist_matrix):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "calculate_scores",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--scorer",
        "RandomScorer",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--output",
        os.path.join(tmpdir, "results.json"),
        "--n-chunks",
        "2",
        "--chunk-index",
        "1",
        "--distance-matrix",
        os.path.join(tmpdir, "distance_matrix.h5"),
        "--output",
        os.path.join(tmpdir, "scores.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    results_holder = SparseDrugComboResults(
        n_thetas=10,
        n_unique_samples=test_dataset.n_unique_samples,
        n_unique_treatments=test_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)

    test_dist_matrix.save(os.path.join(tmpdir, "distance_matrix.h5"))

    try:
        calculate_scores.main()
        result = ChunkedScoresHolder.load_h5(os.path.join(tmpdir, "scores.h5"))

        assert result.scores.size == 1
    finally:
        shutil.rmtree(tmpdir)


def test_main_exclude(mocker, test_dataset, test_dist_matrix):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "calculate_scores",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--scorer",
        "RandomScorer",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--output",
        os.path.join(tmpdir, "results.json"),
        "--distance-matrix",
        os.path.join(tmpdir, "distance_matrix.h5"),
        "--output",
        os.path.join(tmpdir, "scores.h5"),
        "--batch-plate-ids",
        "1",
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    results_holder = SparseDrugComboResults(
        n_thetas=10,
        n_unique_samples=test_dataset.n_unique_samples,
        n_unique_treatments=test_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)

    test_dist_matrix.save(os.path.join(tmpdir, "distance_matrix.h5"))

    try:
        calculate_scores.main()
        result = ChunkedScoresHolder.load_h5(os.path.join(tmpdir, "scores.h5"))

        assert result.scores.size == 1
    finally:
        shutil.rmtree(tmpdir)
