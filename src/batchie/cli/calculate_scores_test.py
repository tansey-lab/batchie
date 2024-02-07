import os
import shutil
import tempfile

import numpy as np
import pytest

from batchie.cli import calculate_scores
from batchie.core import ThetaHolder
from batchie.data import Screen
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.sparse_combo import SparseDrugComboMCMCSample
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
    n_thetas = 10
    results_holder = ThetaHolder(n_thetas=n_thetas)
    for _ in range(n_thetas):
        theta = SparseDrugComboMCMCSample(
            W=np.zeros((5, 5)),
            W0=np.zeros((5,)),
            V2=np.zeros((10, 5)),
            V1=np.zeros((10, 5)),
            V0=np.zeros((10,)),
            alpha=5.0,
            precision=100.0,
        )
        results_holder.add_theta(theta)

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
    n_thetas = 10
    results_holder = ThetaHolder(n_thetas=n_thetas)
    for _ in range(n_thetas):
        theta = SparseDrugComboMCMCSample(
            W=np.zeros((5, 5)),
            W0=np.zeros((5,)),
            V2=np.zeros((10, 5)),
            V1=np.zeros((10, 5)),
            V0=np.zeros((10,)),
            alpha=5.0,
            precision=100.0,
        )
        results_holder.add_theta(theta)

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)

    test_dist_matrix.save(os.path.join(tmpdir, "distance_matrix.h5"))

    try:
        calculate_scores.main()
        result = ChunkedScoresHolder.load_h5(os.path.join(tmpdir, "scores.h5"))

        assert result.scores.size == 1
    finally:
        shutil.rmtree(tmpdir)
