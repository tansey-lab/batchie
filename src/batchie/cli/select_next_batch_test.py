import os.path
import shutil
import tempfile
import pytest
import numpy as np
import json
from batchie.cli import select_next_batch
from batchie.core import DistanceMatrix
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.data import Experiment
from batchie.common import SELECTED_PLATES_KEY


@pytest.fixture
def test_dataset():
    return Experiment(
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


@pytest.mark.parametrize("use_policy", [True, False])
def test_main(mocker, test_dataset, use_policy):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "select_next_batch",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--scorer",
        "RandomScorer",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--samples",
        os.path.join(tmpdir, "samples.h5"),
        "--output",
        os.path.join(tmpdir, "results.json"),
        "--batch-size",
        "1",
        "--distance-matrix",
        os.path.join(tmpdir, "distance_matrix.h5"),
    ]

    if use_policy:
        command_line_args.extend(
            [
                "--policy",
                "KPerSamplePlatePolicy",
                "--policy-param",
                "k=1",
            ]
        )

    test_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    results_holder = SparseDrugComboResults(
        n_samples=10,
        n_unique_samples=test_dataset.n_unique_samples,
        n_unique_treatments=test_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)

    distance_matrix = DistanceMatrix(size=10)

    for i in range(10):
        for j in range(10):
            distance_matrix.add_value(i, j, i + j)

    distance_matrix.save(os.path.join(tmpdir, "distance_matrix.h5"))

    try:
        select_next_batch.main()
        with open(os.path.join(tmpdir, "results.json"), "r") as f:
            results = json.load(f)

        assert len(results[SELECTED_PLATES_KEY]) == 1
    finally:
        shutil.rmtree(tmpdir)
