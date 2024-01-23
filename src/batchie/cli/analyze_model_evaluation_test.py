import os.path
import shutil
import tempfile

import numpy as np
import pytest

from batchie.cli import analyze_model_evaluation
from batchie.data import Screen
from batchie.models.main import ModelEvaluation
from batchie.models.sparse_combo import SparseDrugComboResults


@pytest.fixture
def dataset():
    return Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        observation_mask=np.array([True, True, True, True, True, True]),
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


def test_main(mocker, dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "evaluate_model",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=5",
        "--model-evaluation",
        os.path.join(tmpdir, "model_evaluation.h5"),
        "--screen",
        os.path.join(tmpdir, "screen.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--output-dir",
        os.path.join(tmpdir, "out"),
    ]

    dataset.save_h5(os.path.join(tmpdir, "screen.h5"))

    n_thetas = 10
    results_holder = SparseDrugComboResults(
        n_thetas=n_thetas,
        n_unique_samples=dataset.n_unique_samples,
        n_unique_treatments=dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = n_thetas

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    me = ModelEvaluation(
        observations=dataset.observations[:4],
        predictions=np.random.random((4, 5)),
        sample_names=dataset.sample_names[:4],
        chain_ids=np.array([1, 1, 1, 1, 1]),
    )

    me.save_h5(os.path.join(tmpdir, "model_evaluation.h5"))
    mocker.patch("sys.argv", command_line_args)

    try:
        analyze_model_evaluation.main()
    finally:
        shutil.rmtree(tmpdir)
