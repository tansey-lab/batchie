import os.path
import shutil
import tempfile

import numpy as np
import pytest

from batchie.core import ThetaHolder
from batchie.cli import analyze_model_evaluation
from batchie.data import Screen
from batchie.models.main import ModelEvaluation
from batchie.models.sparse_combo import SparseDrugComboMCMCSample


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
