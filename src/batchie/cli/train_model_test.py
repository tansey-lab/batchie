from batchie.cli import train_model
import tempfile
import pytest
import numpy as np
import os.path
import shutil
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.data import Experiment


@pytest.fixture
def test_dataset():
    test_dataset = Experiment(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2]),
        sample_names=np.array(["a", "a", "a", "b", "b", "b"], dtype=str),
        plate_names=np.array(["1"] * 6, dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
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
            ]
        ),
        control_treatment_name="control",
    )
    return test_dataset


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "train_model",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--n-burnin",
        "1",
        "--n-samples",
        "1",
        "--thin",
        "1",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--output",
        os.path.join(tmpdir, "samples.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    mocker.patch("sys.argv", command_line_args)
    try:
        train_model.main()
        results = SparseDrugComboResults.load_h5(os.path.join(tmpdir, "samples.h5"))
        assert results.n_samples == 1

    finally:
        shutil.rmtree(tmpdir)
