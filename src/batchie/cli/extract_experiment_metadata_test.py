import os.path
import shutil
import tempfile
import pytest
import numpy as np
import json
from batchie.cli import extract_experiment_metadata
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


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "extract_experiment_metadata",
        "--experiment",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "experiment_metadata.json"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        extract_experiment_metadata.main()
        with open(os.path.join(tmpdir, "experiment_metadata.json"), "r") as f:
            results = json.load(f)

        assert results == {
            "n_observed_plates": 1,
            "n_plates": 3,
            "n_unique_samples": 3,
            "n_unique_treatments": 5,
            "n_unobserved_plates": 2,
            "size": 6,
        }
    finally:
        shutil.rmtree(tmpdir)
