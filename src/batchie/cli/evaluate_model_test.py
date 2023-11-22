import os.path
import shutil
import tempfile
import pytest
import numpy as np
import json
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.data import Screen
from batchie.common import SELECTED_PLATES_KEY


@pytest.fixture
def training_dataset():
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
def test_dataset():
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


def test_main(mocker, training_dataset, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "advance_retrospective_simulation",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--test-screen",
        os.path.join(tmpdir, "test.screen.h5"),
        "--training-screen",
        os.path.join(tmpdir, "training.screen.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--simulation-tracker-input",
        os.path.join(tmpdir, "simulation_tracker_input.json"),
        "--simulation-tracker-output",
        os.path.join(tmpdir, "simulation_tracker_output.json"),
        "--batch-selection",
        os.path.join(tmpdir, "batch_selection.json"),
        "--screen-output",
        os.path.join(tmpdir, "advanced_screen.h5"),
    ]

    training_dataset.save_h5(os.path.join(tmpdir, "training.screen.h5"))
    test_dataset.save_h5(os.path.join(tmpdir, "test.screen.h5"))
    results_holder = SparseDrugComboResults(
        n_thetas=10,
        n_unique_samples=training_dataset.n_unique_samples,
        n_unique_treatments=training_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)
