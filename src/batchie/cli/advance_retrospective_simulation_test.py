import os.path
import shutil
import tempfile
import pytest
import numpy as np
import json
from batchie.cli import advance_retrospective_simulation
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.data import Screen
from batchie.common import SELECTED_PLATES_KEY


@pytest.fixture
def test_masked_dataset():
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
def test_unmasked_dataset():
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


def test_main(mocker, test_masked_dataset, test_unmasked_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "advance_retrospective_simulation",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--unmasked-screen",
        os.path.join(tmpdir, "unmasked_experiment.h5"),
        "--masked-screen",
        os.path.join(tmpdir, "masked_experiment.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--experiment-tracker-input",
        os.path.join(tmpdir, "experiment_tracker_input.json"),
        "--experiment-tracker-output",
        os.path.join(tmpdir, "experiment_tracker_output.json"),
        "--batch-selection",
        os.path.join(tmpdir, "batch_selection.json"),
        "--screen-output",
        os.path.join(tmpdir, "advanced_experiment.h5"),
    ]

    test_masked_dataset.save_h5(os.path.join(tmpdir, "masked_experiment.h5"))
    test_masked_dataset.save_h5(os.path.join(tmpdir, "unmasked_experiment.h5"))
    results_holder = SparseDrugComboResults(
        n_thetas=10,
        n_unique_samples=test_masked_dataset.n_unique_samples,
        n_unique_treatments=test_masked_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

    results_holder.save_h5(os.path.join(tmpdir, "samples.h5"))

    mocker.patch("sys.argv", command_line_args)

    with open(os.path.join(tmpdir, "batch_selection.json"), "w") as f:
        json.dump({SELECTED_PLATES_KEY: [1]}, f)

    with open(os.path.join(tmpdir, "experiment_tracker_input.json"), "w") as f:
        json.dump({"losses": [0.0], "plate_ids_selected": [[0]], "seed": 0}, f)

    try:
        advance_retrospective_simulation.main()
        with open(os.path.join(tmpdir, "experiment_tracker_output.json"), "r") as f:
            results = json.load(f)

        assert results["plate_ids_selected"][-1] == [1]
        assert len(results["losses"]) == 2

        exp_output = Screen.load_h5(os.path.join(tmpdir, "advanced_experiment.h5"))

        np.testing.assert_array_equal(
            exp_output.observation_mask,
            np.array([True, True, True, True, False, False]),
        )
    finally:
        shutil.rmtree(tmpdir)
