import os.path
import shutil
import tempfile
import pytest
import numpy as np
import json
from batchie.cli import prepare_retrospective_experiment
from batchie.models.sparse_combo import SparseDrugComboResults
from batchie.data import Experiment
from batchie.common import SELECTED_PLATES_KEY


@pytest.fixture
def test_dataset():
    return Experiment(
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


def test_main_no_initial(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "prepare_retrospective_experiment",
        "--plate-generator",
        "RandomPlateGenerator",
        "--experiment",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "prepared_experiment.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        prepare_retrospective_experiment.main()

        exp_output = Experiment.load_h5(os.path.join(tmpdir, "prepared_experiment.h5"))

        assert len([x for x in exp_output.plates if x.is_observed]) == 1
    finally:
        shutil.rmtree(tmpdir)


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "prepare_retrospective_experiment",
        "--plate-generator",
        "RandomPlateGenerator",
        "--initial-plate-generator",
        "SparseCoverPlateGenerator",
        "--experiment",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "prepared_experiment.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        prepare_retrospective_experiment.main()

        exp_output = Experiment.load_h5(os.path.join(tmpdir, "prepared_experiment.h5"))

        assert len([x for x in exp_output.plates if x.is_observed]) == 1
    finally:
        shutil.rmtree(tmpdir)
