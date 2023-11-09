import os.path
import shutil
import tempfile
import pytest
import numpy as np
from batchie.cli import prepare_retrospective_simulation
from batchie.data import Screen


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


def test_main_no_initial_no_smoother(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "prepare_retrospective_simulation",
        "--plate-generator",
        "PlatePermutationPlateGenerator",
        "--data",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "prepared_experiment.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        prepare_retrospective_simulation.main()

        exp_output = Screen.load_h5(os.path.join(tmpdir, "prepared_experiment.h5"))

        assert len([x for x in exp_output.plates if x.is_observed]) == 1
    finally:
        shutil.rmtree(tmpdir)


def test_main_no_smoother(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "prepare_retrospective_simulation",
        "--plate-generator",
        "PlatePermutationPlateGenerator",
        "--initial-plate-generator",
        "SparseCoverPlateGenerator",
        "--initial-plate-generator-param",
        "reveal_single_treatment_experiments=False",
        "--data",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "prepared_experiment.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        prepare_retrospective_simulation.main()

        exp_output = Screen.load_h5(os.path.join(tmpdir, "prepared_experiment.h5"))

        assert len([x for x in exp_output.plates if x.is_observed]) == 1
    finally:
        shutil.rmtree(tmpdir)


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "prepare_retrospective_simulation",
        "--plate-generator",
        "PlatePermutationPlateGenerator",
        "--initial-plate-generator",
        "SparseCoverPlateGenerator",
        "--initial-plate-generator-param",
        "reveal_single_treatment_experiments=False",
        "--plate-smoother",
        "FixedSizeSmoother",
        "--plate-smoother-param",
        "plate_size=2",
        "--data",
        os.path.join(tmpdir, "experiment.h5"),
        "--output",
        os.path.join(tmpdir, "prepared_experiment.h5"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "experiment.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        prepare_retrospective_simulation.main()

        exp_output = Screen.load_h5(os.path.join(tmpdir, "prepared_experiment.h5"))

        assert len([x for x in exp_output.plates if x.is_observed]) == 1
    finally:
        shutil.rmtree(tmpdir)
