import os.path
import shutil
import tempfile
import pytest
import numpy as np
from batchie.cli import reveal_plate
from batchie.data import Screen


@pytest.fixture
def test_dataset():
    return Screen(
        observations=np.array([0.1] * 10),
        observation_mask=np.array([True, True] + [False] * 8),
        sample_names=np.array(
            ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"], dtype=str
        ),
        plate_names=np.array(
            ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"], dtype=str
        ),
        treatment_names=np.array(
            [["a", "b"]] * 10,
            dtype=str,
        ),
        treatment_doses=np.array([[2.0, 2.0]] * 10),
    )


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "reveal_plate",
        "--screen",
        os.path.join(tmpdir, "screen.h5"),
        "--output",
        os.path.join(tmpdir, "advanced_screen.h5"),
        "--plate-id",
        "1",
        "2",
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "screen.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        reveal_plate.main()

        advanced_screen = Screen.load_h5(os.path.join(tmpdir, "advanced_screen.h5"))

        assert advanced_screen.observation_mask.sum() == 6
        assert sum([1 for p in advanced_screen.plates if p.is_observed]) == 3

    finally:
        shutil.rmtree(tmpdir)
