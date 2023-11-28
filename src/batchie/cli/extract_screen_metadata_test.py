import json
import os.path
import shutil
import tempfile

import numpy as np
import pytest

from batchie.cli import extract_screen_metadata
from batchie.data import Screen


@pytest.fixture
def test_dataset():
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


def test_main(mocker, test_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "extract_screen_metadata",
        "--screen",
        os.path.join(tmpdir, "screen.h5"),
        "--output",
        os.path.join(tmpdir, "screen_metadata.json"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "screen.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        extract_screen_metadata.main()
        with open(os.path.join(tmpdir, "screen_metadata.json"), "r") as f:
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
