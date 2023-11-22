import os.path
import shutil
import tempfile
import pytest
import numpy as np
from batchie.cli import select_next_plate
from batchie.scoring.main import ChunkedScoresHolder
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


@pytest.fixture
def test_scores():
    scores = ChunkedScoresHolder(size=4)
    scores.add_score(0, 0.1)
    scores.add_score(1, 0.2)
    scores.add_score(2, 0.3)
    scores.add_score(3, 0.4)
    return scores


@pytest.mark.parametrize("use_policy", [True, False])
def test_main(mocker, test_dataset, test_scores, use_policy):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "select_next_plate",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--scores",
        os.path.join(tmpdir, "scores.h5"),
        "--output",
        os.path.join(tmpdir, "results.txt"),
    ]

    test_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    if use_policy:
        command_line_args.extend(
            [
                "--policy",
                "KPerSamplePlatePolicy",
                "--policy-param",
                "k=1",
            ]
        )

        # patch KPerSamplePlatePolicy
        mocker.patch(
            "batchie.policies.k_per_sample.KPerSamplePlatePolicy.filter_eligible_plates",
            return_value=[],
        )

    test_scores.save_h5(os.path.join(tmpdir, "scores.h5"))

    mocker.patch("sys.argv", command_line_args)

    try:
        select_next_plate.main()

        with open(os.path.join(tmpdir, "results.txt"), "r") as f:
            content = f.read()
        if use_policy:
            assert content == ""
        else:
            assert int(content.strip()) == 0

    finally:
        shutil.rmtree(tmpdir)
