import os.path
import shutil
import tempfile
import numpy as np
from batchie.cli import calculate_distance_matrix
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.sparse_combo import SparseDrugComboMCMCSample
from batchie.core import ThetaHolder


def test_main_complete(mocker, test_combo_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "train_model",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--distance-metric",
        "MSEDistance",
        "--n-chunks",
        "1",
        "--chunk-index",
        "0",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--thetas",
        os.path.join(tmpdir, "samples.h5"),
        "--output",
        os.path.join(tmpdir, "matrix.h5"),
    ]

    test_combo_dataset.save_h5(os.path.join(tmpdir, "data.h5"))

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

    mocker.patch("sys.argv", command_line_args)
    try:
        calculate_distance_matrix.main()
        results = ChunkedDistanceMatrix.load(os.path.join(tmpdir, "matrix.h5"))
        assert results.is_complete()

    finally:
        shutil.rmtree(tmpdir)


def test_main_partial(mocker, test_combo_dataset):
    all_results = []
    for i in range(2):
        tmpdir = tempfile.mkdtemp()
        command_line_args = [
            "train_model",
            "--model",
            "SparseDrugCombo",
            "--model-param",
            "n_embedding_dimensions=2",
            "--distance-metric",
            "MSEDistance",
            "--n-chunks",
            "2",
            "--chunk-index",
            f"{i}",
            "--data",
            os.path.join(tmpdir, "data.h5"),
            "--thetas",
            os.path.join(tmpdir, "samples.h5"),
            "--output",
            os.path.join(tmpdir, "matrix.h5"),
        ]

        test_combo_dataset.save_h5(os.path.join(tmpdir, "data.h5"))

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

        mocker.patch("sys.argv", command_line_args)
        try:
            calculate_distance_matrix.main()
            results = ChunkedDistanceMatrix.load(os.path.join(tmpdir, "matrix.h5"))
            assert not results.is_complete()
            all_results.append(results)

        finally:
            shutil.rmtree(tmpdir)

    end_result = ChunkedDistanceMatrix.concat(all_results)
    assert end_result.is_complete()
