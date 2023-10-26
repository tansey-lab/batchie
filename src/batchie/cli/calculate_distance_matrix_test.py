import os.path
import shutil
import tempfile

from batchie.cli import calculate_distance_matrix
from batchie.distance_calculation import ChunkedDistanceMatrix
from batchie.models.sparse_combo import SparseDrugComboResults


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
    results_holder = SparseDrugComboResults(
        n_thetas=10,
        n_unique_samples=test_combo_dataset.n_unique_samples,
        n_unique_treatments=test_combo_dataset.n_unique_treatments,
        n_embedding_dimensions=5,
    )

    results_holder._cursor = 10

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
        results_holder = SparseDrugComboResults(
            n_thetas=10,
            n_unique_samples=test_combo_dataset.n_unique_samples,
            n_unique_treatments=test_combo_dataset.n_unique_treatments,
            n_embedding_dimensions=5,
        )

        results_holder._cursor = 10

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
