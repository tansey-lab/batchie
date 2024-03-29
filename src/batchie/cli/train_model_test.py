import os.path
import shutil
import tempfile

from batchie.cli import train_model
from batchie.core import ThetaHolder


def test_main(mocker, test_combo_dataset):
    tmpdir = tempfile.mkdtemp()
    command_line_args = [
        "train_model",
        "--model",
        "SparseDrugCombo",
        "--model-param",
        "n_embedding_dimensions=2",
        "--n-burnin",
        "1",
        "--n-samples",
        "1",
        "--thin",
        "1",
        "--data",
        os.path.join(tmpdir, "data.h5"),
        "--output",
        os.path.join(tmpdir, "samples.h5"),
    ]

    test_combo_dataset.save_h5(os.path.join(tmpdir, "data.h5"))
    mocker.patch("sys.argv", command_line_args)
    try:
        train_model.main()
        results = ThetaHolder.load_h5(os.path.join(tmpdir, "samples.h5"))
        assert results.n_thetas == 1

    finally:
        shutil.rmtree(tmpdir)
