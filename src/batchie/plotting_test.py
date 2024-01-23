import tempfile
import shutil
import pandas
import numpy as np
import os

from batchie import plotting


def test_plot_correlation_heatmap():
    data = pandas.DataFrame(
        {
            "Feature1": np.random.rand(100),
            "Feature2": np.random.rand(100),
            "Feature3": np.random.rand(100),
            "Feature4": np.random.rand(100),
        }
    )

    corr = data.corr()

    tmpdir = tempfile.mkdtemp()

    try:
        plotting.plot_correlation_heatmap(corr, os.path.join(tmpdir, "example.pdf"))
    finally:
        shutil.rmtree(tmpdir)
