import tempfile
import shutil
import pandas
import numpy as np
import os

from batchie import plotting
from batchie.models.main import ModelEvaluation


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


def test_predicted_vs_observed_scatterplot():
    preds = np.random.random((10, 10))
    obs = np.random.random((10,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(["A"] * 10)

    me = ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )
    tmpdir = tempfile.mkdtemp()

    try:
        plotting.predicted_vs_observed_scatterplot(me, os.path.join(tmpdir, "fig.pdf"))
    finally:
        shutil.rmtree(tmpdir)


def test_predicted_vs_observed_scatterplot_per_sample():
    preds = np.random.random((13, 10))
    obs = np.random.random((13,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(
        ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D"]
    )

    me = ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )
    tmpdir = tempfile.mkdtemp()

    try:
        plotting.predicted_vs_observed_scatterplot_per_sample(
            me, os.path.join(tmpdir, "fig.pdf")
        )
    finally:
        shutil.rmtree(tmpdir)


def test_per_sample_violin_plot():
    rng = np.random.default_rng(42)
    preds = rng.random(size=(130, 10))
    obs = rng.random(size=(130,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(
        ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D"] * 10
    )
    me = ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )
    tmpdir = tempfile.mkdtemp()

    try:
        plotting.per_sample_violin_plot(
            me, os.path.join(tmpdir, "fig.pdf"), percentile=0
        )
        plotting.per_sample_violin_plot(
            me, os.path.join(tmpdir, "fig_95p.pdf"), percentile=95
        )
    finally:
        shutil.rmtree(tmpdir)
