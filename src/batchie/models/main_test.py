import numpy as np
import tempfile
import os
import shutil

from batchie.models import main
from batchie.common import FloatingPointType
from batchie.core import BayesianModel, ThetaHolder, ScreenBase
from batchie.data import Screen


def test_model_evaluation():
    preds = np.ones((10, 10), dtype=FloatingPointType)
    obs = np.ones((10, 10), dtype=FloatingPointType)
    chain_ids = np.ones(10, dtype=int)

    tmpdir = tempfile.mkdtemp()

    try:
        me = main.ModelEvaluation(
            predictions=preds, observations=obs, chain_ids=chain_ids
        )
        me.save_h5(os.path.join(tmpdir, "model_evaluation.h5"))
        main.ModelEvaluation.load_h5(os.path.join(tmpdir, "model_evaluation.h5"))
    finally:
        shutil.rmtree(tmpdir)


def test_predict_all(mocker):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    screen = Screen(
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

    thetas.n_thetas = 5

    model.predict.return_value = np.ones((screen.size,), dtype=FloatingPointType)

    result = main.predict_all(model, screen, thetas)

    assert result.shape == (5, 6)


def test_predict_avg(mocker):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    screen = Screen(
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

    thetas.n_thetas = 5

    model.predict.return_value = np.ones((screen.size,), dtype=FloatingPointType)

    result = main.predict_avg(model, screen, thetas)

    assert result.shape == (6,)
