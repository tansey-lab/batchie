import os
import shutil
import tempfile
import pytest
import numpy as np

from batchie.common import FloatingPointType
from batchie.core import BayesianModel, ThetaHolder
from batchie.data import Screen
from batchie.models import main


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
        new_me = main.ModelEvaluation.load_h5(
            os.path.join(tmpdir, "model_evaluation.h5")
        )

        assert np.allclose(me.predictions, new_me.predictions)
        assert np.allclose(me.observations, new_me.observations)
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


def test_predict_raises_for_bad_values(mocker):
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

    rv = np.ones((screen.size,), dtype=FloatingPointType)

    rv[0] = np.nan

    model.predict.return_value = rv

    with pytest.raises(ValueError):
        main.predict_all(model, screen, thetas)

    with pytest.raises(ValueError):
        main.predict_avg(model, screen, thetas)
