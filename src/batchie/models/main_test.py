import os
import shutil
import tempfile

import numpy as np
import pytest

from batchie.common import FloatingPointType
from batchie.core import (
    Theta,
    ThetaHolder,
)
from batchie.data import Screen
from batchie.models import main


@pytest.fixture(scope="module")
def screen():
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


def test_model_evaluation():
    preds = np.ones((10, 10), dtype=FloatingPointType)
    obs = np.ones((10, 10), dtype=FloatingPointType)
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(["1"] * 10, dtype=str)
    tmpdir = tempfile.mkdtemp()

    try:
        me = main.ModelEvaluation(
            predictions=preds,
            observations=obs,
            chain_ids=chain_ids,
            sample_names=sample_names.astype(str),
        )
        me.mse()
        me.save_h5(os.path.join(tmpdir, "model_evaluation.h5"))
        new_me = main.ModelEvaluation.load_h5(
            os.path.join(tmpdir, "model_evaluation.h5")
        )

        assert np.allclose(me.predictions, new_me.predictions)
        assert np.allclose(me.observations, new_me.observations)
    finally:
        shutil.rmtree(tmpdir)


def test_model_evaluation_mse():
    preds = np.random.random((10, 10))
    obs = np.ones((10,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(["1"] * 10, dtype=str)

    me = main.ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )

    assert me.mse() > 0


def test_model_evaluation_mse_var():
    rng = np.random.default_rng(42)
    preds = rng.random((10, 10))
    obs = np.ones((10,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(["1"] * 10, dtype=str)

    me = main.ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )

    assert me.mse_variance() > 0

    preds = np.zeros((10, 10))
    obs = np.ones((10,))
    chain_ids = np.ones(10, dtype=int)
    sample_names = np.array(["1"] * 10, dtype=str)

    me2 = main.ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )

    assert me2.mse_variance() < me.mse_variance()


def test_model_evaluation_interchain_mse_variance():
    rng = np.random.default_rng(42)
    preds = np.hstack(
        [
            rng.normal(loc=1, scale=0.1, size=(10, 6)),
            rng.normal(loc=10, scale=0.1, size=(10, 6)),
        ]
    )
    preds2 = np.hstack(
        [
            rng.normal(loc=1, scale=0.1, size=(10, 6)),
            rng.normal(loc=2, scale=0.1, size=(10, 6)),
        ]
    )
    obs = rng.random(size=(10,))
    chain_ids = np.array(([1] * 6) + ([2] * 6), dtype=int)
    sample_names = np.array(["1"] * 10, dtype=str)

    me = main.ModelEvaluation(
        predictions=preds,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )

    me2 = main.ModelEvaluation(
        predictions=preds2,
        observations=obs,
        chain_ids=chain_ids,
        sample_names=sample_names,
    )

    assert me.inter_chain_mse_variance() > 0
    assert me2.inter_chain_mse_variance() > 0
    assert me.inter_chain_mse_variance() > me2.inter_chain_mse_variance()


def test_predict_all(mocker, screen):

    thetas = ThetaHolder(n_thetas=5)
    for _ in range(5):
        theta = mocker.MagicMock(spec=Theta)
        theta.predict_viability.return_value = np.ones(
            (screen.size,), dtype=FloatingPointType
        )
        thetas.add_theta(theta)

    result = main.predict_viability_all(screen, thetas)

    assert result.shape == (5, 6)


def test_predict_avg(mocker, screen):
    theta = mocker.MagicMock(spec=Theta)
    theta.predict_viability.return_value = np.ones(
        (screen.size,), dtype=FloatingPointType
    )
    thetas = ThetaHolder(5)
    for _ in range(5):
        thetas.add_theta(theta)

    result = main.predict_viability_avg(screen, thetas)

    assert result.shape == (6,)


def test_predict_raises_for_bad_values(mocker, screen):
    theta = mocker.MagicMock(spec=Theta)
    thetas = ThetaHolder(5)

    rv = np.ones((screen.size,), dtype=FloatingPointType)
    rv[0] = np.nan

    theta.predict_viability.return_value = rv
    theta.predict_conditional_mean.return_value = rv

    for _ in range(5):
        thetas.add_theta(theta)

    with pytest.raises(ValueError):
        main.predict_viability_all(screen, thetas)

    with pytest.raises(ValueError):
        main.predict_mean_all(screen, thetas)


def test_correlation_matrix(mocker, screen):
    thetas = ThetaHolder(3)
    theta = mocker.MagicMock(spec=Theta)
    rng = np.random.default_rng(0)

    def predict(screen: Screen):
        sample_id = list(screen.unique_sample_ids)[0]

        return rng.normal(loc=float(sample_id), scale=1.0, size=(screen.size,))

    theta.predict_viability.side_effect = predict

    for _ in range(3):
        thetas.add_theta(theta)

    result = main.correlation_matrix(screen, thetas)

    assert result.shape == (3, 3)
    assert np.all(result["c"].diff()[1:] > 0)


def test_predict_variance_all(mocker, screen):
    thetas = ThetaHolder(n_thetas=3)
    for i in range(3):
        theta = mocker.Mock(Theta)
        theta.predict_conditional_variance.return_value = np.arange(
            i, screen.size + i, dtype=FloatingPointType
        )
        thetas.add_theta(theta)

    result = main.predict_variance_all(screen, thetas)
    np.testing.assert_array_equal(
        result,
        np.array(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]],
            dtype=FloatingPointType,
        ),
    )


def test_predict_variance_all_raises_wrong_shape(mocker, screen):
    thetas = ThetaHolder(n_thetas=3)
    for i in range(3):
        theta = mocker.Mock(Theta)
        theta.predict_conditional_variance.return_value = np.arange(
            i + 5, dtype=FloatingPointType
        )
        thetas.add_theta(theta)

    with pytest.raises(ValueError):
        main.predict_variance_all(screen, thetas)


def test_predict_variance_all_raises_on_na(mocker, screen):
    thetas = ThetaHolder(n_thetas=3)
    for i in range(3):
        theta = mocker.Mock(Theta)
        v = np.arange(i, screen.size + i, dtype=FloatingPointType)
        if i == 0:
            v[-1] = np.nan
        theta.predict_conditional_variance.return_value = v
        thetas.add_theta(theta)

    with pytest.raises(ValueError):
        main.predict_variance_all(screen, thetas)
