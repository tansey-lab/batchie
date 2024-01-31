import os
import shutil
import tempfile
import pytest
import numpy as np

from batchie.common import FloatingPointType
from batchie.core import (
    BayesianModel,
    HomoscedasticModel,
    HeteroscedasticModel,
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
        me.rmse()
        me.save_h5(os.path.join(tmpdir, "model_evaluation.h5"))
        new_me = main.ModelEvaluation.load_h5(
            os.path.join(tmpdir, "model_evaluation.h5")
        )

        assert np.allclose(me.predictions, new_me.predictions)
        assert np.allclose(me.observations, new_me.observations)
    finally:
        shutil.rmtree(tmpdir)


def test_model_evaluation_rmse():
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

    assert me.rmse() > 0


def test_predict_all(mocker, screen):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    thetas.n_thetas = 5

    model.predict.return_value = np.ones((screen.size,), dtype=FloatingPointType)

    result = main.predict_all(model, screen, thetas)

    assert result.shape == (5, 6)


def test_predict_avg(mocker, screen):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    thetas.n_thetas = 5

    model.predict.return_value = np.ones((screen.size,), dtype=FloatingPointType)

    result = main.predict_avg(model, screen, thetas)

    assert result.shape == (6,)


def test_predict_raises_for_bad_values(mocker, screen):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    thetas.n_thetas = 5

    rv = np.ones((screen.size,), dtype=FloatingPointType)

    rv[0] = np.nan

    model.predict.return_value = rv

    with pytest.raises(ValueError):
        main.predict_all(model, screen, thetas)

    with pytest.raises(ValueError):
        main.predict_avg(model, screen, thetas)


def test_correlation_matrix(mocker, screen):
    model = mocker.MagicMock(spec=BayesianModel)
    thetas = mocker.MagicMock(spec=ThetaHolder)

    thetas.n_thetas = 5

    model.predict.return_value = np.ones((10,), dtype=FloatingPointType)

    rng = np.random.default_rng(0)

    def predict(screen: Screen):
        sample_id = list(screen.unique_sample_ids)[0]

        return rng.normal(loc=float(sample_id), scale=1.0, size=(screen.size,))

    model.predict.side_effect = predict

    result = main.correlation_matrix(model, screen, thetas)

    assert result.shape == (3, 3)
    assert np.all(result["c"].diff()[1:] > 0)


def test_get_homoescedastic_variances(mocker, screen):
    model = mocker.Mock()
    thetas = mocker.MagicMock(spec=ThetaHolder)
    model.variance.side_effect = [1, 2, 3]
    thetas.n_thetas = 3

    result = main.get_homoescedastic_variances(model, screen, thetas)
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=FloatingPointType))


def test_get_heteroescedastic_variances(mocker, screen):
    model = mocker.Mock()
    thetas = mocker.MagicMock(spec=ThetaHolder)

    model.variance.side_effect = [
        np.array([1, 2, 3, 4, 5, 6], dtype=FloatingPointType),
        np.array([7, 8, 9, 10, 11, 12], dtype=FloatingPointType),
        np.array([13, 14, 15, 16, 17, 18], dtype=FloatingPointType),
    ]
    thetas.n_thetas = 3

    result = main.get_heteroescedastic_variances(model, screen, thetas)
    np.testing.assert_array_equal(
        result,
        np.array(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
            dtype=FloatingPointType,
        ),
    )


def test_get_heteroescedastic_variances_raises_wrong_shape(mocker, screen):
    model = mocker.Mock()
    thetas = mocker.MagicMock(spec=ThetaHolder)

    model.variance.side_effect = [
        np.array([1, 2, 3, 4, 5, 6, 1], dtype=FloatingPointType),
        np.array([7, 8, 9, 10, 11, 12], dtype=FloatingPointType),
        np.array([13, 14, 15, 16, 17, 18], dtype=FloatingPointType),
    ]
    thetas.n_thetas = 3

    with pytest.raises(ValueError):
        main.get_heteroescedastic_variances(model, screen, thetas)


def test_get_homoescedastic_variances_raises_on_na(mocker, screen):
    model = mocker.Mock()
    thetas = mocker.MagicMock(spec=ThetaHolder)
    model.variance.side_effect = [1, 2, np.nan]
    thetas.n_thetas = 3

    with pytest.raises(ValueError):
        main.get_homoescedastic_variances(model, screen, thetas)


def test_get_heteroescedastic_variances_raises_on_na(mocker, screen):
    model = mocker.Mock()
    thetas = mocker.MagicMock(spec=ThetaHolder)

    model.variance.side_effect = [
        np.array([1, 2, 3, 4, 5, np.nan], dtype=FloatingPointType),
        np.array([7, 8, 9, 10, 11, 12], dtype=FloatingPointType),
        np.array([13, 14, 15, 16, 17, 18], dtype=FloatingPointType),
    ]
    thetas.n_thetas = 3

    with pytest.raises(ValueError):
        main.get_heteroescedastic_variances(model, screen, thetas)
