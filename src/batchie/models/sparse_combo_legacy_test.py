import numpy as np
import pytest

from batchie.data import Screen
from batchie.models import sparse_combo_legacy


@pytest.fixture
def test_dataset():
    test_dataset = Screen(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2]),
        sample_names=np.array(["a", "a", "a", "b", "b", "b"], dtype=str),
        plate_names=np.array(["1"] * 6, dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
                ["a", "b"],
                ["a", "control"],
                ["control", "b"],
            ],
            dtype=str,
        ),
        treatment_doses=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
        control_treatment_name="control",
    )
    return test_dataset


def test_sparse_drug_combo_mcmc_step_with_observed_data(test_dataset):
    model = sparse_combo_legacy.LegacySparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=2, n_unique_samples=2
    )

    model.add_observations(test_dataset)

    model.step()


def test_sparse_drug_combo_mcmc_step_without_observed_data(test_dataset):
    model = sparse_combo_legacy.LegacySparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.step()


@pytest.mark.parametrize(
    "predict_interactions,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_predict_and_set_model_state(
    test_dataset, predict_interactions, interaction_log_transform
):
    model = sparse_combo_legacy.LegacySparseDrugCombo(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
        predict_interactions=predict_interactions,
        interaction_log_transform=interaction_log_transform,
    )

    model.step()
    sample = model.get_model_state()

    prediction = model.predict(test_dataset)

    assert prediction.shape == (test_dataset.size,)
    if not predict_interactions:
        assert np.all(prediction >= 0.0)
        assert np.all(prediction <= 1.0)

    model.reset_model()
    model.set_model_state(sample)
    prediction2 = model.predict(test_dataset)

    np.testing.assert_array_equal(prediction, prediction2)


@pytest.mark.parametrize(
    "predict_interactions,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_variance_and_set_model_state(
    test_dataset, predict_interactions, interaction_log_transform
):
    model = sparse_combo_legacy.LegacySparseDrugCombo(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
        predict_interactions=predict_interactions,
        interaction_log_transform=interaction_log_transform,
    )

    model.step()
    sample = model.get_model_state()
    variance = model.variance()

    model.reset_model()
    model.set_model_state(sample)
    variance2 = model.variance()

    np.testing.assert_array_equal(variance, variance2)
