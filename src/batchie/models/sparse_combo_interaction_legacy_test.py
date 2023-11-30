import tempfile

import numpy as np
import pytest

from batchie.data import Screen
from batchie.models import sparse_combo_interaction_legacy


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


def test_step_with_observed_data(test_dataset):
    model = sparse_combo_interaction_legacy.LegacySparseDrugComboInteraction(
        n_embedding_dimensions=5, n_unique_treatments=2, n_unique_samples=4
    )

    model.add_observations(test_dataset)

    model.step()


def test_predict_and_set_model_state(test_dataset):
    model = sparse_combo_interaction_legacy.LegacySparseDrugComboInteraction(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
    )
    model.add_observations(test_dataset)
    model.step()
    sample = model.get_model_state()

    prediction = model.predict(test_dataset)

    assert prediction.shape == (test_dataset.size,)

    model.reset_model()
    model.set_model_state(sample)
    prediction2 = model.predict(test_dataset)

    np.testing.assert_array_equal(prediction, prediction2)
    assert (prediction < 1).all()
    assert (prediction > 0).all()


def test_results_holder_serde(test_dataset):
    model = sparse_combo_interaction_legacy.LegacySparseDrugComboInteraction(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
    )

    results_holder = sparse_combo_interaction_legacy.SparseDrugComboInteractionResults(
        n_thetas=2,
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
    )
    results_holder.add_theta(model.get_model_state(), model.variance())
    results_holder.add_theta(model.get_model_state(), model.variance())

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = (
            sparse_combo_interaction_legacy.SparseDrugComboInteractionResults.load_h5(
                f.name
            )
        )

    assert results_holder2.n_unique_samples == results_holder.n_thetas
    np.testing.assert_array_equal(results_holder2.V2, results_holder.V2)
    np.testing.assert_array_equal(results_holder2.W, results_holder.W)

    results_holder2 = sparse_combo_interaction_legacy.SparseDrugComboInteractionResults(
        n_thetas=2,
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.treatment_arity,
        n_unique_samples=test_dataset.n_unique_samples,
    )

    results_holder2.add_theta(model.get_model_state(), model.variance())
    results_holder2.add_theta(model.get_model_state(), model.variance())
    assert results_holder.combine(results_holder2).n_thetas == 4
