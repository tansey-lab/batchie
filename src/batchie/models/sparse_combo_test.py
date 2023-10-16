import numpy as np
import pytest
import tempfile

from batchie.data import Dataset
from batchie.models import sparse_combo


@pytest.fixture
def test_dataset():
    test_dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        single_effects=np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
        sample_names=np.array(["a", "b", "c", "d", "a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b", "a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
                ["a", "b"],
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
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
    )
    return test_dataset


def test_sparse_drug_combo_mcmc_step_with_observed_data(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=2, n_unique_samples=4
    )

    model.add_observations(test_dataset)

    model.mcmc_step()


def test_sparse_drug_combo_mcmc_step_without_observed_data(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.mcmc_step()


@pytest.mark.parametrize(
    "predict_interactions,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_predict_and_set_model_state(
    test_dataset, predict_interactions, interaction_log_transform
):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_samples,
        predict_interactions=predict_interactions,
        interaction_log_transform=interaction_log_transform,
    )

    model.mcmc_step()
    sample = model.get_model_state()

    prediction = model.predict(test_dataset)

    assert prediction.shape == (test_dataset.n_experiments,)

    model.reset_model()
    model.set_model_state(sample)
    prediction2 = model.predict(test_dataset)

    np.testing.assert_array_equal(prediction, prediction2)


def test_results_holder_accumulate(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_samples,
    )

    results_holder = sparse_combo.SparseDrugComboResults(
        n_mcmc_steps=2,
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_samples,
    )

    while not results_holder.is_complete:
        model.mcmc_step()
        sample = model.get_model_state()
        results_holder.add_sample(sample)


def test_results_holder_serde(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_samples,
    )

    results_holder = sparse_combo.SparseDrugComboResults(
        n_mcmc_steps=2,
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_samples,
    )
    results_holder.add_sample(model.get_model_state())

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = sparse_combo.SparseDrugComboResults.load_h5(f.name)

    assert results_holder2.n_samples == results_holder.n_mcmc_steps
    np.testing.assert_array_equal(results_holder2.V2, results_holder.V2)
    np.testing.assert_array_equal(results_holder2.V1, results_holder.V1)
    np.testing.assert_array_equal(results_holder2.V0, results_holder.V0)
    np.testing.assert_array_equal(results_holder2.W, results_holder.W)
