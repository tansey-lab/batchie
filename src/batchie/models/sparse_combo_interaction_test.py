import numpy as np
import pytest
import tempfile

from batchie.data import Experiment
from batchie.models import sparse_combo_interaction


@pytest.fixture
def test_dataset():
    test_dataset = Experiment(
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
    model = sparse_combo_interaction.SparseDrugComboInteraction(
        n_embedding_dimensions=5, n_unique_treatments=2, n_unique_samples=4
    )

    model.add_observations(test_dataset)

    model.step()


@pytest.mark.parametrize(
    "adjust_single,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_predict_and_set_model_state(
    test_dataset, adjust_single, interaction_log_transform
):
    model = sparse_combo_interaction.SparseDrugComboInteraction(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_unique_samples,
        adjust_single=adjust_single,
        interaction_log_transform=interaction_log_transform,
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


def test_results_holder_serde(test_dataset):
    model = sparse_combo_interaction.SparseDrugComboInteraction(
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_unique_samples,
    )

    results_holder = sparse_combo_interaction.SparseDrugComboInteractionResults(
        n_samples=2,
        n_embedding_dimensions=5,
        n_unique_treatments=test_dataset.n_treatments,
        n_unique_samples=test_dataset.n_unique_samples,
    )
    results_holder.add_sample(model.get_model_state(), model.variance())

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = (
            sparse_combo_interaction.SparseDrugComboInteractionResults.load_h5(f.name)
        )

    assert results_holder2.n_unique_samples == results_holder.n_samples
    np.testing.assert_array_equal(results_holder2.V2, results_holder.V2)
    np.testing.assert_array_equal(results_holder2.W, results_holder.W)
