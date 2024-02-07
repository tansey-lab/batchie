import tempfile

import numpy as np
import pytest

from batchie.core import ThetaHolder
from batchie.data import Screen, ExperimentSpace
from batchie.models import sparse_combo


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
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, experiment_space=experiment_space
    )

    model.add_observations(test_dataset)

    model.step()


def test_sparse_drug_combo_refuses_to_accept_bad_observations(test_dataset):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, experiment_space=experiment_space
    )

    test_dataset.observations[0] = -1

    with pytest.raises(ValueError):
        model.add_observations(test_dataset)

    test_dataset.observations[0] = np.nan

    with pytest.raises(ValueError):
        model.add_observations(test_dataset)


def test_sparse_drug_combo_mcmc_step_without_observed_data(test_dataset):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, experiment_space=experiment_space
    )

    model.step()


@pytest.mark.parametrize(
    "predict_interactions,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_predict(test_dataset, predict_interactions, interaction_log_transform):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
        predict_interactions=predict_interactions,
        interaction_log_transform=interaction_log_transform,
    )

    model.step()
    theta = model.get_model_state()

    prediction = theta.predict_viability(test_dataset)

    assert prediction.shape == (test_dataset.size,)
    if not predict_interactions:
        assert np.all(prediction >= 0.0)
        assert np.all(prediction <= 1.0)


@pytest.mark.parametrize(
    "predict_interactions,interaction_log_transform",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_variance(test_dataset, predict_interactions, interaction_log_transform):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
        predict_interactions=predict_interactions,
        interaction_log_transform=interaction_log_transform,
    )

    model.step()
    theta = model.get_model_state()
    variance = theta.predict_conditional_variance(test_dataset)

    np.all(variance >= 0.0)


def test_results_holder_accumulate(test_dataset):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
    )

    results_holder = ThetaHolder(n_thetas=2)

    while not results_holder.is_complete:
        model.step()
        sample = model.get_model_state()
        results_holder.add_theta(sample)


def test_results_holder_serde(test_dataset):
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
    )

    results_holder = ThetaHolder(n_thetas=2)

    theta = model.get_model_state()
    model.step()
    theta2 = model.get_model_state()
    results_holder.add_theta(theta)
    results_holder.add_theta(theta2)

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = ThetaHolder.load_h5(f.name)
        assert results_holder2.get_theta(0).equals(results_holder.get_theta(0))
        assert results_holder2.get_theta(1).equals(results_holder.get_theta(1))
