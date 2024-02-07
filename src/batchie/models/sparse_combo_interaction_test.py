import tempfile

import numpy as np
import pytest

from batchie.core import ThetaHolder
from batchie.data import Screen, ExperimentSpace
from batchie.models import sparse_combo_interaction


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
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo_interaction.SparseDrugComboInteraction(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
    )

    model.add_observations(test_dataset)

    model.step()


def test_results_holder_serde(test_dataset):

    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = sparse_combo_interaction.SparseDrugComboInteraction(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
    )

    results_holder = ThetaHolder(2)
    theta1 = model.get_model_state()
    model.step()
    theta2 = model.get_model_state()

    results_holder.add_theta(theta1)
    results_holder.add_theta(theta2)

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = ThetaHolder.load_h5(f.name)

        assert results_holder2.get_theta(0).equals(results_holder.get_theta(0))
        assert results_holder2.get_theta(1).equals(results_holder.get_theta(1))
