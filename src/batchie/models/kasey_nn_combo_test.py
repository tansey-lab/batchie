import tempfile

import numpy as np
# import pytest

from batchie.core import ThetaHolder
from batchie.data import Screen, ExperimentSpace
from batchie.models import sparse_combo_interaction


# import kasey_nn_combo
# from batchie.models import kasey_nn_combo
import os
os.chdir('/athena/elementolab/scratch/ksc4004/BATCHIE/batchie_container/batchie_github/src/batchie/models/')
import kasey_nn_combo

# @pytest.fixture
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
    print("trying test1!")
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = kasey_nn_combo.Kasey_NNDrugComboInteraction(
        n_embedding_dimensions=5,
        experiment_space=experiment_space,
    )

    model.add_observations(test_dataset)
    model.train()
    model.reset_model()


def test_results_holder_serde(test_dataset):
    print("trying test2!")
    experiment_space = ExperimentSpace.from_screen(test_dataset)

    model = kasey_nn_combo.Kasey_NNDrugComboInteraction(
        experiment_space=experiment_space,
        n_embedding_dimensions=5,
    )
    model._add_observations(experiment_space)

    
    results_holder = ThetaHolder(2)
    # theta1 = model.get_model_state()
    model.train()
    theta1 = model.get_model_state()
    print(theta1)
    # theta2 = model.get_model_state()

    # results_holder.add_theta(theta1)
    # results_holder.add_theta(theta2)

    # create temporary file

    with tempfile.NamedTemporaryFile() as f:
        results_holder.save_h5(f.name)

        results_holder2 = ThetaHolder.load_h5(f.name)

        assert results_holder2.get_theta(0).equals(results_holder.get_theta(0))
        assert results_holder2.get_theta(1).equals(results_holder.get_theta(1))


if __name__ == "__main__":
    test_dataset = test_dataset()
    # test_step_with_observed_data(test_dataset)
    test_results_holder_serde(test_dataset)
    