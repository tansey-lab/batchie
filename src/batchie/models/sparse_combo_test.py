import numpy as np
import pytest
from batchie.data import Dataset

from batchie.models import sparse_combo


@pytest.fixture
def test_dataset():
    test_dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
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
        n_embedding_dimensions=5, n_unique_treatments=4, n_unique_samples=4
    )

    model.add_observations(test_dataset)

    model.mcmc_step()


def test_sparse_drug_combo_mcmc_step_without_observed_data(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.mcmc_step()


def test_predict(test_dataset):
    model = sparse_combo.SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.mcmc_step()
