import numpy as np
from batchie.models.sparse_combo import SparseDrugCombo
from batchie.datasets import Dataset


def test_sparse_drug_combo_mcmc_step_with_observed_data():
    model = SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        sample_ids=np.array([0, 1, 2, 3, 4]),
        treatments=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [-1, -1]]),
        plate_ids=np.array([0, 0, 0, 0, 0]),
    )

    model.add_observations(dataset)

    model.mcmc_step()


def test_sparse_drug_combo_mcmc_step_without_observed_data():
    model = SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.mcmc_step()
