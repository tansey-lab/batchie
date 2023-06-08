import numpy as np
from batchie.data import Dataset

from batchie.models.sparse_combo import SparseDrugCombo


def test_sparse_drug_combo_mcmc_step_with_observed_data():
    model = SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=4, n_unique_samples=4
    )

    dataset = Dataset(
        observations=np.array([0.1, 0.2, 0.3, 0.4]),
        sample_names=np.array(["a", "b", "c", "d"], dtype=str),
        plate_names=np.array(["a", "a", "b", "b"], dtype=str),
        treatment_names=np.array(
            [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]], dtype=str
        ),
        treatment_doses=np.array([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0]]),
    )

    model.add_observations(dataset)

    model.mcmc_step()


def test_sparse_drug_combo_mcmc_step_without_observed_data():
    model = SparseDrugCombo(
        n_embedding_dimensions=5, n_unique_treatments=5, n_unique_samples=5
    )

    model.mcmc_step()
