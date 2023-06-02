import numpy as np
from batchie.models.sparse_combo import SparseDrugCombo


def test_sparse_drug_combo():
    model = SparseDrugCombo(n_embedding_dimensions=5, n_doses=5, n_samples=5)

    model.update(
        y=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        cline=np.array([0, 1, 2, 3, 4]),
        dd1=np.array([0, 0, 0, 0, -1]),
        dd2=np.array([1, 1, 1, 1, -1]),
    )

    model.mcmc_step()
