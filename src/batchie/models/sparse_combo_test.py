from batchie.models.sparse_combo import SparseDrugCombo


def test_sparse_drug_combo():
    model = SparseDrugCombo(n_embedding_dimensions=5, n_doses=5, n_samples=5)

    model._update(y=1, cl=1, dd1=1, dd2=1)

    model.mcmc_step()
