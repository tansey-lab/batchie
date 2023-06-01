from batchie.models.sparse_combo import SparseDrugCombo


def test_sparse_drug_combo():
    model = SparseDrugCombo(n_dims=5, n_drugdoses=5, n_clines=5)

    model._update(y=1, cl=1, dd1=1, dd2=1)

    model.mcmc_step()
