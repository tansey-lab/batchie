from batchie import introspection
from batchie.core import BayesianModel


def test_create_instance():
    ins = introspection.create_instance(
        package_name="batchie",
        class_name="SparseDrugCombo",
        base_class=BayesianModel,
        kwargs={
            "n_embedding_dimensions": 5,
            "n_unique_treatments": 10,
            "n_unique_samples": 10,
        },
    )

    assert ins.n_embedding_dimensions == 5


def test_get_required_init_args_with_annotations():
    cls = introspection.get_class(
        package_name="batchie",
        class_name="SparseDrugCombo",
        base_class=BayesianModel,
    )

    args = introspection.get_required_init_args_with_annotations(cls)

    assert args == {
        "n_embedding_dimensions": int,
        "n_unique_treatments": int,
        "n_unique_samples": int,
    }
