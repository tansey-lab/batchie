from batchie import introspection
from batchie.core import BayesianModel, ExperimentSpace


def test_create_instance(mocker):
    experiment_space = mocker.Mock(spec=ExperimentSpace)
    experiment_space.n_unique_treatments = 5
    experiment_space.n_unique_samples = 10

    ins = introspection.create_instance(
        package_name="batchie",
        class_name="SparseDrugCombo",
        base_class=BayesianModel,
        kwargs={
            "n_embedding_dimensions": 5,
            "experiment_space": experiment_space,
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
        "experiment_space": ExperimentSpace,
    }
