import batchie.sampling
from batchie.core import BayesianModel, ModelParamsHolder


def test_sample(mocker):
    fake_model = mocker.MagicMock(spec=BayesianModel)
    fake_results_holder = mocker.MagicMock(spec=ModelParamsHolder)

    fake_results_holder.n_samples = 10

    batchie.sampling.sample(
        model=fake_model,
        results=fake_results_holder,
        seed=0,
        n_chains=1,
        chain_index=0,
        n_burnin=10,
        thin=2,
        progress_bar=False,
    )

    assert fake_model.step.call_count == 30
    assert fake_results_holder.add_sample.call_count == 10
