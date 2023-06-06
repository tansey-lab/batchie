import batchie.sampling
from batchie.interfaces import BayesianModel, ResultsHolder


def test_sample(mocker):
    fake_model = mocker.MagicMock(spec=BayesianModel)
    fake_results_holder = mocker.MagicMock(spec=ResultsHolder)

    def fake_model_factory(rng):
        return fake_model

    def fake_results_holder_factory():
        return fake_results_holder

    batchie.sampling.sample(
        model_factory=fake_model_factory,
        results_factory=fake_results_holder_factory,
        seed=0,
        n_chains=1,
        chain_index=0,
        n_burnin=10,
        n_samples=10,
        thin=2,
        disable_progress_bar=True,
    )

    assert fake_model.mcmc_step.call_count == 30
    assert fake_results_holder.add_mcmc_sample.call_count == 10
