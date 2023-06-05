import batchie.sampling
from batchie.interfaces import BayesianModel


def test_sample(mocker):
    fake_model = mocker.MagicMock(spec=BayesianModel)

    def fake_model_factory(rng):
        return fake_model

    result = batchie.sampling.sample(
        model_factory=fake_model_factory,
        seed=0,
        n_chains=1,
        chain_index=0,
        n_burnin=10,
        n_samples=10,
        thin=2,
        disable_progress_bar=True,
    )

    assert len(result) == 10
    assert fake_model.mcmc_step.call_count == 30
