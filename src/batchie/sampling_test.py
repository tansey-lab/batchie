import batchie.sampling
from batchie.core import BayesianModel, VIModel, MCMCModel, ThetaHolder


def test_sample_mcmc(mocker):
    class M(BayesianModel, MCMCModel):
        pass

    fake_model = mocker.Mock(spec=M)
    fake_results_holder = mocker.MagicMock(spec=ThetaHolder)

    fake_results_holder.n_thetas = 10

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
    assert fake_results_holder.add_theta.call_count == 10


def test_sample_vi(mocker):
    class M(BayesianModel, VIModel):
        pass

    fake_model = mocker.Mock(spec=M)
    fake_model.sample.return_value = list(range(10))
    fake_results_holder = mocker.MagicMock(spec=ThetaHolder)

    fake_results_holder.n_thetas = 10

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

    assert fake_model.sample.call_count == 1
    assert fake_results_holder.add_theta.call_count == 10
