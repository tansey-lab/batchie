import pyro
import pyro.distributions as dist
import torch
import numpy as np
from pyro import poutine
from tqdm import trange
from scipy.special import logit, expit
from batchie.models.sparse_combo import (
    SparseDrugCombo,
    SparseDrugComboResults,
    SparseDrugComboMCMCSample,
)
from batchie.data import Screen
import torch
import matplotlib.pyplot as plt


def model(
    treatment_ids,
    sample_ids,
    y=None,
    n_drug_doses=None,
    n_samples=None,
    n_embedding_dimension=10,
):
    if y is None:
        alpha = 0.0
    else:
        alpha = y.mean()

    tau0 = pyro.sample("tau0", dist.Exponential(1))
    tau = pyro.sample("tau", dist.Gamma(1.1, 1.1))

    a = torch.tensor([2.0] + [3.0] * (n_embedding_dimension - 1))

    with pyro.plate("k", n_samples):
        upsilon0 = pyro.sample("upsilon0", dist.Normal(0, 1 / tau0))
        gamma = pyro.sample("gamma", dist.Gamma(a, 1.0).to_event(1))
        tau_t = torch.cumprod(gamma, dim=0)
        upsilon = pyro.sample("upsilon", dist.Normal(0, 1 / tau_t).to_event(1))

    phi0 = pyro.sample("phi0", dist.Exponential(torch.ones(n_drug_doses)))
    nu0 = pyro.sample("nu0", dist.Normal(0, 1 / phi0))

    with pyro.plate("i", n_drug_doses):
        phi1 = pyro.sample(
            "phi1", dist.Exponential(torch.ones(n_embedding_dimension)).to_event(1)
        )
        nu1 = pyro.sample("nu1", dist.Normal(0, 1 / phi1).to_event(1))
    with pyro.plate("i2", n_drug_doses):
        phi2 = pyro.sample(
            "phi2", dist.Exponential(torch.ones(n_embedding_dimension)).to_event(1)
        )
        nu2 = pyro.sample("nu2", dist.Normal(0, 1 / phi2).to_event(1))

    D_ijk = (
        (
            nu2[treatment_ids[:, 0], ...]
            * nu2[treatment_ids[:, 1], ...]
            * upsilon[sample_ids, ...]
        ).sum(dim=1)
        - upsilon0[sample_ids, ...]
        - alpha
    )
    mu_ik = (
        (nu1[treatment_ids[:, 0], ...] * upsilon[sample_ids, ...]).sum(dim=1)
        + nu0[treatment_ids[:, 0], ...]
        + upsilon0[sample_ids, ...]
        + alpha
    )
    mu_jk = (
        (nu1[treatment_ids[:, 1], ...] * upsilon[sample_ids, ...]).sum(dim=1)
        + nu0[treatment_ids[:, 1], ...]
        + upsilon0[sample_ids, ...]
        + alpha
    )

    return pyro.sample("y", dist.Normal((D_ijk + mu_ik + mu_jk), 1 / tau), obs=y)


def gewecke_test(
    n_iter=1000,
):
    rng = np.random.default_rng(0)
    n_data = 1000
    y = logit(rng.normal(0, 1, size=n_data))

    treatment_ids = rng.choice(
        [f"{i}" for i in range(20)], size=(n_data, 2), replace=True
    ).astype(str)
    sample_ids = rng.choice(
        [f"{i}" for i in range(10)], size=n_data, replace=True
    ).astype(str)
    doses = np.ones((n_data, 2)).astype(float)

    test_dataset = Screen(
        observations=y,
        sample_names=sample_ids,
        plate_names=np.array(["1"] * n_data, dtype=str),
        treatment_names=treatment_ids,
        treatment_doses=doses,
        control_treatment_name="control",
    )

    sparse_drug_combo = SparseDrugCombo(
        n_embedding_dimensions=10,
        n_unique_treatments=test_dataset.n_unique_treatments,
        n_unique_samples=test_dataset.n_unique_samples,
    )
    trace = poutine.trace(model).get_trace(
        torch.tensor(test_dataset.treatment_ids),
        torch.tensor(test_dataset.sample_ids),
        y=None,
        n_drug_doses=20,
        n_samples=10,
        n_embedding_dimension=10,
    )

    W0_init = trace.nodes["upsilon0"]["value"].detach().numpy()
    W_init = trace.nodes["upsilon"]["value"].detach().numpy()
    V0_init = trace.nodes["nu0"]["value"].detach().numpy()
    V1_init = trace.nodes["nu1"]["value"].detach().numpy()
    V2_init = trace.nodes["nu2"]["value"].detach().numpy()
    y_init = trace.nodes["y"]["value"].detach().numpy()
    tau_init = trace.nodes["tau"]["value"].detach().numpy()
    tau0_init = trace.nodes["tau0"]["value"].detach().numpy()
    gamma_init = trace.nodes["gamma"]["value"].detach().numpy()
    phi0_init = trace.nodes["phi0"]["value"].detach().numpy()
    phi1_init = trace.nodes["phi1"]["value"].detach().numpy()
    phi2_init = trace.nodes["phi2"]["value"].detach().numpy()

    test_dataset._observations = expit(y_init)

    theta = SparseDrugComboMCMCSample(
        W0=W0_init,
        W=W_init,
        V0=V0_init,
        V1=V1_init,
        V2=V2_init,
        alpha=y_init.mean(),
        precision=tau_init,
    )

    forward_samples = []

    for i in trange(n_iter):
        poutine.condition(
            model,
            data={
                "upsilon0": W0_init,
                "upsilon": W_init,
                # "nu0": V0_init,
                "nu1": V1_init,
                "nu2": V2_init,
                "y": y_init,
                "tau": tau_init,
                "tau0": tau0_init,
                "gamma": gamma_init,
                "phi0": phi0_init,
                "phi1": phi1_init,
                "phi2": phi2_init,
            },
        )

        trace = poutine.trace(model).get_trace(
            torch.tensor(test_dataset.treatment_ids),
            torch.tensor(test_dataset.sample_ids),
            y=None,
            n_drug_doses=test_dataset.n_unique_treatments,
            n_samples=test_dataset.n_unique_samples,
            n_embedding_dimension=10,
        )

        forward_samples.append(trace.nodes["nu0"]["value"].detach().numpy())

    gibbs_samples = []
    sparse_drug_combo.add_observations(test_dataset)
    sparse_drug_combo.set_model_state(theta)

    sparse_drug_combo.phi0 = phi0_init
    sparse_drug_combo.phi1 = phi1_init
    sparse_drug_combo.phi2 = phi2_init
    sparse_drug_combo.tau = tau_init
    sparse_drug_combo.tau0 = tau0_init
    sparse_drug_combo.gam = gamma_init

    for i in trange(n_iter):
        sparse_drug_combo._V0_step()

        gibbs_samples.append(sparse_drug_combo.get_model_state().V0)

    return forward_samples, gibbs_samples


if __name__ == "__main__":
    forward_samples, gibbs_samples = gewecke_test()

    plt.plot(
        sorted([x.mean() for x in forward_samples]),
        sorted([x.mean() for x in gibbs_samples]),
    )

    plt.savefig("/tmp/qq.pdf")
