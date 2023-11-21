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
from pyro.infer.autoguide.guides import AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import tqdm
import numpy.linalg
from batchie.retrospective import calculate_mse
from batchie.models.main import predict_avg


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

    phi0 = pyro.sample("phi0", dist.Exponential(torch.ones(n_drug_doses)).to_event(1))
    nu0 = pyro.sample("nu0", dist.Normal(0, 1 / phi0).to_event(1))

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

    return pyro.sample(
        "y", dist.Normal((D_ijk + mu_ik + mu_jk), 1 / tau).to_event(1), obs=y
    )


def main():
    screen = Screen.load_h5("/Users/jquinn/Documents/test.screen.h5")

    rng = np.random.default_rng(0)

    indices = np.arange(screen.size)

    true_indices = rng.choice(indices, size=round(screen.size * 0.80), replace=False)

    selection_vector = np.zeros(screen.size, dtype=bool)
    selection_vector[true_indices] = True

    train = screen.subset(selection_vector)
    test = screen.subset(~selection_vector)

    print(train.size, test.size)

    n_theta = 100
    sparse_drug_combo = SparseDrugCombo(
        n_embedding_dimensions=10,
        n_unique_treatments=screen.n_unique_treatments,
        n_unique_samples=screen.n_unique_samples,
    )

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    guide = AutoNormal(model)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    for step in trange(10_000):
        svi.step(
            treatment_ids=torch.tensor(train.treatment_ids),
            sample_ids=torch.tensor(train.sample_ids),
            y=torch.tensor(logit(train.observations)),
            n_drug_doses=train.n_unique_treatments,
            n_samples=train.n_unique_samples,
        )

    W0s = dist.Normal(
        pyro.param("AutoNormal.locs.upsilon0").detach(),
        pyro.param("AutoNormal.scales.upsilon0").detach(),
    ).sample((n_theta,))

    Ws = dist.Normal(
        pyro.param("AutoNormal.locs.upsilon").detach(),
        pyro.param("AutoNormal.scales.upsilon").detach(),
    ).sample((n_theta,))

    V0s = dist.Normal(
        pyro.param("AutoNormal.locs.nu0").detach(),
        pyro.param("AutoNormal.scales.nu0").detach(),
    ).sample((n_theta,))

    V1s = dist.Normal(
        pyro.param("AutoNormal.locs.nu1").detach(),
        pyro.param("AutoNormal.scales.nu1").detach(),
    ).sample((n_theta,))

    V2s = dist.Normal(
        pyro.param("AutoNormal.locs.nu2").detach(),
        pyro.param("AutoNormal.scales.nu2").detach(),
    ).sample((n_theta,))

    precisions = dist.Normal(
        pyro.param("AutoNormal.locs.tau").detach(),
        pyro.param("AutoNormal.scales.tau").detach(),
    ).sample((n_theta,))

    theta_holder = sparse_drug_combo.get_results_holder(n_theta)

    for i in trange(n_theta):
        theta = SparseDrugComboMCMCSample(
            W0=W0s[i, ...],
            W=Ws[i, ...],
            V0=V0s[i, ...],
            V1=V1s[i, ...],
            V2=V2s[i, ...],
            alpha=0.0,
            precision=precisions[i, ...],
        )
        theta_holder.add_theta(theta, 1 / precisions[i, ...])

    result = calculate_mse(
        observed_screen=test,
        model=sparse_drug_combo,
        thetas=theta_holder,
    )

    preds = predict_avg(
        screen=test,
        model=sparse_drug_combo,
        thetas=theta_holder,
    )
    print(result)

    plt.scatter(test.observations, preds)
    plt.savefig("/tmp/obs_preds_scatter.pdf")
    print("done")


if __name__ == "__main__":
    main()
