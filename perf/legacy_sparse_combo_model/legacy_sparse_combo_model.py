from batchie.models.sparse_combo_legacy import (
    LegacySparseDrugCombo,
    SparseDrugComboResults,
)
from batchie.data import Screen
import numpy as np

from batchie import sampling
from batchie.retrospective import calculate_mse


def generate_data(
    n_obs=10_000,
    n_unique_treatments=5,
    n_unique_samples=5,
    n_unique_doses=5,
):
    unique_sample_names = np.array([f"sample_{x}" for x in range(n_unique_samples)])
    unique_plate_names = np.array([f"plate_{x}" for x in range(n_unique_samples)])
    unique_treatment_names = np.array(
        [f"treatment_{x}" for x in range(n_unique_treatments)]
    )

    unique_doses = np.linspace(1, 1e-6, n_unique_doses)

    rng = np.random.default_rng(0)
    sample_names = rng.choice(unique_sample_names, size=n_obs, replace=True)
    plate_names = rng.choice(unique_plate_names, size=n_obs, replace=True)

    treatment_names = []
    doses = []

    for i in range(n_obs):
        treatment_names.append(
            rng.choice(unique_treatment_names, size=2, replace=False)
        )
        doses.append(rng.choice(unique_doses, size=2, replace=False))

    return Screen(
        observations=np.clip(rng.normal(size=n_obs), 0.01, 0.99),
        sample_names=sample_names,
        plate_names=plate_names,
        treatment_names=np.array(treatment_names),
        treatment_doses=np.array(doses),
    )


def run_benchmark():
    data = Screen.load_h5("/Users/jquinn/Documents/test.screen.h5")

    print("Generated {} observations".format(data.size))
    print("Unique samples: {}".format(data.n_unique_samples))
    print("Unique treatments: {}".format(data.n_unique_treatments))

    model = LegacySparseDrugCombo(
        n_embedding_dimensions=12,
        n_unique_treatments=data.n_unique_treatments,
        n_unique_samples=data.n_unique_samples,
    )

    model.add_observations(data)

    results = SparseDrugComboResults(
        n_unique_samples=data.n_unique_samples,
        n_unique_treatments=data.n_unique_treatments,
        n_embedding_dimensions=12,
        n_thetas=100,
    )

    res = sampling.sample(
        model=model,
        results=results,
        seed=0,
        n_chains=1,
        chain_index=0,
        n_burnin=1000,
        thin=2,
        progress_bar=True,
    )

    mse = calculate_mse(model=model, observed_screen=data, thetas=res)

    print(mse)
    print(res.get_theta(0))


if __name__ == "__main__":
    run_benchmark()
