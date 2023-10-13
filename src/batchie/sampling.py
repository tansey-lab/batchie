from typing import Callable

import numpy.random
from tqdm import trange

from batchie.interfaces import BayesianModel, ResultsHolder


def sample(
    model_factory: Callable[[numpy.random.BitGenerator], BayesianModel],
    results_factory: Callable[[], ResultsHolder],
    seed: int,
    n_chains: int,
    chain_index: int,
    n_burnin: int,
    n_samples: int,
    thin: int,
    disable_progress_bar=False,
) -> ResultsHolder:
    seeds = numpy.random.SeedSequence(seed).spawn(n_chains)
    rng = numpy.random.default_rng(seeds[chain_index])

    model = model_factory(rng)

    results = results_factory()

    for _ in trange(n_burnin, disable=disable_progress_bar):
        model.mcmc_step()

    total_steps = n_samples * thin
    for step_index in trange(total_steps, disable=disable_progress_bar):
        model.mcmc_step()
        if ((step_index + 1) % thin) == 0:
            results.add_mcmc_sample(model.get_model_state())

    return results
