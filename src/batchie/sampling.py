import numpy.random
from tqdm import trange

from batchie.core import BayesianModel, SamplesHolder


def sample(
    model: BayesianModel,
    results: SamplesHolder,
    seed: int,
    n_chains: int,
    chain_index: int,
    n_burnin: int,
    thin: int,
    disable_progress_bar=False,
) -> SamplesHolder:
    seeds = numpy.random.SeedSequence(seed).spawn(n_chains)
    rng = numpy.random.default_rng(seeds[chain_index])

    model.reset_model()
    model.set_rng(rng)

    for _ in trange(n_burnin, disable=disable_progress_bar):
        model.step()

    total_steps = results.n_samples * thin
    for step_index in trange(total_steps, disable=disable_progress_bar):
        model.step()
        if ((step_index + 1) % thin) == 0:
            results.add_sample(model.get_model_state(), model.variance())

    return results
