import numpy.random
from batchie.core import BayesianModel, ThetaHolder
from tqdm import trange


def sample(
    model: BayesianModel,
    results: ThetaHolder,
    seed: int,
    n_chains: int,
    chain_index: int,
    n_burnin: int,
    thin: int,
    progress_bar=False,
) -> ThetaHolder:
    seeds = numpy.random.SeedSequence(seed).spawn(n_chains)
    rng = numpy.random.default_rng(seeds[chain_index])

    model.reset_model()
    model.set_rng(rng)

    for _ in trange(n_burnin, disable=not progress_bar):
        model.step()

    total_steps = results.n_thetas * thin
    for step_index in trange(total_steps, disable=not progress_bar):
        model.step()
        if ((step_index + 1) % thin) == 0:
            results.add_theta(model.get_model_state(), model.variance())

    return results
