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
    """
    Sample from the model posterior using the given parameters.

    :param model: The model which will be sampled from.
    :param results: The object which will store the results
    :param seed: The seed to use for the random number generator
    :param n_chains: The number of parallel chains to run
    :param chain_index: The index of the current chain
    :param n_burnin: The number of burnin steps to run
    :param thin: The thinning factor
    :param progress_bar: Whether to display a progress bar
    :return: a :py:class:`ThetaHolder` containing the sampled parameters
    """
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
