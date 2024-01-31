import logging

import numpy.random
from tqdm import trange

from batchie.core import ThetaHolder, MCMCModel, VIModel

logger = logging.getLogger(__name__)


def sample(
    model,
    results: ThetaHolder,
    seed: int,
    n_chains: int = None,
    chain_index: int = None,
    n_burnin: int = None,
    thin: int = None,
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
    match model:
        case MCMCModel():
            if n_chains is None:
                raise ValueError("n_chains must be set when model is MCMCModel")
            if chain_index is None:
                raise ValueError("chain_index must be set when model is MCMCModel")
            if n_burnin is None:
                raise ValueError("n_burnin must be set when model is MCMCModel")
            if thin is None:
                raise ValueError("thin must be set when model is MCMCModel")

            seeds = numpy.random.SeedSequence(seed).spawn(n_chains)
            rng = numpy.random.default_rng(seeds[chain_index])

            model.reset_model()
            model.set_rng(rng)

            logger.info(
                "Will run {} total iterations on model {}".format(
                    results.n_thetas * thin + n_burnin, model
                )
            )

            for _ in trange(n_burnin, disable=not progress_bar):
                model.step()

            total_steps = results.n_thetas * thin
            for step_index in trange(total_steps, disable=not progress_bar):
                model.step()
                if ((step_index + 1) % thin) == 0:
                    results.add_theta(model.get_model_state())
        case VIModel():
            model.reset_model()
            rng = numpy.random.default_rng(seed)
            model.set_rng(rng)
            samples = model.sample(num_samples=results.n_thetas)
            for theta in samples:
                results.add_theta(theta)
        case other:
            raise ValueError(f"model must be one of MCMCModel, VIModel, got {other}")

    return results
