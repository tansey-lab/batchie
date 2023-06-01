from copy import deepcopy
import numpy as np
from tqdm import trange
from batchie.datasets import Plate

BayesianModel, Predictor,

from common import ArrayType


class Sampler:
    def __init__(
        self,
        model: BayesianModel,
        thin: int,
        burnin: int,
        seed: int,
        verbose: bool = True,
        rng=None,
    ):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.model = deepcopy(model)
        self.thin = thin
        self.burnin = burnin
        self.disable_progress_bar = not verbose

    def sample(self, n: int) -> list[Predictor]:
        ## OPTIONAL: reset the model before sampling
        self.model.reset_model()

        for _ in trange(self.burnin, disable=self.disable_progress_bar):
            self.model.mcmc_step()

        predictors = []
        total_steps = n * self.thin
        for s in trange(total_steps, disable=self.disable_progress_bar):
            self.model.mcmc_step()
            if ((s + 1) % self.thin) == 0:
                predictors.append(self.model.predictor())
        return predictors

    def update(self, plate: Plate, y: ArrayType) -> None:
        self.model.update(plate, y)

    def update_list(self, plates: list[Plate], ys: list[ArrayType]) -> None:
        self.model.update_list(plates, ys)
