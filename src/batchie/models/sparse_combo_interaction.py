import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from batchie.data import Data
from numpy.random import Generator
from scipy.special import logit
from collections import defaultdict

from batchie.common import ArrayType, copy_array_with_control_treatments_set_to_zero
from batchie.fast_mvn import sample_mvn_from_precision
from batchie.core import BayesianModel, BayesianModelSample, SamplesHolder
from batchie.models.sparse_combo import SparseDrugCombo

logger = logging.getLogger(__name__)


@dataclass
class SparseDrugComboInteractionMCMCSample(BayesianModelSample):
    """A single sample from the MCMC chain for the sparse drug combo model"""

    W: ArrayType
    V2: ArrayType
    precision: float


class SparseDrugComboInteraction(SparseDrugCombo):
    """
    Subset of SparseDrugCombo that only models the interaction terms.
    """

    def __init__(self, single_effect_lookup: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_effect_lookup = single_effect_lookup

    def step(self) -> None:
        self.num_mcmc_steps += 1
        # Note! all thse reconstruct Mu's can be changed inplace
        # when updating the parameter, currently not efficiient
        self._reconstruct_Mu(clip=False)
        self._W_step()
        self._V2_step()
        self._prec_obs_step()
        self._prec_V2_step()
        self._prec_W_step()

    def get_model_state(self) -> SparseDrugComboInteractionMCMCSample:
        return SparseDrugComboInteractionMCMCSample(
            precision=self.precision,
            W=self.W.copy(),
            V2=self.V2.copy(),
        )

    def set_model_state(self, model_state: SparseDrugComboInteractionMCMCSample):
        self.W = model_state.W
        self.V2 = model_state.V2
        self.precision = model_state.precision
        self._reconstruct_Mu()

    # region Model Implementation

    def _W_step(self):
        """the strategy is to iterate over each cell line
        and solve the linear problem"""
        for sample_id in range(self.n_unique_samples):
            cidx = self.sample_ids == sample_id
            if cidx.sum() == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.tau)
                self.W[sample_id] = self.rng.normal(0.0, stddev)
                continue

            X = self.V2[cidx] * self.V2[cidx]
            old_contrib = X @ self.W[c]
            resid = y[cidx] - self.Mu[cidx] + old_contrib

            Xt = X.transpose()
            prec = self.precision
            mu_part = (Xt @ resid) * prec
            Q = (Xt @ X) * prec
            Q[np.diag_indices(self.n_embedding_dimensions)] += self.tau
            try:
                self.W[sample_id] = sample_mvn_from_precision(Q, mu_part=mu_part)

                # update Mu
                self.Mu[cidx] += X @ self.W[sample_id] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs W-step...")

    def _reconstruct_Mu(self, clip=True):
        if self.n_obs() == 0:
            return

        interaction = np.sum(
            (
                self.W[self.sample_ids]
                * copy_array_with_control_treatments_set_to_zero(
                    self.V2, self.treatment_1
                )
                * copy_array_with_control_treatments_set_to_zero(
                    self.V2, self.treatment_2
                )
            ),
            -1,
        )

        self.Mu = interaction

        if clip:
            self.Mu = np.clip(self.Mu, self.min_Mu, self.max_Mu)

    # endregion
