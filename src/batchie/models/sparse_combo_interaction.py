import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from batchie.data import Data
from batchie import synergy
from numpy.random import Generator
from scipy.special import logit
from collections import defaultdict
from batchie.common import CONTROL_SENTINEL_VALUE

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

    def __init__(
        self,
        single_effect_lookup: dict[tuple[int, int], float] = None,
        adjust_single: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.single_effect_lookup = single_effect_lookup if single_effect_lookup else {}
        self.adjust_single = adjust_single

    def step(self) -> None:
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

    def add_observations(self, data: Data):
        if data.n_treatments != 2:
            raise ValueError(
                "SparseDrugComboInteraction only works with two-treatments combination datasets, "
                "received a {} treatment dataset".format(data.n_treatments)
            )

        self.single_effect_lookup.update(
            synergy.create_single_treatment_effect_map(
                sample_ids=data.sample_ids,
                treatment_ids=data.treatment_ids,
                observation=data.observations,
            )
        )

        combo_mask = np.sum(data.treatment_ids == CONTROL_SENTINEL_VALUE, axis=1) == (
            data.treatment_ids.shape[1]
        )

        self.y = np.concatenate([self.y, data.observations[combo_mask]])
        self.sample_ids = np.concatenate([self.sample_ids, data.sample_ids[combo_mask]])
        self.treatment_1 = np.concatenate(
            [self.treatment_1, data.treatment_ids[combo_mask, 0]]
        )
        self.treatment_2 = np.concatenate(
            [self.treatment_2, data.treatment_ids[combo_mask, 1]]
        )

    def predict(self, data: Data):
        if not data.n_treatments == 2:
            raise ValueError(
                "SparseDrugComboInteraction only supports data sets with combinations of 2 treatments"
            )
        interaction = np.sum(
            self.W[data.sample_ids]
            * copy_array_with_control_treatments_set_to_zero(
                self.V2, data.treatment_ids[:, 0]
            )
            * copy_array_with_control_treatments_set_to_zero(
                self.V2, data.treatment_ids[:, 1]
            ),
            -1,
        )

        if self.adjust_single:
            single_effect = np.clip(
                [
                    self.single_effect_lookup[c, dd1]
                    * self.single_effect_lookup[c, dd2]
                    for c, dd1, dd2 in zip(
                        data.sample_ids,
                        data.treatment_ids[:, 0],
                        data.treatment_ids[:, 1],
                    )
                ],
                a_min=0.01,
                a_max=0.99,
            )
            if self.interaction_log_transform:
                viability = np.exp(interaction + np.log(single_effect))
            else:
                viability = interaction + single_effect
            y_logit = logit(np.clip(viability, a_min=0.01, a_max=0.99))
            return y_logit
        else:
            return interaction

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
            old_contrib = X @ self.W[sample_id]
            resid = self.y[cidx] - self.Mu[cidx] + old_contrib

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
        if self.n_obs == 0:
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
