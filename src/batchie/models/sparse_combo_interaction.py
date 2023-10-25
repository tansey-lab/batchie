import logging
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
from batchie import synergy
from batchie.common import ArrayType, copy_array_with_control_treatments_set_to_zero
from batchie.common import CONTROL_SENTINEL_VALUE
from batchie.core import Theta, ThetaHolder
from batchie.data import ExperimentBase
from batchie.fast_mvn import sample_mvn_from_precision
from batchie.models.sparse_combo import SparseDrugCombo
from scipy.special import logit

logger = logging.getLogger(__name__)


@dataclass
class SparseDrugComboInteractionMCMCSample(Theta):
    """A single sample from the MCMC chain for the sparse drug combo model"""

    W: ArrayType
    V2: ArrayType
    precision: float


class SparseDrugComboInteractionResults(ThetaHolder):
    def __init__(
        self,
        n_unique_samples: int,
        n_unique_treatments: int,
        n_embedding_dimensions: int,
        n_thetas: int,
    ):
        super().__init__(n_thetas)
        self.n_unique_samples = n_unique_samples
        self.n_unique_treatments = n_unique_treatments
        self.n_embedding_dimensions = n_embedding_dimensions

        self.V2 = np.zeros(
            (n_thetas, self.n_unique_treatments, self.n_embedding_dimensions),
            dtype=np.float32,
        )
        self.W = np.zeros(
            (n_thetas, self.n_unique_samples, self.n_embedding_dimensions),
            dtype=np.float32,
        )

        self.alpha = np.zeros((n_thetas,), np.float32)
        self.precision = np.zeros((n_thetas,), np.float32)

    def get_theta(self, step_index: int) -> SparseDrugComboInteractionMCMCSample:
        """Get one sample from the MCMC chain"""
        # Test if this is beyond the step we are current at with the cursor
        if step_index >= self._cursor:
            raise ValueError("Cannot get a step beyond the current cursor position")

        return SparseDrugComboInteractionMCMCSample(
            W=self.W[step_index],
            V2=self.V2[step_index],
            precision=self.precision[step_index],
        )

    def get_variance(self, step_index: int) -> float:
        return 1.0 / self.precision[step_index]

    def _save_theta(
        self,
        sample: SparseDrugComboInteractionMCMCSample,
        variance: float,
        sample_index: int,
    ):
        self.V2[sample_index] = sample.V2
        self.W[sample_index] = sample.W
        self.precision[sample_index] = sample.precision

    def combine(self, other):
        if type(self) != type(other):
            raise ValueError("Cannot combine with different type")

        if self.n_embedding_dimensions != other.n_embedding_dimensions:
            raise ValueError("Cannot combine with different embedding dimensions")

        if self.n_unique_samples != other.n_unique_samples:
            raise ValueError("Cannot combine with different number of unique samples")

        if self.n_unique_treatments != other.n_unique_treatments:
            raise ValueError(
                "Cannot combine with different number of unique treatments"
            )

        output = SparseDrugComboInteractionResults(
            n_unique_samples=self.n_unique_samples,
            n_unique_treatments=self.n_unique_treatments,
            n_embedding_dimensions=self.n_embedding_dimensions,
            n_thetas=self.n_thetas + other.n_thetas,
        )

        for i in range(self.n_thetas):
            sample = self.get_theta(i)
            variance = self.get_variance(i)
            output.add_theta(sample, variance)

        for i in range(other.n_thetas):
            sample = other.get_theta(i)
            variance = other.get_variance(i)
            output.add_theta(sample, variance)

        return output

    def save_h5(self, fn: str):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset("V2", data=self.V2)
            f.create_dataset("W", data=self.W)
            f.create_dataset("alpha", data=self.alpha)
            f.create_dataset("precision", data=self.precision)

            # Save the cursor value metadata
            f.attrs["cursor"] = self._cursor

    @staticmethod
    def load_h5(path: str):
        """Load saved data from h5 archive"""
        with h5py.File(path, "r") as f:
            n_unique_samples = f["W"].shape[1]
            n_unique_treatments = f["V2"].shape[1]
            n_embedding_dimensions = f["W"].shape[2]
            n_samples = f["W"].shape[0]

            results = SparseDrugComboInteractionResults(
                n_unique_samples=n_unique_samples,
                n_unique_treatments=n_unique_treatments,
                n_embedding_dimensions=n_embedding_dimensions,
                n_thetas=n_samples,
            )

            results.V2 = f["V2"][:]
            results.W = f["W"][:]
            results.precision = f["precision"][:]
            results._cursor = f.attrs["cursor"]

        return results


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

    def get_results_holder(self, n_samples: int):
        return SparseDrugComboInteractionResults(
            n_unique_samples=self.n_unique_samples,
            n_unique_treatments=self.n_unique_treatments,
            n_embedding_dimensions=self.n_embedding_dimensions,
            n_thetas=n_samples,
        )

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

    def add_observations(self, data: ExperimentBase):
        if data.treatment_arity != 2:
            raise ValueError(
                "SparseDrugComboInteraction only works with two-treatments combination datasets, "
                "received a {} treatment dataset".format(data.treatment_arity)
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

    def predict(self, data: ExperimentBase):
        if not data.treatment_arity == 2:
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
                self.W[sample_id] = sample_mvn_from_precision(
                    Q, mu_part=mu_part, rng=self.rng
                )

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
