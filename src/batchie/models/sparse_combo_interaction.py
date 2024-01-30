import logging
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import h5py
from numpy.random import Generator
from scipy.special import logit

from batchie.common import (
    ArrayType,
    FloatingPointType,
    CONTROL_SENTINEL_VALUE,
    copy_array_with_control_treatments_set_to_zero,
)
from batchie.core import BayesianModel, Theta, ThetaHolder
from batchie.data import ScreenBase, create_single_treatment_effect_map
from batchie.fast_mvn import sample_mvn_from_precision
from dataclasses import dataclass

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
            precision=self.precision[step_index].item(),
        )

    def _save_theta(
        self,
        sample: SparseDrugComboInteractionMCMCSample,
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
            output.add_theta(sample)

        for i in range(other.n_thetas):
            sample = other.get_theta(i)
            output.add_theta(sample)

        return output

    def save_h5(self, fn: str):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset("V2", data=self.V2, compression="gzip")
            f.create_dataset("W", data=self.W, compression="gzip")
            f.create_dataset("alpha", data=self.alpha, compression="gzip")
            f.create_dataset("precision", data=self.precision, compression="gzip")

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


class LegacySparseDrugComboInteractionImpl:
    """
    This is the original implementation of the sparse drug combo interaction model.
    Preserved here without changes to ensure reproducibility of results.
    """

    ### Assumes that we are only modeling interaction terms
    def __init__(
        self,
        n_dims: int,  # embedding dimension
        n_drugdoses: int,  # number of total drug/doses
        n_clines: int,  # number of total cell lines
        mult_gamma_proc: bool = True,
        local_shrinkage: bool = True,
        a0: float = 1.1,  # gamma prior hyperparmas for all precisions
        b0: float = 1.1,
        min_Mu: float = -10.0,
        max_Mu: float = 10.0,
    ):
        self.D = n_dims  # embedding size
        self.n_drugdoses = n_drugdoses
        self.n_clines = n_clines
        self.min_Mu = min_Mu
        self.max_Mu = max_Mu

        # for the next two see BHATTACHARYA and DUNSON (2011)
        self.local_shrinkage = local_shrinkage
        self.mult_gamma_proc = mult_gamma_proc
        # self.dummy_last = int(has_controls)

        # data holders
        self.y = []
        self.num_mcmc_steps = 0

        # indicators for each entry AND sparse query of speficic combos
        self.cline = []
        self.dd1 = []
        self.dd2 = []
        self.cline_idxs = defaultdict(list)
        self.dd1_idxs = defaultdict(list)
        self.dd2_idxs = defaultdict(list)

        # hyperpriors
        self.a0 = a0  # inverse gamma param 1 for prec
        self.b0 = b0  # inverse gamma param 2 for prec

        # strategy
        # cell line interaction matrices have shrinking gamam process variances
        # for each gamma process we need phi and tau
        # drug interaction terms have sparsity
        # for sparsity term need only local shrinkage since global shrinkage is
        # determined by tau
        # V<k> means drug-dose embedding for order k interactions
        # W<k> means cell line embedding for order k interactions

        sh = (self.n_drugdoses, self.D)
        # Careful! If changing the inialization make sure the last entry of the
        # V's stay at zero becuase it's used for single controls.
        self.V2 = np.zeros(sh, dtype=np.float32)

        sh = (self.n_clines, self.D)
        self.W = np.zeros(sh, dtype=np.float32)

        # paramaters for hroseshow priors
        self.phi2 = 100.0 * np.ones_like(self.V2)
        self.eta2 = np.ones(self.D, dtype=np.float32)

        # shrinkage
        self.tau = 100.0 * np.ones(self.D, np.float32)
        self.tau0 = 100.0

        if mult_gamma_proc:
            self.gam = np.ones(self.D, np.float32)

        # intercept and overall precision
        self.prec = 100.0

        # holder for model prediction during fit and for eval
        self.Mu = np.zeros(0, np.float32)

    def n_obs(self):
        return len(self.y)

    def _update(self, y, cl, dd1, dd2):
        n = self.n_obs()
        self.y.append(y)
        self.cline.append(cl)
        self.dd1.append(dd1)
        self.dd2.append(dd2)
        self.cline_idxs[cl].append(n)
        self.dd1_idxs[dd1].append(n)
        self.dd2_idxs[dd2].append(n)

    def encode_obs(self):
        y = np.array(self.y, copy=False)
        cline = np.array(self.cline, copy=False)
        dd1 = np.array(self.dd1, copy=False)
        dd2 = np.array(self.dd2, copy=False)
        return (y, cline, dd1, dd2)

    def reset_model(self):
        self.W = self.W * 0.0
        self.V2 = self.V2 * 0.0
        self.prec = 100.0
        self.Mu = np.zeros(0, np.float32)

    def _W_step(self) -> None:
        """the strategy is to iterate over each cell line
        and solve the linear problem"""
        y, _, dd1, dd2 = self.encode_obs()
        for c in range(self.n_clines):
            cidx = np.array(self.cline_idxs[c], copy=False)
            if len(cidx) == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.tau)
                self.W[c] = np.random.normal(0.0, stddev)
                continue

            X = self.V2[dd1[cidx]] * self.V2[dd2[cidx]]
            old_contrib = X @ self.W[c]
            resid = y[cidx] - self.Mu[cidx] + old_contrib

            Xt = X.transpose()
            prec = self.prec
            mu_part = (Xt @ resid) * prec
            Q = (Xt @ X) * prec
            Q[np.diag_indices(self.D)] += self.tau
            try:
                self.W[c] = sample_mvn_from_precision(Q, mu_part=mu_part)

                # update Mu
                self.Mu[cidx] += X @ self.W[c] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs W-step...")

    def _V2_step(self) -> None:
        """the strategy is to iterate over each drug pair combination
        but the trick is to handle the case when drug-pair appears in first
        postition and when it appears in second position
        and solve the linear problem"""
        y, cline, dd1, dd2 = self.encode_obs()
        for m in range(self.n_drugdoses):
            # slice where m, k appears as drug1
            idx1 = np.array(self.dd1_idxs[m], copy=False, dtype=int)
            # slice where m, k appears as drug2
            idx2 = np.array(self.dd2_idxs[m], copy=False, dtype=int)

            if len(idx1) + len(idx2) == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi2[m] * self.eta2)
                self.V2[m] = np.random.normal(0, stddev)
                continue

            if len(idx1) == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(0, self.D)
            else:
                X1 = self.W[cline[idx1]] * self.V2[dd2[idx1]]
                old_contrib1 = X1 @ self.V2[m]
                resid1 = y[idx1] - self.Mu[idx1] + old_contrib1

            if len(idx2) == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(0, self.D)
            else:
                X2 = self.W[cline[idx2]] * self.V2[dd1[idx2]]
                old_contrib2 = X2 @ self.V2[m]
                resid2 = y[idx2] - self.Mu[idx2] + old_contrib2

            # combine slices of data
            X = np.concatenate([X1, X2])
            resid = np.concatenate([resid1, resid2])
            # resid = np.clip(resid, -10.0, 10.0)
            old_contrib = np.concatenate([old_contrib1, old_contrib2])
            idx = np.concatenate([idx1, idx2])

            # sample form posterior
            # X = np.clip(X, -10.0, 10.0)
            Xt = X.transpose()
            mu_part = (Xt @ resid) * self.prec
            Q = (Xt @ X) * self.prec
            dix = np.diag_indices(self.D)
            Q[dix] += self.phi2[m] * self.eta2
            try:
                self.V2[m] = sample_mvn_from_precision(Q, mu_part=mu_part)
                # self.V2[m] = np.clip(self.V2[m], -10.0, 10.0)
                self.Mu[idx] += X @ self.V2[m] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs V-step...")

    def _prec_obs_step(self) -> None:
        if self.n_obs() == 0:
            self.prec = np.random.gamma(self.a0, 1.0 / self.b0)
            return
        # self._reconstruct_Mu()
        sse = np.square(self.y - self.Mu).sum()
        an = self.a0 + 0.5 * self.n_obs()
        bn = self.b0 + 0.5 * sse
        self.prec = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.last_rmse = np.sqrt(np.square(self.y - self.Mu).mean())
        self.prec = np.clip(self.prec, C, 1e6)

    def _prec_V2_step(self):
        if self.local_shrinkage:
            phiaux2 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi2))
            bn = phiaux2 + 0.5 * self.eta2 * self.V2**2
            self.phi2 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array([len(self.dd1_idxs[c]) for c in range(self.n_drugdoses)])
            N2 = np.array([len(self.dd2_idxs[c]) for c in range(self.n_drugdoses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi2 = np.clip(self.phi2, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_drugdoses)
        etaaux2 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta2))
        bn = etaaux2 + 0.5 * (self.phi2 * self.V2**2).sum(0)
        self.eta2 = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.eta2 = np.clip(self.eta2, C, 1e6)

    def _prec_W_step(self):
        # gamma process
        parssq = self.W**2
        if self.mult_gamma_proc:
            tmp = np.cumprod(self.gam) / self.gam[0]
            an = 2 + 0.5 * self.n_clines * self.D
            bn = 1 + 0.5 * (tmp * parssq).sum()
            self.gam[0] = np.random.gamma(an, 1.0 / (bn + 1e-3))
            # self.gam[0] = np.clip(self.gam[0], 1.0, 10000.0)
            # sample all others
            for d in range(1, self.D):
                tmp = np.cumprod(self.gam)[d:] / self.gam[d]
                an = 3 + 0.5 * self.n_clines * (self.D - d)
                bn = 1 + 0.5 * (tmp * parssq[:, d:]).sum()
                self.gam[d] = np.random.gamma(an, 1.0 / (bn + 1e-3))
                # self.gam[d] = np.clip(self.gam[d], 0.5, 100.0)
            self.tau = np.cumprod(self.gam)
        else:
            an = self.a0 + 0.5 * self.n_clines
            bn = self.b0 + 0.5 * parssq.sum(0)
            self.tau = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.tau = np.clip(self.tau, C, 1e6)

    def _prec_W0_step(self):
        # W0 precision
        an = self.a0 + 0.5 * self.n_clines
        bn = self.b0 + 0.5 * (self.W0**2).sum()
        self.tau0 = np.random.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.tau0 = np.clip(self.tau0, C, 1e6)

    def mcmc_step(self) -> None:
        self.num_mcmc_steps += 1
        # Note! all thse reconstruct Mu's can be changed inplace
        # when updating the parameter, currently not efficiient
        self._reconstruct_Mu(clip=False)
        self._W_step()
        self._V2_step()
        self._prec_obs_step()
        self._prec_V2_step()
        self._prec_W_step()

    def _reconstruct_Mu(self, clip=True):
        if self.n_obs() == 0:
            return
        # Reconstruct the entire matrix of means, excluding the upper triangular portion
        _, cline, dd1, dd2 = self.encode_obs()
        self.Mu = np.sum(self.W[cline] * self.V2[dd1] * self.V2[dd2], axis=-1)

        if clip:
            self.Mu = np.clip(self.Mu, self.min_Mu, self.max_Mu)

    def predict(self, cline: np.ndarray, dd1: np.ndarray, dd2: np.ndarray):
        Mu = np.sum(self.W[cline] * self.V2[dd1] * self.V2[dd2], axis=-1)
        return Mu


class SparseDrugComboInteraction(BayesianModel):
    def __init__(
        self,
        n_embedding_dimensions: int,  # embedding dimension
        n_unique_treatments: int,  # number of total drug/doses
        n_unique_samples: int,  # number of total cell lines
        mult_gamma_proc: bool = True,
        local_shrinkage: bool = True,
        a0: float = 1.1,  # gamma prior hyperparmas for all precisions
        b0: float = 1.1,
        min_Mu: float = -10.0,
        max_Mu: float = 10.0,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_unique_treatments = n_unique_treatments
        self.n_unique_samples = n_unique_samples
        self.single_effect_lookup = {}
        self.wrapped_model = LegacySparseDrugComboInteractionImpl(
            n_dims=n_embedding_dimensions,
            n_drugdoses=n_unique_treatments,
            n_clines=n_unique_samples,
            mult_gamma_proc=mult_gamma_proc,
            local_shrinkage=local_shrinkage,
            a0=a0,
            b0=b0,
            min_Mu=min_Mu,
            max_Mu=max_Mu,
        )

    def set_model_state(self, theta: SparseDrugComboInteractionMCMCSample):
        self.wrapped_model.reset_model()
        self.wrapped_model.W = theta.W.astype(np.float32)
        self.wrapped_model.V2 = theta.V2.astype(np.float32)
        self.wrapped_model._reconstruct_Mu()

    def get_model_state(self) -> SparseDrugComboInteractionMCMCSample:
        return SparseDrugComboInteractionMCMCSample(
            precision=self.wrapped_model.prec,
            W=self.wrapped_model.W.copy().astype(FloatingPointType),
            V2=self.wrapped_model.V2.copy().astype(FloatingPointType),
        )

    def predict(self, data: ScreenBase) -> ArrayType:
        if not data.treatment_arity == 2:
            raise ValueError(
                "SparseDrugComboInteraction only supports data sets with combinations of 2 treatments"
            )
        interaction = np.sum(
            self.wrapped_model.W[data.sample_ids]
            * copy_array_with_control_treatments_set_to_zero(
                self.wrapped_model.V2, data.treatment_ids[:, 0]
            )
            * copy_array_with_control_treatments_set_to_zero(
                self.wrapped_model.V2, data.treatment_ids[:, 1]
            ),
            -1,
        )

        single_effect = np.clip(
            [
                self.single_effect_lookup[c, dd1] * self.single_effect_lookup[c, dd2]
                for c, dd1, dd2 in zip(
                    data.sample_ids,
                    data.treatment_ids[:, 0],
                    data.treatment_ids[:, 1],
                )
            ],
            a_min=0.01,
            a_max=0.99,
        )
        viability = np.exp(interaction + np.log(single_effect))
        return np.clip(viability, a_min=0.01, a_max=0.99)

    def variance(self, data: ScreenBase) -> FloatingPointType:
        return 1.0 / self.wrapped_model.prec

    def step(self):
        self.wrapped_model.mcmc_step()

    def set_rng(self, rng: np.random.Generator):
        self._rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def _add_observations(self, data: ScreenBase):
        if data.treatment_arity != 2:
            raise ValueError(
                "SparseDrugComboInteraction only works with two-treatments combination datasets, "
                "received a {} treatment dataset".format(data.treatment_arity)
            )

        self.single_effect_lookup.update(
            create_single_treatment_effect_map(
                sample_ids=data.sample_ids,
                treatment_ids=data.treatment_ids,
                observation=data.observations,
            )
        )

        combo_mask = np.sum(data.treatment_ids == CONTROL_SENTINEL_VALUE, axis=1) == (
            data.treatment_ids.shape[1]
        )

        obs = data.observations[combo_mask]
        cls = data.sample_ids[combo_mask]
        dd1s = data.treatment_ids[combo_mask, 0]
        dd2s = data.treatment_ids[combo_mask, 1]
        masks = data.observation_mask[combo_mask]

        for y, dd1, dd2, cl, mask in zip(
            logit(obs.astype(np.float32)), dd1s, dd2s, cls, masks
        ):
            if mask:
                self.wrapped_model._update(y=y, cl=cl, dd1=dd1, dd2=dd2)

    def n_obs(self) -> int:
        return self.wrapped_model.n_obs()

    def get_results_holder(self, n_samples: int) -> ThetaHolder:
        return SparseDrugComboInteractionResults(
            n_unique_samples=self.n_unique_samples,
            n_unique_treatments=self.n_unique_treatments,
            n_embedding_dimensions=self.n_embedding_dimensions,
            n_thetas=n_samples,
        )

    def reset_model(self):
        self.wrapped_model.reset_model()
