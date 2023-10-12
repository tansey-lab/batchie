import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from batchie.data import Dataset
from numpy.random import BitGenerator
from scipy.special import logit

from batchie.common import ArrayType, copy_array_with_control_treatments_set_to_zero
from batchie.fast_mvn import sample_mvn_from_precision
from batchie.interfaces import BayesianModel, BayesianModelSample, ResultsHolder

logger = logging.getLogger(__name__)


def interactions_to_logits(
    interaction: ArrayType, single_effects: ArrayType, transform: str
):
    if transform == "log":
        viability = np.exp(interaction + np.log(single_effects))
    else:
        viability = interaction + single_effects
    y_logit = logit(np.clip(viability, a_min=0.01, a_max=0.99))
    return y_logit


@dataclass
class SparseDrugComboMCMCSample(BayesianModelSample):
    """A single sample from the MCMC chain for the sparse drug combo model"""

    W: ArrayType
    W0: ArrayType
    V2: ArrayType
    V1: ArrayType
    V0: ArrayType
    alpha: float
    prec: float


class SparseDrugCombo(BayesianModel):
    """Simple Gibbs sampler for Sparse Representation"""

    def __init__(
        self,
        n_embedding_dimensions: int,  # embedding dimension
        n_unique_treatments: int,  # number of total drug/doses
        n_unique_samples: int,  # number of total cell lines
        fake_intercept: bool = True,  # instead of sample fix to mean
        individual_eff: bool = True,
        mult_gamma_proc: bool = True,
        local_shrinkage: bool = True,
        a0: float = 1.1,  # gamma prior hyperparams for all precisions
        b0: float = 1.1,
        min_Mu: float = -10.0,
        max_Mu: float = 10.0,
        rng: Optional[BitGenerator] = None,
    ):
        self.rng = rng if rng else np.random.default_rng()
        self.n_embedding_dimensions = n_embedding_dimensions  # embedding size
        self.n_unique_treatments = n_unique_treatments
        self.n_unique_samples = n_unique_samples
        self.min_Mu = min_Mu
        self.max_Mu = max_Mu
        self.individual_eff = individual_eff  # use linear effects
        self.fake_intercept = fake_intercept

        # for the next two see BHATTACHARYA and DUNSON (2011)
        self.local_shrinkage = local_shrinkage
        self.mult_gamma_proc = mult_gamma_proc

        # data holders
        self.y = np.array([], dtype=np.float64)
        self.num_mcmc_steps = 0

        # indicators for each entry AND sparse query of specific combos
        self.sample_ids = np.array([], dtype=np.int32)
        self.treatment_1 = np.array([], dtype=np.int32)
        self.treatment_2 = np.array([], dtype=np.int32)

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

        # If changing the initialization make sure the last entry of the
        # V's stay at zero because it's used for single controls.
        self.V2 = np.zeros(
            (self.n_unique_treatments, self.n_embedding_dimensions), dtype=np.float32
        )
        self.V1 = np.zeros(
            (self.n_unique_treatments, self.n_embedding_dimensions), dtype=np.float32
        )
        self.W = np.zeros(
            (self.n_unique_samples, self.n_embedding_dimensions), dtype=np.float32
        )
        self.V0 = np.zeros((self.n_unique_treatments,), np.float32)
        self.W0 = np.zeros((self.n_unique_samples,), np.float32)

        # parameters for horseshoe priors
        self.phi2 = 100.0 * np.ones_like(self.V2)
        self.phi1 = 100.0 * np.ones_like(self.V1)
        self.phi0 = 100.0 * np.ones_like(self.V0)
        self.eta2 = np.ones(self.n_embedding_dimensions, dtype=np.float32)
        self.eta1 = np.ones(self.n_embedding_dimensions, dtype=np.float32)
        self.eta0 = 1.0

        # shrinkage
        self.tau = 100.0 * np.ones(self.n_embedding_dimensions, np.float32)
        self.tau0 = 100.0

        if mult_gamma_proc:
            self.gam = np.ones(self.n_embedding_dimensions, np.float32)

        # intercept and overall precision
        self.alpha = 0.0
        self.prec = 100.0

        # holder for model prediction during fit and for eval
        self.Mu = np.zeros(0, np.float32)

    @property
    def n_obs(self):
        return self.y.size

    def add_observations(self, data: Dataset):
        if data.n_treatments != 2:
            raise ValueError(
                "SparseDrugCombo only works with two-treatments combination datasets, "
                "received a {} treatment dataset".format(data.n_treatments)
            )

        self.y = np.concatenate([self.y, data.observations])
        self.sample_ids = np.concatenate([self.sample_ids, data.sample_ids])
        self.treatment_1 = np.concatenate([self.treatment_1, data.treatment_ids[:, 0]])
        self.treatment_2 = np.concatenate([self.treatment_2, data.treatment_ids[:, 1]])

    def reset_model(self):
        self.W = self.W * 0.0
        self.W0 = self.W0 * 0.0
        self.V2 = self.V2 * 0.0
        self.V1 = self.V1 * 0.0
        self.V0 = self.V0 * 0.0
        self.alpha = 0.0
        self.prec = 100.0
        self.Mu = np.zeros(0, np.float32)

    def set_model_state(self, model_state: SparseDrugComboMCMCSample):
        self.W = model_state.W
        self.W0 = model_state.W0
        self.V2 = model_state.V2
        self.V1 = model_state.V1
        self.V0 = model_state.V0
        self.alpha = model_state.alpha
        self.prec = model_state.prec
        self._reconstruct_Mu()

    def mcmc_step(self):
        self.num_mcmc_steps += 1
        self._reconstruct_Mu(clip=False)
        self._alpha_step()
        self._W0_step()
        self._V0_step()
        self._W_step()
        self._V2_step()
        self._V1_step()
        self._prec_W0_step()
        self._prec_V0_step()
        self._prec_obs_step()
        self._prec_V2_step()
        self._prec_V1_step()
        self._prec_W_step()

    def get_model_state(self) -> SparseDrugComboMCMCSample:
        return SparseDrugComboMCMCSample(
            prec=self.prec,
            alpha=self.alpha,
            W0=self.W0.copy(),
            V0=self.V0.copy(),
            W=self.W.copy(),
            V2=self.V2.copy(),
            V1=self.V1.copy(),
        )

    def predict(self, data: Dataset):
        state = self.get_model_state()
        if data.n_treatments == 1:
            return predict_single_drug(state, data)
        elif data.n_treatments == 2:
            return predict(state, data)
        else:
            raise NotImplementedError("SparseDrugCombo only supports 1 or 2 treatments")

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
            tmp1 = (
                copy_array_with_control_treatments_set_to_zero(
                    self.V2, self.treatment_1
                )[cidx]
                * copy_array_with_control_treatments_set_to_zero(
                    self.V2, self.treatment_2
                )[cidx]
            )

            tmp2 = (
                copy_array_with_control_treatments_set_to_zero(
                    self.V1, self.treatment_1
                )[cidx]
                * copy_array_with_control_treatments_set_to_zero(
                    self.V1, self.treatment_2
                )[cidx]
            )

            X = tmp1 + tmp2
            old_contrib = X @ self.W[sample_id]
            resid = self.y[cidx] - self.Mu[cidx] + old_contrib

            Xt = X.transpose()
            prec = self.prec
            mu_part = (Xt @ resid) * prec
            Q = (Xt @ X) * prec
            Q[np.diag_indices(self.n_embedding_dimensions)] += self.tau
            try:
                self.W[sample_id] = sample_mvn_from_precision(Q, mu_part=mu_part)

                # update Mu
                self.Mu[cidx] += X @ self.W[sample_id] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs W-step...")

    def _W0_step(self):
        for sample_id in range(self.n_unique_samples):
            cidx = self.sample_ids == 1
            if cidx.sum() == 0:
                # sample from prior
                stddev = 1.0 / np.sqrt(self.tau0)
                self.W0[sample_id] = self.rng.normal(0.0, stddev)
            else:
                resid = self.y[cidx] - self.Mu[cidx] + self.W0[sample_id]
                old_contrib = self.W0[sample_id]
                N = cidx.sum()
                mean = self.prec * resid.sum() / (self.prec * N + self.tau0)
                stddev = 1.0 / np.sqrt(self.prec * N + self.tau0)
                self.W0[sample_id] = self.rng.normal(mean, stddev)
                self.Mu[cidx] += self.W0[sample_id] - old_contrib

    def _V2_step(self) -> None:
        """the strategy is to iterate over each drug pair combination
        but the trick is to handle the case when drug-pair appears in first
        postition and when it appears in second position
        and solve the linear problem"""
        for treatment_id in range(self.n_unique_treatments):
            # slice where treatment_id, k appears as drug1
            idx1 = self.treatment_1 == treatment_id
            # slice where treatment_id, k appears as drug2
            idx2 = self.treatment_2 == treatment_id
            if idx1.sum() + idx2.sum() == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi2[treatment_id] * self.eta2)
                self.V2[treatment_id] = self.rng.normal(0, stddev)
                continue

            if idx1.sum() == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X1 = (
                    self.W[self.sample_ids[idx1]]
                    * copy_array_with_control_treatments_set_to_zero(
                        self.V2, self.treatment_2
                    )[idx1]
                )
                old_contrib1 = X1 @ self.V2[treatment_id]
                resid1 = self.y[idx1] - self.Mu[idx1] + old_contrib1

            if idx2.sum() == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X2 = (
                    self.W[self.sample_ids[idx2]]
                    * copy_array_with_control_treatments_set_to_zero(
                        self.V2, self.treatment_1
                    )[idx2]
                )
                old_contrib2 = X2 @ self.V2[treatment_id]
                resid2 = self.y[idx2] - self.Mu[idx2] + old_contrib2

            # combine slices of data
            X = np.concatenate([X1, X2])
            resid = np.concatenate([resid1, resid2])
            # resid = np.clip(resid, -10.0, 10.0)
            old_contrib = np.concatenate([old_contrib1, old_contrib2])
            idx = idx1 | idx2

            # sample form posterior
            # X = np.clip(X, -10.0, 10.0)
            Xt = X.transpose()
            mu_part = (Xt @ resid) * self.prec
            Q = (Xt @ X) * self.prec
            dix = np.diag_indices(self.n_embedding_dimensions)
            Q[dix] += self.phi2[treatment_id] * self.eta2
            try:
                self.V2[treatment_id] = sample_mvn_from_precision(Q, mu_part=mu_part)
                # self.V2[treatment_id] = np.clip(self.V2[treatment_id], -10.0, 10.0)
                self.Mu[idx] += X @ self.V2[treatment_id] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs V-step...")

    def _V1_step(self) -> None:
        """the strategy is to iterate over each drug pair combination
        but the trick is to handle the case when drug-pair appears in first
        postition and when it appears in second position
        and solve the linear problem"""
        for treatment_id in range(self.n_unique_treatments):
            # slice where treatment_id, k appears as drug1
            idx1 = self.treatment_1 == treatment_id
            # slice where treatment_id, k appears as drug2
            idx2 = self.treatment_2 == treatment_id
            if idx1.sum() + idx2.sum() == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi1[treatment_id] * self.eta1)
                self.V1[treatment_id] = self.rng.normal(0, stddev)
                continue

            if idx1.sum() == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X1 = self.W[self.sample_ids[idx1]]
                old_contrib1 = X1 @ self.V1[treatment_id]
                resid1 = self.y[idx1] - self.Mu[idx1] + old_contrib1

            if idx2.sum() == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X2 = self.W[self.sample_ids[idx2]]
                old_contrib2 = X2 @ self.V1[treatment_id]
                resid2 = self.y[idx2] - self.Mu[idx2] + old_contrib2

            # combine slices of data
            X = np.concatenate([X1, X2])
            resid = np.concatenate([resid1, resid2])
            old_contrib = np.concatenate([old_contrib1, old_contrib2])
            idx = idx1 | idx2

            # sample form posterior
            Xt = X.transpose()
            mu_part = (Xt @ resid) * self.prec
            Q = (Xt @ X) * self.prec
            dix = np.diag_indices(self.n_embedding_dimensions)
            Q[dix] += self.phi1[treatment_id] * self.eta1
            try:
                self.V1[treatment_id] = sample_mvn_from_precision(Q, mu_part=mu_part)
                # self.V1[treatment_id] = np.clip(self.V1[treatment_id], -10.0, 10.0)
                self.Mu[idx] += X @ self.V1[treatment_id] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs V-step...")

    def _V0_step(self) -> None:
        for treatment_id in range(self.n_unique_treatments):
            # slice where treatment_id, k appears as drug1
            idx1 = self.treatment_1 == treatment_id
            # slice where treatment_id, k appears as drug2
            idx2 = self.treatment_2 == treatment_id

            if (idx1.sum() + idx2.sum()) == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi0[treatment_id] * self.eta0)
                self.V0[treatment_id] = self.rng.normal(0, stddev)
                continue

            old_value = self.V0[treatment_id]

            if idx1.sum() == 0:
                resid1 = []
            else:
                resid1 = self.y[idx1] - self.Mu[idx1] + old_value

            # slice where treatment_id, k appears as drug2
            if idx2.sum() == 0:
                resid2 = []
            else:
                resid2 = self.y[idx2] - self.Mu[idx2] + old_value

            # combine slices of data
            resid = np.concatenate([resid1, resid2])
            idx = idx1 | idx2
            N = idx.sum()
            mean = (
                self.prec
                * resid.sum()
                / (self.prec * N + self.phi0[treatment_id] * self.eta0)
            )
            stddev = 1.0 / np.sqrt(self.prec * N + self.phi0[treatment_id] * self.eta0)
            self.V0[treatment_id] = self.rng.normal(mean, stddev)
            self.Mu[idx] += self.V0[treatment_id] - old_value

    def _prec_obs_step(self) -> None:
        if self.n_obs == 0:
            self.prec = self.rng.gamma(self.a0, 1.0 / self.b0)
            return
        sse = np.square(self.y - self.Mu).sum()
        an = self.a0 + 0.5 * self.n_obs
        bn = self.b0 + 0.5 * sse
        self.prec = self.rng.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.last_rmse = np.sqrt(np.square(self.y - self.Mu).mean())
        self.prec = np.clip(self.prec, C, 1e6)

    def _prec_V2_step(self):
        if self.local_shrinkage:
            phiaux2 = self.rng.gamma(1.0, 1.0 / (1.0 + self.phi2))
            bn = phiaux2 + 0.5 * self.eta2 * self.V2**2
            self.phi2 = self.rng.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array(
                [(self.treatment_1 == c).sum() for c in range(self.n_unique_treatments)]
            )
            N2 = np.array(
                [(self.treatment_2 == c).sum() for c in range(self.n_unique_treatments)]
            )
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi2 = np.clip(self.phi2, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_unique_treatments)
        etaaux2 = self.rng.gamma(1.0, 1.0 / (1.0 + self.eta2))
        bn = etaaux2 + 0.5 * (self.phi2 * self.V2**2).sum(0)
        self.eta2 = self.rng.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta2 = np.clip(self.eta2, C, 1e6)

    def _prec_V1_step(self):
        if self.local_shrinkage:
            phiaux1 = self.rng.gamma(1.0, 1.0 / (1.0 + self.phi1))
            bn = phiaux1 + 0.5 * self.eta1 * self.V1**2
            self.phi1 = self.rng.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array(
                [(self.treatment_1 == c).sum() for c in range(self.n_unique_treatments)]
            )
            N2 = np.array(
                [(self.treatment_2 == c).sum() for c in range(self.n_unique_treatments)]
            )
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi1 = np.clip(self.phi1, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_unique_treatments)
        etaaux1 = self.rng.gamma(1.0, 1.0 / (1.0 + self.eta1))
        bn = etaaux1 + 0.5 * (self.phi1 * self.V1**2).sum(0)
        self.eta1 = self.rng.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta1 = np.clip(self.eta1, C, 1e6)

    def _prec_V0_step(self):
        if self.local_shrinkage:
            phiaux0 = self.rng.gamma(1.0, 1.0 / (1.0 + self.phi0))
            bn = phiaux0 + 0.5 * self.eta0 * self.V0**2
            self.phi0 = self.rng.gamma(1.0, 1.0 / (bn + 1e-3))  # +0.01 for stability
            N1 = np.array(
                [(self.treatment_1 == c).sum() for c in range(self.n_unique_treatments)]
            )
            N2 = np.array(
                [(self.treatment_2 == c).sum() for c in range(self.n_unique_treatments)]
            )
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi0 = np.clip(self.phi0, C, 1e6)

        # V0 precision
        an = 0.5 * (1 + self.n_unique_treatments)
        etaaux0 = self.rng.gamma(1.0, 1.0 / (1.0 + self.eta0))
        bn = etaaux0 + 0.5 * (self.phi0 * self.V0**2).sum()
        self.eta0 = self.rng.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta0 = np.clip(self.eta0, C, 1e6)

    def _prec_W_step(self):
        # gamma process
        parssq = self.W**2
        if self.mult_gamma_proc:
            tmp = np.cumprod(self.gam) / self.gam[0]
            an = 2 + 0.5 * self.n_unique_samples * self.n_embedding_dimensions
            bn = 1 + 0.5 * (tmp * parssq).sum()
            self.gam[0] = self.rng.gamma(an, 1.0 / (bn + 1e-3))
            # sample all others
            for d in range(1, self.n_embedding_dimensions):
                tmp = np.cumprod(self.gam)[d:] / self.gam[d]
                an = 3 + 0.5 * self.n_unique_samples * (self.n_embedding_dimensions - d)
                bn = 1 + 0.5 * (tmp * parssq[:, d:]).sum()
                self.gam[d] = self.rng.gamma(an, 1.0 / (bn + 1e-3))
            self.tau = np.cumprod(self.gam)
        else:
            an = self.a0 + 0.5 * self.n_unique_samples
            bn = self.b0 + 0.5 * parssq.sum(0)
            self.tau = self.rng.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.tau = np.clip(self.tau, C, 1e6)

    def _prec_W0_step(self):
        # W0 precision
        an = self.a0 + 0.5 * self.n_unique_samples
        bn = self.b0 + 0.5 * (self.W0**2).sum()
        self.tau0 = self.rng.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.tau0 = np.clip(self.tau0, C, 1e6)

    def _alpha_step(self) -> None:
        old_value = self.alpha
        if self.n_obs == 0:
            # alpha is undefined, so assume there is no intercept and sample from prior
            # it's better than sampling from super degenerate prior
            return
        if self.fake_intercept:
            self.alpha = np.mean(self.y)
        else:
            mean = (self.y - self.Mu + self.alpha).mean()
            stddev = 1.0 / np.sqrt(self.n_obs)
            self.alpha = self.rng.normal(mean, stddev)
        self.Mu += self.alpha - old_value

    def _reconstruct_Mu(self, clip=True):
        if self.n_obs == 0:
            return
        # Reconstruct the entire matrix of means, excluding the upper triangular portion
        interaction2 = np.sum(
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
        interaction1 = np.sum(
            self.W[self.sample_ids]
            * (
                copy_array_with_control_treatments_set_to_zero(
                    self.V1, self.treatment_1
                )
                + copy_array_with_control_treatments_set_to_zero(
                    self.V1, self.treatment_2
                )
            ),
            -1,
        )
        intercept = (
            self.alpha
            + self.W0[self.sample_ids]
            + copy_array_with_control_treatments_set_to_zero(self.V0, self.treatment_1)
            + copy_array_with_control_treatments_set_to_zero(self.V0, self.treatment_2)
        )
        self.Mu = intercept + interaction1 + interaction2
        if clip:
            self.Mu = np.clip(self.Mu, self.min_Mu, self.max_Mu)

    def _ess_pars(self):
        return [1.0 / np.sqrt(self.prec)] + self.Mu.tolist()


class SparseDrugComboResults(ResultsHolder):
    def __init__(
        self,
        n_unique_samples: int,
        n_unique_treatments: int,
        n_embedding_dimensions: int,
        n_mcmc_steps: int,
    ):
        super().__init__(n_mcmc_steps)
        self.n_unique_samples = n_unique_samples
        self.n_unique_treatments = n_unique_treatments
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_mcmc_steps = n_mcmc_steps

        self.V2 = np.zeros(
            (self.n_mcmc_steps, self.n_unique_treatments, self.n_embedding_dimensions),
            dtype=np.float32,
        )
        self.V1 = np.zeros(
            (self.n_mcmc_steps, self.n_unique_treatments, self.n_embedding_dimensions),
            dtype=np.float32,
        )
        self.W = np.zeros(
            (self.n_mcmc_steps, self.n_unique_samples, self.n_embedding_dimensions),
            dtype=np.float32,
        )
        self.V0 = np.zeros(
            (
                self.n_mcmc_steps,
                self.n_unique_treatments,
            ),
            np.float32,
        )
        self.W0 = np.zeros(
            (
                self.n_mcmc_steps,
                self.n_unique_samples,
            ),
            np.float32,
        )

        self.alpha = np.zeros((self.n_mcmc_steps,), np.float32)
        self.prec = np.zeros((self.n_mcmc_steps,), np.float32)

    def get_mcmc_sample(self, step_index: int) -> SparseDrugComboMCMCSample:
        """Get one sample from the MCMC chain"""
        # Test if this is beyond the step we are current at with the cursor
        if step_index >= self._cursor:
            raise ValueError("Cannot get a step beyond the current cursor position")

        return SparseDrugComboMCMCSample(
            W=self.W[step_index],
            W0=self.W0[step_index],
            V2=self.V2[step_index],
            V1=self.V1[step_index],
            V0=self.V0[step_index],
            alpha=self.alpha[step_index],
            prec=self.prec[step_index],
        )

    def _save_mcmc_sample(self, sample: SparseDrugComboMCMCSample):
        self.V2[self._cursor] = sample.V2
        self.V1[self._cursor] = sample.V1
        self.W[self._cursor] = sample.W
        self.V0[self._cursor] = sample.V0
        self.W0[self._cursor] = sample.W0
        self.alpha[self._cursor] = sample.alpha
        self.prec[self._cursor] = sample.prec
        self._cursor += 1

    def save_h5(self, fn: str):
        """Save all arrays to h5"""
        with h5py.File(fn, "w") as f:
            f.create_dataset("V2", data=self.V2)
            f.create_dataset("V1", data=self.V1)
            f.create_dataset("W", data=self.W)
            f.create_dataset("V0", data=self.V0)
            f.create_dataset("W0", data=self.W0)
            f.create_dataset("alpha", data=self.alpha)
            f.create_dataset("prec", data=self.prec)

            # Save the cursor value metadata
            f.attrs["cursor"] = self._cursor

    @staticmethod
    def load_h5(path: str):
        """Load saved data from h5 archive"""
        with h5py.File(path, "r") as f:
            n_unique_samples = f["W"].shape[1]
            n_unique_treatments = f["V0"].shape[1]
            n_embedding_dimensions = f["W"].shape[2]
            n_mcmc_steps = f["W"].shape[0]

            results = SparseDrugComboResults(
                n_unique_samples=n_unique_samples,
                n_unique_treatments=n_unique_treatments,
                n_embedding_dimensions=n_embedding_dimensions,
                n_mcmc_steps=n_mcmc_steps,
            )

            results.V2 = f["V2"][:]
            results.V1 = f["V1"][:]
            results.W = f["W"][:]
            results.V0 = f["V0"][:]
            results.W0 = f["W0"][:]
            results.alpha = f["alpha"][:]
            results.prec = f["prec"][:]
            results._cursor = f.attrs["cursor"]

        return results


def predict(mcmc_sample: SparseDrugComboMCMCSample, data: Dataset):
    interaction2 = np.sum(
        mcmc_sample.W[data.sample_ids]
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V2, data.treatment_ids[:, 0]
        )
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V2, data.treatment_ids[:, 1]
        ),
        -1,
    )
    interaction1 = np.sum(
        mcmc_sample.W[data.sample_ids]
        * (
            copy_array_with_control_treatments_set_to_zero(
                mcmc_sample.V1, data.treatment_ids[:, 0]
            )
            + copy_array_with_control_treatments_set_to_zero(
                mcmc_sample.V1, data.treatment_ids[:, 1]
            )
        ),
        -1,
    )
    intercept = (
        mcmc_sample.alpha
        + mcmc_sample.W0[data.sample_ids]
        + copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V0, data.treatment_ids[:, 0]
        )
        + copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V0, data.treatment_ids[:, 1]
        )
    )
    Mu = intercept + interaction1 + interaction2
    return Mu


def predict_single_drug(mcmc_sample: SparseDrugComboMCMCSample, plate: Dataset):
    interaction1 = np.sum(
        mcmc_sample.W[plate.sample_ids]
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V1, plate.treatment_ids[:, 0]
        ),
        -1,
    )
    intercept = (
        mcmc_sample.alpha
        + mcmc_sample.W0[plate.sample_ids]
        + copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V0, plate.treatment_ids[:, 0]
        )
    )
    Mu = intercept + interaction1
    return Mu


def bliss(mcmc_sample: SparseDrugComboMCMCSample, plate: Dataset):
    interaction2 = np.sum(
        mcmc_sample.W[plate.treatment_ids]
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V2, plate.treatment_ids[:, 0]
        )
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V2, plate.treatment_ids[:, 1]
        ),
        -1,
    )
    return interaction2
