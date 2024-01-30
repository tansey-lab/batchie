import logging
import warnings
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass
import h5py
import numpy as np
from numpy.random import Generator
from scipy.special import logit, expit

from batchie.core import BayesianModel, Theta, ThetaHolder
from batchie.data import ScreenBase
from batchie.fast_mvn import sample_mvn_from_precision
from batchie.common import (
    ArrayType,
    copy_array_with_control_treatments_set_to_zero,
    FloatingPointType,
)

logger = logging.getLogger(__name__)


@dataclass
class SparseDrugComboMCMCSample(Theta):
    """A single sample from the MCMC chain for the sparse drug combo model"""

    W: ArrayType
    W0: ArrayType
    V2: ArrayType
    V1: ArrayType
    V0: ArrayType
    alpha: float
    precision: float


class SparseDrugComboResults(ThetaHolder):
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
            dtype=FloatingPointType,
        )
        self.V1 = np.zeros(
            (n_thetas, self.n_unique_treatments, self.n_embedding_dimensions),
            dtype=FloatingPointType,
        )
        self.W = np.zeros(
            (n_thetas, self.n_unique_samples, self.n_embedding_dimensions),
            dtype=FloatingPointType,
        )
        self.V0 = np.zeros(
            (
                n_thetas,
                self.n_unique_treatments,
            ),
            FloatingPointType,
        )
        self.W0 = np.zeros(
            (
                n_thetas,
                self.n_unique_samples,
            ),
            FloatingPointType,
        )

        self.alpha = np.zeros((n_thetas,), FloatingPointType)
        self.precision = np.zeros((n_thetas,), FloatingPointType)

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

        output = SparseDrugComboResults(
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

    def get_theta(self, step_index: int) -> SparseDrugComboMCMCSample:
        # Test if this is beyond the step we are current at with the cursor
        if step_index >= self._cursor:
            raise ValueError("Cannot get a step beyond the current cursor position")

        return SparseDrugComboMCMCSample(
            W=self.W[step_index],
            W0=self.W0[step_index],
            V2=self.V2[step_index],
            V1=self.V1[step_index],
            V0=self.V0[step_index],
            alpha=self.alpha[step_index].item(),
            precision=self.precision[step_index].item(),
        )

    def _save_theta(self, sample: SparseDrugComboMCMCSample, sample_index: int):
        self.V2[sample_index] = sample.V2
        self.V1[sample_index] = sample.V1
        self.W[sample_index] = sample.W
        self.V0[sample_index] = sample.V0
        self.W0[sample_index] = sample.W0
        self.alpha[sample_index] = sample.alpha
        self.precision[sample_index] = sample.precision

    def save_h5(self, fn: str):
        with h5py.File(fn, "w") as f:
            f.create_dataset("V2", data=self.V2, compression="gzip")
            f.create_dataset("V1", data=self.V1, compression="gzip")
            f.create_dataset("W", data=self.W, compression="gzip")
            f.create_dataset("V0", data=self.V0, compression="gzip")
            f.create_dataset("W0", data=self.W0, compression="gzip")
            f.create_dataset("alpha", data=self.alpha, compression="gzip")
            f.create_dataset("precision", data=self.precision, compression="gzip")

            # Save the cursor value metadata
            f.attrs["cursor"] = self._cursor

    @staticmethod
    def load_h5(path: str):
        with h5py.File(path, "r") as f:
            n_unique_samples = f["W"].shape[1]
            n_unique_treatments = f["V0"].shape[1]
            n_embedding_dimensions = f["W"].shape[2]
            n_samples = f["W"].shape[0]

            results = SparseDrugComboResults(
                n_unique_samples=n_unique_samples,
                n_unique_treatments=n_unique_treatments,
                n_embedding_dimensions=n_embedding_dimensions,
                n_thetas=n_samples,
            )

            results.V2 = f["V2"][:]
            results.V1 = f["V1"][:]
            results.W = f["W"][:]
            results.V0 = f["V0"][:]
            results.W0 = f["W0"][:]
            results.alpha = f["alpha"][:]
            results.precision = f["precision"][:]
            results._cursor = f.attrs["cursor"]

        return results


class LegacySparseDrugComboImpl:
    """
    Original implementation of Bayesian tensor factorization model for
    predicting combination drug response. Preserved here without changes
    to ensure reproducibility of results.
    """

    def __init__(
        self,
        n_dims: int,  # embedding dimension
        n_drugdoses: int,  # number of total drug/doses
        n_clines: int,  # number of total cell lines
        intercept: bool = True,  # use intercept
        fake_intercept: bool = True,  # instead of sample fix to mean
        individual_eff: bool = True,
        mult_gamma_proc: bool = True,
        local_shrinkage: bool = True,
        a0: float = 1.1,  # gamma prior hyperparmas for all precisions
        b0: float = 1.1,
        min_Mu: float = -10.0,
        max_Mu: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.D = n_dims  # embedding size
        self.n_drugdoses = n_drugdoses
        self.n_clines = n_clines
        self.min_Mu = min_Mu
        self.max_Mu = max_Mu
        self.individual_eff = individual_eff  # use linear effects
        self.intercept = intercept
        self.fake_intercept = fake_intercept

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
        self.V1 = np.zeros(sh, dtype=np.float32)

        sh = (self.n_clines, self.D)
        self.W = np.zeros(sh, dtype=np.float32)

        self.V0 = np.zeros((self.n_drugdoses,), np.float32)
        self.W0 = np.zeros((self.n_clines,), np.float32)

        # paramaters for hroseshow priorss
        self.phi2 = 100.0 * np.ones_like(self.V2)
        self.phi1 = 100.0 * np.ones_like(self.V1)
        self.phi0 = 100.0 * np.ones_like(self.V0)
        self.eta2 = np.ones(self.D, dtype=np.float32)
        self.eta1 = np.ones(self.D, dtype=np.float32)
        self.eta0 = 1.0

        # shrinkage
        self.tau = 100.0 * np.ones(self.D, np.float32)
        self.tau0 = 100.0

        if mult_gamma_proc:
            self.gam = np.ones(self.D, np.float32)

        # intercept and overall precision
        self.alpha = 0.0
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
        self.W0 = self.W0 * 0.0
        self.V2 = self.V2 * 0.0
        self.V1 = self.V1 * 0.0
        self.V0 = self.V0 * 0.0
        self.alpha = 0.0
        self.prec = 100.0
        self.Mu = np.zeros(0, np.float32)

    def get(self, attr, ix):
        # mask -1 for single controls
        A = self.__getattribute__(attr)[ix].copy()
        controls = np.where(ix == -1)[0]
        if len(controls > 0):
            A[controls] = 0.0
        return A

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
            tmp1 = self.get("V2", dd1[cidx]) * self.get("V2", dd2[cidx])
            tmp2 = self.get("V1", dd1[cidx]) + self.get("V1", dd2[cidx])
            X = tmp1 + tmp2
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

    def _W0_step(self) -> None:
        y, _, *_ = self.encode_obs()
        for c in range(self.n_clines):
            cidx = np.array(self.cline_idxs[c], copy=False)
            if len(cidx) == 0:
                # sample from prior
                stddev = 1.0 / np.sqrt(self.tau0)
                self.W0[c] = np.random.normal(0.0, stddev)
            else:
                resid = y[cidx] - self.Mu[cidx] + self.W0[c]
                # resid = np.clip(resid, -10.0, 10.0)
                old_contrib = self.W0[c]
                N = len(cidx)
                mean = self.prec * resid.sum() / (self.prec * N + self.tau0)
                stddev = 1.0 / np.sqrt(self.prec * N + self.tau0)
                self.W0[c] = np.random.normal(mean, stddev)
                self.Mu[cidx] += self.W0[c] - old_contrib

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
                X1 = self.W[cline[idx1]] * self.get("V2", dd2[idx1])
                old_contrib1 = X1 @ self.V2[m]
                resid1 = y[idx1] - self.Mu[idx1] + old_contrib1

            if len(idx2) == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(0, self.D)
            else:
                X2 = self.W[cline[idx2]] * self.get("V2", dd1[idx2])
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

    def _V1_step(self) -> None:
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
                stddev = 1.0 / np.sqrt(self.phi1[m] * self.eta1)
                self.V1[m] = np.random.normal(0, stddev)
                continue

            if len(idx1) == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(0, self.D)
            else:
                X1 = self.W[cline[idx1]]
                old_contrib1 = X1 @ self.V1[m]
                resid1 = y[idx1] - self.Mu[idx1] + old_contrib1

            if len(idx2) == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(0, self.D)
            else:
                X2 = self.W[cline[idx2]]
                old_contrib2 = X2 @ self.V1[m]
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
            Q[dix] += self.phi1[m] * self.eta1
            try:
                self.V1[m] = sample_mvn_from_precision(Q, mu_part=mu_part)
                # self.V1[m] = np.clip(self.V1[m], -10.0, 10.0)
                self.Mu[idx] += X @ self.V1[m] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs V-step...")

    def _V0_step(self) -> None:
        y, cline, dd1, dd2 = self.encode_obs()
        for m in range(self.n_drugdoses):
            # slice where m, k appears as drug1
            idx1 = np.array(self.dd1_idxs[m], copy=False, dtype=int)
            # slice where m, k appears as drug2
            idx2 = np.array(self.dd2_idxs[m], copy=False, dtype=int)

            if len(idx1) + len(idx2) == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi0[m] * self.eta0)
                self.V0[m] = np.random.normal(0, stddev)
                continue

            old_value = self.V0[m]

            if len(idx1) == 0:
                resid1 = []
            else:
                resid1 = y[idx1] - self.Mu[idx1] + old_value

            # slice where m, k appears as drug2
            if len(idx2) == 0:
                resid2 = []
            else:
                resid2 = y[idx2] - self.Mu[idx2] + old_value

            # combine slices of data
            resid = np.concatenate([resid1, resid2])
            # resid = np.clip(resid, -10.0, 10.0)
            idx = np.concatenate([idx1, idx2])
            N = len(idx)
            mean = self.prec * resid.sum() / (self.prec * N + self.phi0[m] * self.eta0)
            stddev = 1.0 / np.sqrt(self.prec * N + self.phi0[m] * self.eta0)
            self.V0[m] = np.random.normal(mean, stddev)
            self.Mu[idx] += self.V0[m] - old_value

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

    def _prec_V1_step(self):
        if self.local_shrinkage:
            phiaux1 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi1))
            bn = phiaux1 + 0.5 * self.eta1 * self.V1**2
            self.phi1 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array([len(self.dd1_idxs[c]) for c in range(self.n_drugdoses)])
            N2 = np.array([len(self.dd2_idxs[c]) for c in range(self.n_drugdoses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi1 = np.clip(self.phi1, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_drugdoses)
        etaaux1 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta1))
        bn = etaaux1 + 0.5 * (self.phi1 * self.V1**2).sum(0)
        self.eta1 = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.eta1 = np.clip(self.eta1, C, 1e6)

    def _prec_V0_step(self):
        if self.local_shrinkage:
            phiaux0 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi0))
            bn = phiaux0 + 0.5 * self.eta0 * self.V0**2
            self.phi0 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))  # +0.01 for stability
            N1 = np.array([len(self.dd1_idxs[c]) for c in range(self.n_drugdoses)])
            N2 = np.array([len(self.dd2_idxs[c]) for c in range(self.n_drugdoses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi0 = np.clip(self.phi0, C, 1e6)

        # V0 precision
        an = 0.5 * (1 + self.n_drugdoses)
        etaaux0 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta0))
        bn = etaaux0 + 0.5 * (self.phi0 * self.V0**2).sum()
        self.eta0 = np.random.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
        C = 1.0 / np.sqrt(1 + self.n_obs())
        self.eta0 = np.clip(self.eta0, C, 1e6)

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

    def _alpha_step(self) -> None:
        old_value = self.alpha
        if self.n_obs() == 0:
            # alpha is undefined, so assume there is no intercept and sample from prior
            # it's better than sampling from super degenerate prior
            return
        if self.fake_intercept:
            self.alpha = np.mean(self.y)
        else:
            # self._reconstruct_Mu()
            y, *_ = self.encode_obs()
            mean = (y - self.Mu + self.alpha).mean()
            stddev = 1.0 / np.sqrt(self.n_obs())
            self.alpha = np.random.normal(mean, stddev)
        self.Mu += self.alpha - old_value

    def mcmc_step(self) -> None:
        self.num_mcmc_steps += 1
        # Note! all thse reconstruct Mu's can be changed inplace
        # when updating the parameter, currently not efficiient
        self._reconstruct_Mu(clip=False)
        # if self.intercept:
        self._alpha_step()
        # self._reconstruct_Mu()
        # if self.individual_eff:
        self._W0_step()
        # self._reconstruct_Mu()
        self._V0_step()
        # self._reconstruct_Mu()
        self._W_step()
        self._V2_step()
        self._V1_step()
        # self._reconstruct_Mu()
        # self._reconstruct_Mu()
        self._prec_W0_step()
        self._prec_V0_step()
        self._prec_obs_step()
        self._prec_V2_step()
        self._prec_V1_step()
        self._prec_W_step()

    def _reconstruct_Mu(self, clip=True):
        if self.n_obs() == 0:
            return
        # Reconstruct the entire matrix of means, excluding the upper triangular portion
        _, cline, dd1, dd2 = self.encode_obs()
        interaction2 = np.sum(
            self.W[cline] * self.get("V2", dd1) * self.get("V2", dd2), -1
        )
        interaction1 = np.sum(
            self.W[cline] * (self.get("V1", dd1) + self.get("V1", dd2)), -1
        )
        intercept = (
            self.alpha + self.W0[cline] + self.get("V0", dd1) + self.get("V0", dd2)
        )
        self.Mu = intercept + interaction1 + interaction2
        if clip:
            self.Mu = np.clip(self.Mu, self.min_Mu, self.max_Mu)

    def predict(self, cline: np.ndarray, dd1: np.ndarray, dd2: np.ndarray):
        interaction2 = np.sum(
            self.W[cline] * self.get("V2", dd1) * self.get("V2", dd2), -1
        )
        interaction1 = np.sum(
            self.W[cline] * (self.get("V1", dd1) + self.get("V1", dd2)), -1
        )
        intercept = (
            self.alpha + self.W0[cline] + self.get("V0", dd1) + self.get("V0", dd2)
        )
        Mu = intercept + interaction1 + interaction2
        return Mu

    def predict_single_drug(self, cline: np.ndarray, dd1: np.ndarray):
        interaction1 = np.sum(self.W[cline] * self.get("V1", dd1), -1)
        intercept = self.alpha + self.W0[cline] + self.get("V0", dd1)
        Mu = intercept + interaction1
        return Mu

    def bliss(self, cline: np.ndarray, dd1: np.ndarray, dd2: np.ndarray):
        interaction2 = np.sum(
            self.W[cline] * self.get("V2", dd1) * self.get("V2", dd2), -1
        )
        return interaction2

    def ess_pars(self):
        # return [1.0 / np.sqrt(self.prec)]
        return [1.0 / np.sqrt(self.prec)] + self.Mu.tolist()


class SparseDrugCombo(BayesianModel):
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
        rng: Optional[Generator] = None,
        predict_interactions: bool = False,
        interaction_log_transform: bool = True,
        intercept: bool = True,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_unique_treatments = n_unique_treatments
        self.n_unique_samples = n_unique_samples
        self._rng = rng
        self.predict_interactions = predict_interactions
        self.interaction_log_transform = interaction_log_transform
        self.wrapped_model = LegacySparseDrugComboImpl(
            n_dims=n_embedding_dimensions,
            n_drugdoses=n_unique_treatments,
            n_clines=n_unique_samples,
            intercept=intercept,
            fake_intercept=fake_intercept,
            individual_eff=individual_eff,
            mult_gamma_proc=mult_gamma_proc,
            local_shrinkage=local_shrinkage,
            a0=a0,
            b0=b0,
            min_Mu=min_Mu,
            max_Mu=max_Mu,
        )

    def set_model_state(self, theta: SparseDrugComboMCMCSample):
        self.wrapped_model.reset_model()
        self.wrapped_model.W0 = theta.W0.astype(np.float32)
        self.wrapped_model.W = theta.W.astype(np.float32)
        self.wrapped_model.V0 = theta.V0.astype(np.float32)
        self.wrapped_model.V1 = theta.V1.astype(np.float32)
        self.wrapped_model.V2 = theta.V2.astype(np.float32)
        self.wrapped_model.alpha = theta.alpha
        self.wrapped_model.prec = theta.precision
        self.wrapped_model._reconstruct_Mu()

    def get_model_state(self) -> SparseDrugComboMCMCSample:
        return SparseDrugComboMCMCSample(
            precision=self.wrapped_model.prec,
            alpha=self.wrapped_model.alpha,
            W0=self.wrapped_model.W0.copy().astype(FloatingPointType),
            V0=self.wrapped_model.V0.copy().astype(FloatingPointType),
            W=self.wrapped_model.W.copy().astype(FloatingPointType),
            V2=self.wrapped_model.V2.copy().astype(FloatingPointType),
            V1=self.wrapped_model.V1.copy().astype(FloatingPointType),
        )

    def predict(self, data: ScreenBase) -> ArrayType:
        state = self.get_model_state()
        if data.treatment_arity == 1:
            return predict_single_drug(state, data)
        elif data.treatment_arity == 2:
            predictions = predict(state, data)
            if self.predict_interactions:
                single_effects = data.single_treatment_effects

                if single_effects is None:
                    raise ValueError(
                        "Cannot predict interactions without observed single treatment effects"
                    )

                return interactions_to_logits(
                    predictions, single_effects, self.interaction_log_transform
                )
            else:
                return predictions
        else:
            raise NotImplementedError("SparseDrugCombo only supports 1 or 2 treatments")

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
        if not (data.observations >= 0.0).all():
            raise ValueError(
                "Observations should be non-negative, please check input data"
            )

        observations_transformed = logit(
            np.clip(data.observations.astype(np.float32), a_min=0.01, a_max=0.99)
        )

        if np.isnan(observations_transformed).any():
            raise ValueError("NaNs in observations, please check input data")

        for y, dd, cl, mask in zip(
            observations_transformed,
            data.treatment_ids,
            data.sample_ids,
            data.observation_mask,
        ):
            if mask:
                self.wrapped_model._update(y=y, cl=cl, dd1=dd[0], dd2=dd[1])

    def n_obs(self) -> int:
        return self.wrapped_model.n_obs()

    def get_results_holder(self, n_samples: int) -> ThetaHolder:
        return SparseDrugComboResults(
            n_unique_samples=self.n_unique_samples,
            n_unique_treatments=self.n_unique_treatments,
            n_embedding_dimensions=self.n_embedding_dimensions,
            n_thetas=n_samples,
        )

    def reset_model(self):
        self.wrapped_model.reset_model()


def predict(mcmc_sample: SparseDrugComboMCMCSample, data: ScreenBase):
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
    return np.clip(expit(Mu), a_min=0.01, a_max=0.99)


def predict_single_drug(mcmc_sample: SparseDrugComboMCMCSample, data: ScreenBase):
    interaction1 = np.sum(
        mcmc_sample.W[data.sample_ids]
        * copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V1, data.treatment_ids[:, 0]
        ),
        -1,
    )
    intercept = (
        mcmc_sample.alpha
        + mcmc_sample.W0[data.sample_ids]
        + copy_array_with_control_treatments_set_to_zero(
            mcmc_sample.V0, data.treatment_ids[:, 0]
        )
    )
    Mu = intercept + interaction1
    return np.clip(expit(Mu), a_min=0.01, a_max=0.99)


def interactions_to_logits(
    interaction: ArrayType, single_effects: ArrayType, log_transform: bool
):
    multiplicative_single_effects = np.clip(
        np.product(single_effects, axis=1), a_min=0.01, a_max=0.99
    )
    if log_transform:
        viability = np.exp(interaction + np.log(multiplicative_single_effects))
    else:
        viability = interaction + multiplicative_single_effects
    y_logit = logit(np.clip(viability, a_min=0.01, a_max=0.99))
    return y_logit
