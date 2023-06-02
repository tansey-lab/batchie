import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Union
from copy import deepcopy
from scipy.special import logit
from batchie.fast_mvn import sample_mvn_from_precision
import warnings
from batchie.interfaces import BayesianModel, Predictor, PredictorHolder
from batchie.datasets import ComboPlate
from batchie.common import ArrayType


class ComboPredictor(Predictor):
    def __init__(
        self,
        W: ArrayType,
        W0: ArrayType,
        V2: ArrayType,
        V1: ArrayType,
        V0: ArrayType,
        alpha: float,
        prec: float,
    ):
        self.W = deepcopy(W)
        self.W0 = deepcopy(W0)
        self.V2 = deepcopy(V2)
        self.V1 = deepcopy(V1)
        self.V0 = deepcopy(V0)
        self.alpha = alpha
        self.var = 1.0 / prec

    def get(self, attr, ix):
        # mask -1 for single controls
        A = self.__getattribute__(attr)[ix].copy()
        controls = np.where(ix == -1)[0]
        if len(controls > 0):
            A[controls] = 0.0
        return A

    def predict(self, plate: ComboPlate, **kwargs):
        interaction2 = np.sum(
            self.W[plate.cline] * self.get("V2", plate.dd1) * self.get("V2", plate.dd2),
            -1,
        )
        interaction1 = np.sum(
            self.W[plate.cline]
            * (self.get("V1", plate.dd1) + self.get("V1", plate.dd2)),
            -1,
        )
        intercept = (
            self.alpha
            + self.W0[plate.cline]
            + self.get("V0", plate.dd1)
            + self.get("V0", plate.dd2)
        )
        Mu = intercept + interaction1 + interaction2
        return Mu

    def variance(self):
        return self.var


def interactions_to_logits(
    interaction: ArrayType, single_effects: ArrayType, transform: str
):
    if transform == "log":
        viability = np.exp(interaction + np.log(single_effects))
    else:
        viability = interaction + single_effects
    y_logit = logit(np.clip(viability, a_min=0.01, a_max=0.99))
    return y_logit


class SparseDrugCombo(BayesianModel):
    """Simple Gibbs sampler for Sparse Representation"""

    def __init__(
        self,
        n_embedding_dimensions: int,  # embedding dimension
        n_doses: int,  # number of total drug/doses
        n_samples: int,  # number of total cell lines
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
        self.n_embedding_dimensions = n_embedding_dimensions  # embedding size
        self.n_doses = n_doses
        self.n_samples = n_samples
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
        self.y = np.array([], dtype=np.float64)
        self.num_mcmc_steps = 0

        # indicators for each entry AND sparse query of speficic combos
        self.cline = np.array([], dtype=np.int32)
        self.dd1 = np.array([], dtype=np.float64)
        self.dd2 = np.array([], dtype=np.float64)

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
            (self.n_doses, self.n_embedding_dimensions), dtype=np.float32
        )
        self.V1 = np.zeros(
            (self.n_doses, self.n_embedding_dimensions), dtype=np.float32
        )
        self.W = np.zeros(
            (self.n_samples, self.n_embedding_dimensions), dtype=np.float32
        )
        self.V0 = np.zeros((self.n_doses,), np.float32)
        self.W0 = np.zeros((self.n_samples,), np.float32)

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

    def _update(self, y, cline, dd1, dd2):
        self.y = np.concatenate([self.y, y])
        self.cline = np.concatenate([self.cline, cline])
        self.dd1 = np.concatenate([self.dd1, dd1])
        self.dd2 = np.concatenate([self.dd2, dd2])

    def update(self, plate: ComboPlate, y: ArrayType):
        for y_, cline, dd1, dd2 in zip(y, plate.cline, plate.dd1, plate.dd2):
            self._update(y_, cline, dd1, dd2)

    def predictor(self) -> ComboPredictor:
        pred = ComboPredictor(
            self.W, self.W0, self.V2, self.V1, self.V0, self.alpha, self.prec
        )
        return pred

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
        for c in range(self.n_samples):
            cidx = self.cline == c
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
            Q[np.diag_indices(self.n_embedding_dimensions)] += self.tau
            try:
                self.W[c] = sample_mvn_from_precision(Q, mu_part=mu_part)

                # update Mu
                self.Mu[cidx] += X @ self.W[c] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs W-step...")

    def _W0_step(self) -> None:
        y, _, *_ = self.encode_obs()
        for c in range(self.n_samples):
            cidx = self.cline == 1
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
        for m in range(self.n_doses):
            # slice where m, k appears as drug1
            idx1 = self.dd1 == m
            # slice where m, k appears as drug2
            idx2 = self.dd2 == m
            if idx1.sum() + idx2.sum() == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi2[m] * self.eta2)
                self.V2[m] = np.random.normal(0, stddev)
                continue

            if idx1.sum() == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X1 = self.W[cline[idx1]] * self.get("V2", dd2[idx1])
                old_contrib1 = X1 @ self.V2[m]
                resid1 = y[idx1] - self.Mu[idx1] + old_contrib1

            if idx2.sum() == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X2 = self.W[cline[idx2]] * self.get("V2", dd1[idx2])
                old_contrib2 = X2 @ self.V2[m]
                resid2 = y[idx2] - self.Mu[idx2] + old_contrib2

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
        for m in range(self.n_doses):
            # slice where m, k appears as drug1
            idx1 = self.dd1 == m
            # slice where m, k appears as drug2
            idx2 = self.dd2 == m
            if idx1.sum() + idx2.sum() == 0:
                # no data seen yet, sample from prior
                stddev = 1.0 / np.sqrt(self.phi1[m] * self.eta1)
                self.V1[m] = np.random.normal(0, stddev)
                continue

            if idx1.sum() == 0:
                resid1 = []
                old_contrib1 = []
                X1 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X1 = self.W[cline[idx1]]
                old_contrib1 = X1 @ self.V1[m]
                resid1 = y[idx1] - self.Mu[idx1] + old_contrib1

            if idx2.sum() == 0:
                resid2 = []
                old_contrib2 = []
                X2 = np.array([], dtype=np.float32).reshape(
                    0, self.n_embedding_dimensions
                )
            else:
                X2 = self.W[cline[idx2]]
                old_contrib2 = X2 @ self.V1[m]
                resid2 = y[idx2] - self.Mu[idx2] + old_contrib2

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
            Q[dix] += self.phi1[m] * self.eta1
            try:
                self.V1[m] = sample_mvn_from_precision(Q, mu_part=mu_part)
                # self.V1[m] = np.clip(self.V1[m], -10.0, 10.0)
                self.Mu[idx] += X @ self.V1[m] - old_contrib
            except:
                warnings.warn("Numeric instability in Gibbs V-step...")

    def _V0_step(self) -> None:
        y, cline, dd1, dd2 = self.encode_obs()
        for m in range(self.n_doses):
            # slice where m, k appears as drug1
            idx1 = dd1 == m
            # slice where m, k appears as drug2
            idx2 = dd2 == m

            if (idx1.sum() + idx2.sum()) == 0:
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
            if idx2.sum() == 0:
                resid2 = []
            else:
                resid2 = y[idx2] - self.Mu[idx2] + old_value

            # combine slices of data
            resid = np.concatenate([resid1, resid2])
            # resid = np.clip(resid, -10.0, 10.0)
            idx = idx1 | idx2
            N = len(idx)
            mean = self.prec * resid.sum() / (self.prec * N + self.phi0[m] * self.eta0)
            stddev = 1.0 / np.sqrt(self.prec * N + self.phi0[m] * self.eta0)
            self.V0[m] = np.random.normal(mean, stddev)
            self.Mu[idx] += self.V0[m] - old_value

    def _prec_obs_step(self) -> None:
        if self.n_obs == 0:
            self.prec = np.random.gamma(self.a0, 1.0 / self.b0)
            return
        # self._reconstruct_Mu()
        sse = np.square(self.y - self.Mu).sum()
        an = self.a0 + 0.5 * self.n_obs
        bn = self.b0 + 0.5 * sse
        self.prec = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.last_rmse = np.sqrt(np.square(self.y - self.Mu).mean())
        self.prec = np.clip(self.prec, C, 1e6)

    def _prec_V2_step(self):
        if self.local_shrinkage:
            phiaux2 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi2))
            bn = phiaux2 + 0.5 * self.eta2 * self.V2**2
            self.phi2 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array([(self.dd1 == c).sum() for c in range(self.n_doses)])
            N2 = np.array([(self.dd2 == c).sum() for c in range(self.n_doses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi2 = np.clip(self.phi2, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_doses)
        etaaux2 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta2))
        bn = etaaux2 + 0.5 * (self.phi2 * self.V2**2).sum(0)
        self.eta2 = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta2 = np.clip(self.eta2, C, 1e6)

    def _prec_V1_step(self):
        if self.local_shrinkage:
            phiaux1 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi1))
            bn = phiaux1 + 0.5 * self.eta1 * self.V1**2
            self.phi1 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))
            N1 = np.array([(self.dd1 == c).sum() for c in range(self.n_doses)])
            N2 = np.array([(self.dd2 == c).sum() for c in range(self.n_doses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi1 = np.clip(self.phi1, C[:, None], 1e6)

        an = 0.5 * (1 + self.n_doses)
        etaaux1 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta1))
        bn = etaaux1 + 0.5 * (self.phi1 * self.V1**2).sum(0)
        self.eta1 = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta1 = np.clip(self.eta1, C, 1e6)

    def _prec_V0_step(self):
        if self.local_shrinkage:
            phiaux0 = np.random.gamma(1.0, 1.0 / (1.0 + self.phi0))
            bn = phiaux0 + 0.5 * self.eta0 * self.V0**2
            self.phi0 = np.random.gamma(1.0, 1.0 / (bn + 1e-3))  # +0.01 for stability
            N1 = np.array([(self.dd1 == c).sum() for c in range(self.n_doses)])
            N2 = np.array([(self.dd2 == c).sum() for c in range(self.n_doses)])
            C = 1.0 / np.sqrt(1.0 + N1 + N2)
            self.phi0 = np.clip(self.phi0, C, 1e6)

        # V0 precision
        an = 0.5 * (1 + self.n_doses)
        etaaux0 = np.random.gamma(1.0, 1.0 / (1.0 + self.eta0))
        bn = etaaux0 + 0.5 * (self.phi0 * self.V0**2).sum()
        self.eta0 = np.random.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.eta0 = np.clip(self.eta0, C, 1e6)

    def _prec_W_step(self):
        # gamma process
        parssq = self.W**2
        if self.mult_gamma_proc:
            tmp = np.cumprod(self.gam) / self.gam[0]
            an = 2 + 0.5 * self.n_samples * self.n_embedding_dimensions
            bn = 1 + 0.5 * (tmp * parssq).sum()
            self.gam[0] = np.random.gamma(an, 1.0 / (bn + 1e-3))
            # self.gam[0] = np.clip(self.gam[0], 1.0, 10000.0)
            # sample all others
            for d in range(1, self.n_embedding_dimensions):
                tmp = np.cumprod(self.gam)[d:] / self.gam[d]
                an = 3 + 0.5 * self.n_samples * (self.n_embedding_dimensions - d)
                bn = 1 + 0.5 * (tmp * parssq[:, d:]).sum()
                self.gam[d] = np.random.gamma(an, 1.0 / (bn + 1e-3))
                # self.gam[d] = np.clip(self.gam[d], 0.5, 100.0)
            self.tau = np.cumprod(self.gam)
        else:
            an = self.a0 + 0.5 * self.n_samples
            bn = self.b0 + 0.5 * parssq.sum(0)
            self.tau = np.random.gamma(an, 1.0 / (bn + 1e-3))
        C = 1.0 / np.sqrt(1 + self.n_obs)
        self.tau = np.clip(self.tau, C, 1e6)

    def _prec_W0_step(self):
        # W0 precision
        an = self.a0 + 0.5 * self.n_samples
        bn = self.b0 + 0.5 * (self.W0**2).sum()
        self.tau0 = np.random.gamma(an, 1.0 / (bn + 1e-3))  # +0.01 for stability
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
            y, *_ = self.encode_obs()
            mean = (y - self.Mu + self.alpha).mean()
            stddev = 1.0 / np.sqrt(self.n_obs)
            self.alpha = np.random.normal(mean, stddev)
        self.Mu += self.alpha - old_value

    def mcmc_step(self) -> None:
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

    def _reconstruct_Mu(self, clip=True):
        if self.n_obs == 0:
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

    def predict(self, cline: ArrayType, dd1: ArrayType, dd2: ArrayType):
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

    def predict_single_drug(self, cline: ArrayType, dd1: ArrayType):
        interaction1 = np.sum(self.W[cline] * self.get("V1", dd1), -1)
        intercept = self.alpha + self.W0[cline] + self.get("V0", dd1)
        Mu = intercept + interaction1
        return Mu

    def bliss(self, cline: ArrayType, dd1: ArrayType, dd2: ArrayType):
        interaction2 = np.sum(
            self.W[cline] * self.get("V2", dd1) * self.get("V2", dd2), -1
        )
        return interaction2

    def ess_pars(self):
        return [1.0 / np.sqrt(self.prec)] + self.Mu.tolist()
