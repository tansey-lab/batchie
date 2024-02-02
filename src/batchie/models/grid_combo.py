from batchie.common import ArrayType
import numpy as np
import math
from dataclasses import dataclass
import h5py

from batchie.core import (
    BayesianModel,
    VIModel,
    Theta,
    ThetaHolder,
)
from numpy.random._generator import Generator as Generator
import torch
from torch.nn.functional import softplus
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from batchie.data import ScreenBase
from batchie.models.grid_helper import (
    unpack_data,
    interp_01_vals,
    BatchIterator,
    ConcentrationGrid,
)
from tqdm import tqdm


@dataclass
class GridComboSample(Theta):
    """A single sample from the posterior for the grid drug combo model"""

    sample_embeddings: torch.FloatTensor
    single_drug_embeddings: torch.FloatTensor
    combo_drug_embeddings: torch.FloatTensor

    sample_sigma_embeddings: torch.FloatTensor
    single_drug_sigma_embeddings: torch.FloatTensor
    combo_drug_sigma_embeddings: torch.FloatTensor

    mean_obs_sigma: torch.FloatTensor
    obs_sigma: float

    drugname2idx: dict[str, int]
    conc_grid: ConcentrationGrid
    batch_size: int

    def private_parameters_dict(self) -> dict[str, ArrayType]:
        params = {
            "sample_embeddings": self.sample_embeddings.detach().numpy(),
            "single_drug_embeddings": self.single_drug_embeddings.detach().numpy(),
            "combo_drug_embeddings": self.combo_drug_embeddings.detach().numpy(),
            "sample_sigma_embeddings": self.sample_sigma_embeddings.detach().numpy(),
            "single_drug_sigma_embeddings": self.single_drug_sigma_embeddings.detach().numpy(),
            "combo_drug_sigma_embeddings": self.combo_drug_sigma_embeddings.detach().numpy(),
            "mean_obs_sigma": self.mean_obs_sigma.detach().numpy(),
            "obs_sigma": self.obs_sigma,
        }
        return params

    def shared_parameters_dict(self) -> dict[str, ArrayType]:
        params = {
            "drugname2idx_keys": np.array(list(self.drugname2idx.keys())).astype("S"),
            "drugname2idx_vals": np.array(list(self.drugname2idx.values())).astype(int),
            "conc_grid": self.conc_grid.conc_grid.numpy(),
            "batch_size": int,
        }
        return params

    @classmethod
    def from_dicts(cls, private_params, shared_params):
        drugname2idx_keys = shared_params["drugname2idx_keys"].astype(str)
        drugname2idx = dict(zip(drugname2idx_keys, shared_params["drugname2idx_vals"]))
        conc_grid = ConcentrationGrid(
            torch.from_numpy(shared_params["conc_grid"]).float()
        )
        result = cls(
            sample_embeddings=torch.from_numpy(
                private_params["sample_embeddings"]
            ).float(),
            single_drug_embeddings=torch.from_numpy(
                private_params["single_drug_embeddings"]
            ).float(),
            combo_drug_embeddings=torch.from_numpy(
                private_params["combo_drug_embeddings"]
            ).float(),
            sample_sigma_embeddings=torch.from_numpy(
                private_params["sample_sigma_embeddings"]
            ).float(),
            single_drug_sigma_embeddings=torch.from_numpy(
                private_params["single_drug_sigma_embeddings"]
            ).float(),
            combo_drug_sigma_embeddings=torch.from_numpy(
                private_params["combo_drug_sigma_embeddings"]
            ).float(),
            mean_obs_sigma=torch.from_numpy(private_params["mean_obs_sigma"]).float(),
            obs_sigma=private_params["obs_sigma"],
            drugname2idx=drugname2idx,
            conc_grid=conc_grid,
            batch_size=shared_params["batch_size"],
        )
        return result

    def predict_viability(self, data: ScreenBase) -> ArrayType:
        return self.predict_mean(
            data
        )  ## Probability space = viability space for this model

    def predict_mean(self, data: ScreenBase) -> ArrayType:
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = unpack_data(
            data=data, drugname2idx=self.drugname2idx, use_mask=False
        )

        ## Organize data
        sample_ids = torch.from_numpy(sample_ids).long()
        drug_ids_1 = torch.from_numpy(drug_ids_1).long()
        drug_ids_2 = torch.from_numpy(drug_ids_2).long()

        log_conc1 = torch.from_numpy(log_conc1).float()
        log_conc2 = torch.from_numpy(log_conc2).float()
        log_conc2 = torch.nan_to_num(log_conc2)  ## Just 0 out the nans

        ub_idx_1, lin_p_1 = self.conc_grid.lookup_conc(
            log_concs=log_conc1, drug_ids=drug_ids_1
        )
        ub_idx_2, lin_p_2 = self.conc_grid.lookup_conc(
            log_concs=log_conc2, drug_ids=drug_ids_2
        )

        n_data = sample_ids.shape[0]

        batch_iterator = BatchIterator(
            sample_ids,
            drug_ids_1,
            drug_ids_2,
            ub_idx_1,
            ub_idx_2,
            lin_p_1,
            lin_p_2,
            n_grid=self.conc_grid.n_grid,
            batch_size=self.batch_size,
            n_epochs=1,
            shuffle=False,
            combo_smooth=False,
        )

        mus = []
        for s_ids, d_ids_1, d_ids_2, u_idx_1, u_idx_2, l_p_1, l_p_2 in batch_iterator:
            samp = self.sample_embeddings[s_ids]
            sdrug1 = self.single_drug_embeddings[d_ids_1]
            sdrug2 = self.single_drug_embeddings[d_ids_2]

            cdrug1 = self.combo_drug_embeddings[d_ids_1]
            cdrug2 = self.combo_drug_embeddings[d_ids_2]

            single_mask = d_ids_2 < 0

            mu = mean_combo(
                sample_embeddings=samp,
                single_drug_embeddings_1=sdrug1,
                single_drug_embeddings_2=sdrug2,
                combo_drug_embeddings_1=cdrug1,
                combo_drug_embeddings_2=cdrug2,
                ub_idx_1=u_idx_1,
                ub_idx_2=u_idx_2,
                lin_p_1=l_p_1,
                lin_p_2=l_p_2,
                single_mask=single_mask,
            )
            mus.append(mu)
        mu = torch.concat(mus)
        mu = mu.detach().numpy()
        return mu

    def predict_variance(self, data: ScreenBase) -> ArrayType:
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = unpack_data(
            data=data, drugname2idx=self.drugname2idx, use_mask=False
        )

        ## Organize data
        sample_ids = torch.from_numpy(sample_ids).long()
        drug_ids_1 = torch.from_numpy(drug_ids_1).long()
        drug_ids_2 = torch.from_numpy(drug_ids_2).long()

        log_conc1 = torch.from_numpy(log_conc1).float()
        log_conc2 = torch.from_numpy(log_conc2).float()
        log_conc2 = torch.nan_to_num(log_conc2)  ## Just 0 out the nans

        mu = torch.from_numpy(self.predict_mean(data)).float()

        n_data = sample_ids.shape[0]

        batch_iterator = BatchIterator(
            sample_ids,
            drug_ids_1,
            drug_ids_2,
            mu,
            n_grid=self.n_grid,
            batch_size=self.batch_size,
            n_epochs=1,
            shuffle=False,
            combo_smooth=False,
        )

        sigmas = []
        for s_ids, d_ids_1, d_ids_2, m in batch_iterator:
            samp = self.sample_sigma_embeddings[s_ids]
            sdrug1 = self.single_drug_sigma_embeddings[d_ids_1]
            sdrug2 = self.single_drug_sigma_embeddings[d_ids_2]

            cdrug1 = self.combo_drug_sigma_embeddings[d_ids_1]
            cdrug2 = self.combo_drug_sigma_embeddings[d_ids_2]

            single_mask = d_ids_2 < 0

            sigma = scale_combo(
                sample_sigma_embeddings=samp,
                single_drug_sigma_embeddings_1=sdrug1,
                single_drug_sigma_embeddings_2=sdrug2,
                combo_drug_sigma_embeddings_1=cdrug1,
                combo_drug_sigma_embeddings_2=cdrug2,
                obs_sigma=self.obs_sigma,
                mu=m,
                mean_obs_sigma=self.mean_obs_sigma,
                single_mask=single_mask,
                n_grid=self.conc_grid.n_grid,
            )

            sigmas.append(sigma)
        var = torch.square(torch.concat(sigmas))
        var = var.detach().numpy()
        return var


"""
    Batch size B, grid size n, embedding dimension d.
    sample_embeddings: <B x 1 x d >
    drug_embeddings: <B x n-1 x d >
    output: <B x n>
"""


def predict_grid(
    sample_embeddings: torch.FloatTensor, drug_embeddings: torch.FloatTensor
):
    N, *_ = sample_embeddings.shape

    interval_grid = torch.softmax(
        (sample_embeddings * drug_embeddings).sum(dim=-1), dim=-1
    )  ## N x n_grid-1
    cum_sum = torch.cumsum(interval_grid, dim=1)

    ## Prediction over entire grid
    pred_grid = torch.concatenate(
        ((interval_grid - cum_sum + 1.0), torch.zeros(N, 1)), dim=1
    )  ## N x n_grid

    pred_grid = torch.clip(pred_grid, 0, 1)
    return pred_grid


def predict_single(
    sample_embeddings: torch.FloatTensor,
    single_drug_embeddings: torch.FloatTensor,
    ub_idx: torch.LongTensor,
    lin_p: torch.FloatTensor,
):
    single_pred_grid = predict_grid(sample_embeddings, single_drug_embeddings)
    single_effect = (
        lin_p * single_pred_grid.gather(1, ub_idx).squeeze()
        + (1.0 - lin_p) * single_pred_grid.gather(1, (ub_idx - 1)).squeeze()
    )
    return single_effect


def predict_synergy(
    sample_embeddings: torch.FloatTensor,
    combo_drug_embeddings_1: torch.FloatTensor,
    combo_drug_embeddings_2: torch.FloatTensor,
    ub_idx_1: torch.LongTensor,
    ub_idx_2: torch.LongTensor,
    lin_p_1: torch.FloatTensor,
    lin_p_2: torch.FloatTensor,
):
    index = torch.arange(sample_embeddings.shape[0])
    S = sample_embeddings.squeeze()
    U1 = combo_drug_embeddings_1[index, ub_idx_1.squeeze(), :]
    U2 = combo_drug_embeddings_2[index, ub_idx_2.squeeze(), :]
    L1 = combo_drug_embeddings_1[index, (ub_idx_1 - 1).squeeze(), :]
    L2 = combo_drug_embeddings_2[index, (ub_idx_2 - 1).squeeze(), :]

    u1u2 = (S * U1 * U2).sum(dim=-1).squeeze()
    u1l2 = (S * U1 * L2).sum(dim=-1).squeeze()
    l1u2 = (S * L1 * U2).sum(dim=-1).squeeze()
    l1l2 = (S * L1 * L2).sum(dim=-1).squeeze()

    synergy_effect = (
        (lin_p_1 * lin_p_2 * u1u2)
        + (lin_p_1 * (1.0 - lin_p_2) * u1l2)
        + ((1.0 - lin_p_1) * lin_p_2 * l1u2)
        + ((1.0 - lin_p_1) * (1.0 - lin_p_2) * l1l2)
    )
    return synergy_effect


def mean_combo(
    sample_embeddings: torch.FloatTensor,
    single_drug_embeddings_1: torch.FloatTensor,
    single_drug_embeddings_2: torch.FloatTensor,
    combo_drug_embeddings_1: torch.FloatTensor,
    combo_drug_embeddings_2: torch.FloatTensor,
    ub_idx_1: torch.LongTensor,
    ub_idx_2: torch.LongTensor,
    lin_p_1: torch.FloatTensor,
    lin_p_2: torch.FloatTensor,
    single_mask: torch.BoolTensor,
):
    single_effect_1 = predict_single(
        sample_embeddings=sample_embeddings,
        single_drug_embeddings=single_drug_embeddings_1,
        ub_idx=ub_idx_1,
        lin_p=lin_p_1,
    )

    single_effect_2 = predict_single(
        sample_embeddings=sample_embeddings,
        single_drug_embeddings=single_drug_embeddings_2,
        ub_idx=ub_idx_2,
        lin_p=lin_p_2,
    )

    synergy_effect = predict_synergy(
        sample_embeddings=sample_embeddings,
        combo_drug_embeddings_1=combo_drug_embeddings_1,
        combo_drug_embeddings_2=combo_drug_embeddings_2,
        ub_idx_1=ub_idx_1,
        ub_idx_2=ub_idx_2,
        lin_p_1=lin_p_1,
        lin_p_2=lin_p_2,
    )

    combo_effect = single_effect_1 * single_effect_2 + synergy_effect

    mu = torch.where(single_mask, single_effect_1, combo_effect)
    mu = torch.clip(mu, min=0.0, max=1.0)
    return mu


def scale_combo(
    sample_sigma_embeddings: torch.FloatTensor,
    single_drug_sigma_embeddings_1: torch.FloatTensor,
    single_drug_sigma_embeddings_2: torch.FloatTensor,
    combo_drug_sigma_embeddings_1: torch.FloatTensor,
    combo_drug_sigma_embeddings_2: torch.FloatTensor,
    mu: torch.FloatTensor,
    mean_obs_sigma: torch.FloatTensor,
    obs_sigma: float,
    single_mask: torch.BoolTensor,
    n_grid: int,
):
    ## Combination effects
    combo_sigma_raw = torch.sum(
        sample_sigma_embeddings
        * combo_drug_sigma_embeddings_1
        * combo_drug_sigma_embeddings_2,
        dim=-1,
    )

    single_sigma_raw = torch.sum(
        sample_sigma_embeddings * single_drug_sigma_embeddings_1, dim=-1
    )

    sigma_raw = torch.where(single_mask, single_sigma_raw, combo_sigma_raw)
    sigma = obs_sigma + softplus(sigma_raw)

    k_upper, k_lower, p = interp_01_vals(mu, n_grid)
    mean_level_sigma = p * mean_obs_sigma[k_upper] + (1 - p) * mean_obs_sigma[k_lower]
    sigma = sigma * mean_level_sigma
    return sigma


class ComboGridFactorModel(BayesianModel, VIModel):
    def __init__(
        self,
        n_unique_samples: int,
        unique_drug_names: np.ndarray,
        log_conc_range: dict | tuple,
        n_grid: int,
        n_embedding_dimensions: int,
        n_sigma_embedding_dimensions: int,
        log_conc_padding: float = 2.0,
        likelihood: str = "normal",
        smooth_scale: float = 1.0,
        guide_type: str = "normal",
        lr: float = 0.001,
        n_epochs=10,
        batch_size=50000,
        min_steps=1000,
        max_steps=None,
    ):
        uniq_drug_names = np.unique(unique_drug_names)
        self.drugname2idx = {x: i for i, x in enumerate(uniq_drug_names)}

        self.n_samples = n_unique_samples
        self.n_drugs = len(uniq_drug_names)

        self.n_dims = n_embedding_dimensions
        self.n_sigma_dims = n_sigma_embedding_dimensions

        self.n_grid = n_grid
        self.conc_grid = ConcentrationGrid.init_from_range(
            log_conc_range, self.drugname2idx, n_grid, log_conc_padding=log_conc_padding
        )

        self.likelihood = likelihood
        self.smooth_scale = smooth_scale
        self.guide_type = guide_type
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.grid_scales = self.conc_grid.grid_scale()

        self.sample_ids = np.array([], dtype=np.int32)
        self.drug_ids_1 = np.array([], dtype=np.int32)
        self.drug_ids_2 = np.array([], dtype=np.int32)
        self.log_concs_1 = np.array([], dtype=np.float32)
        self.log_concs_2 = np.array([], dtype=np.float32)
        self.y = np.array([], dtype=np.float32)

    def reset_model(self):
        self.auto_guide = None
        pyro.clear_param_store()

    def n_obs(self) -> int:
        return self.sample_ids.shape[0]

    def rng(self):
        return self.rng_

    def set_rng(self, rng: Generator):
        self.rng_ = rng

    def smoothing_combo_prediction(
        self,
        sample_embeddings: torch.FloatTensor,
        drug_ids1: torch.LongTensor,
        drug_ids2: torch.LongTensor,
        combo_drug_embeddings_1: torch.FloatTensor,
        combo_drug_embeddings_2: torch.FloatTensor,
        conc_combo_idxs: torch.LongTensor,
    ):
        n = sample_embeddings.shape[0]
        n_combo_conc = combo_drug_embeddings_1.shape[1]

        ## Unpack combos
        conc_combo_idx1 = conc_combo_idxs[:, 0]
        conc_combo_idx2 = conc_combo_idxs[:, 1]

        ## Unpack the combinations into individual conc indices
        conc_idx11 = conc_combo_idx1 // n_combo_conc
        conc_idx12 = conc_combo_idx1 % n_combo_conc
        conc_idx21 = conc_combo_idx2 // n_combo_conc
        conc_idx22 = conc_combo_idx2 % n_combo_conc

        ## Get associated concentrations
        concs11 = self.conc_grid.conc_grid[drug_ids1, conc_idx11]
        concs12 = self.conc_grid.conc_grid[drug_ids2, conc_idx12]

        concs21 = self.conc_grid.conc_grid[drug_ids1, conc_idx21]
        concs22 = self.conc_grid.conc_grid[drug_ids2, conc_idx22]

        ## Get the concentration scale of each drug
        conc_scale1 = self.grid_scales[drug_ids1]
        conc_scale2 = self.grid_scales[drug_ids2]

        ## Compute the scaled distance between the concentration pairs
        conc_dist = (
            torch.abs(concs11 - concs21) / conc_scale1
            + torch.abs(concs12 - concs22) / conc_scale2
        )
        conc_dist = conc_dist / self.smooth_scale + 1e-7

        ## Compute the synergy predictions for the two combinations
        index = torch.arange(n)
        S = sample_embeddings.squeeze()
        C11 = combo_drug_embeddings_1[index, conc_idx11, :]
        C12 = combo_drug_embeddings_2[index, conc_idx12, :]

        C21 = combo_drug_embeddings_1[index, conc_idx21, :]
        C22 = combo_drug_embeddings_2[index, conc_idx22, :]

        synergy_effect1 = (S * C11 * C12).sum(dim=-1).squeeze()
        synergy_effect2 = (S * C21 * C22).sum(dim=-1).squeeze()

        synergy_diff = synergy_effect1 - synergy_effect2
        return (synergy_diff, conc_dist)

    def model(self, batch: tuple, n_data: int):
        conc_combo_idxs = None
        obs = None
        if len(batch) == 9:
            (
                sample_ids,
                drug_ids_1,
                drug_ids_2,
                ub_idx_1,
                ub_idx_2,
                lin_p_1,
                lin_p_2,
                obs,
                conc_combo_idxs,
            ) = batch
        elif len(batch) == 8:
            (
                sample_ids,
                drug_ids_1,
                drug_ids_2,
                ub_idx_1,
                ub_idx_2,
                lin_p_1,
                lin_p_2,
                obs,
            ) = batch
        elif len(batch) == 7:
            (
                sample_ids,
                drug_ids_1,
                drug_ids_2,
                ub_idx_1,
                ub_idx_2,
                lin_p_1,
                lin_p_2,
            ) = batch

        batch_size = sample_ids.shape[0]

        obs_sigma = pyro.sample("obs_sigma", dist.Gamma(1.0, 1.0))

        sample_embed_sigma = pyro.sample("sample_embed_sigma", dist.Gamma(1.0, 1.0))
        sample_embed = pyro.sample(
            "sample_embed",
            dist.Normal(
                torch.zeros(self.n_samples, 1, self.n_dims), sample_embed_sigma
            ).to_event(3),
        )

        single_drug_embed_sigma = pyro.sample(
            "single_drug_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        single_drug_embed = pyro.sample(
            "single_drug_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.n_grid - 1, self.n_dims),
                single_drug_embed_sigma,
            ).to_event(3),
        )

        combo_drug_embed_sigma = pyro.sample(
            "combo_drug_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        combo_drug_embed = pyro.sample(
            "combo_drug_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.n_grid, self.n_dims),
                combo_drug_embed_sigma,
            ).to_event(3),
        )

        ## Scale factor for variance sample embeddings
        sample_sigma_embed_sigma = pyro.sample(
            "sample_sigma_embed_sigma", dist.Gamma(1.0, 1.0)
        )

        ## Variance sample embeddings
        sample_sigma_embed = pyro.sample(
            "sample_sigma_embed",
            dist.Normal(
                torch.zeros(self.n_samples, self.n_sigma_dims), sample_sigma_embed_sigma
            ).to_event(2),
        )

        ## Scale factor for variance drug embeddings
        single_drug_sigma_embed_sigma = pyro.sample(
            "single_drug_sigma_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        single_drug_sigma_embed = pyro.sample(
            "single_drug_sigma_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.n_sigma_dims),
                single_drug_sigma_embed_sigma,
            ).to_event(2),
        )

        combo_drug_sigma_embed_sigma = pyro.sample(
            "combo_drug_sigma_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        combo_drug_sigma_embed = pyro.sample(
            "combo_drug_sigma_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.n_sigma_dims),
                combo_drug_sigma_embed_sigma,
            ).to_event(2),
        )

        ## Per-mean value observational noise
        mean_obs_sigma = pyro.sample(
            "mean_obs_sigma",
            dist.Gamma(torch.ones(self.n_grid), torch.ones(self.n_grid)).to_event(1),
        )

        with pyro.plate("viability_plate", size=n_data, subsample_size=batch_size):
            single_mask = drug_ids_2 < 0

            samp = sample_embed[sample_ids]
            sdrug1 = single_drug_embed[drug_ids_1]
            sdrug2 = single_drug_embed[drug_ids_2]
            cdrug1 = combo_drug_embed[drug_ids_1]
            cdrug2 = combo_drug_embed[drug_ids_2]
            ub1 = ub_idx_1
            ub2 = ub_idx_2
            lin1 = lin_p_1
            lin2 = lin_p_2
            mu = mean_combo(
                sample_embeddings=samp,
                single_drug_embeddings_1=sdrug1,
                single_drug_embeddings_2=sdrug2,
                combo_drug_embeddings_1=cdrug1,
                combo_drug_embeddings_2=cdrug2,
                ub_idx_1=ub1,
                ub_idx_2=ub2,
                lin_p_1=lin1,
                lin_p_2=lin2,
                single_mask=single_mask,
            )

            sigma = scale_combo(
                sample_sigma_embeddings=sample_sigma_embed[sample_ids],
                single_drug_sigma_embeddings_1=single_drug_sigma_embed[drug_ids_1],
                single_drug_sigma_embeddings_2=single_drug_sigma_embed[drug_ids_2],
                combo_drug_sigma_embeddings_1=combo_drug_sigma_embed[drug_ids_1],
                combo_drug_sigma_embeddings_2=combo_drug_sigma_embed[drug_ids_2],
                obs_sigma=obs_sigma,
                mu=mu,
                mean_obs_sigma=mean_obs_sigma,
                single_mask=single_mask,
                n_grid=self.n_grid,
            )

            viab = pyro.sample("viability", dist.Normal(mu, sigma), obs=obs)

            if (self.smooth_scale > 0.0) and (conc_combo_idxs is not None):
                synergy_diff, conc_dist = self.smoothing_combo_prediction(
                    sample_embeddings=samp,
                    drug_ids1=drug_ids_1,
                    drug_ids2=drug_ids_2,
                    combo_drug_embeddings_1=cdrug1,
                    combo_drug_embeddings_2=cdrug2,
                    conc_combo_idxs=conc_combo_idxs,
                )

                zeros = torch.zeros_like(synergy_diff, dtype=synergy_diff.dtype)
                diff = pyro.sample(
                    "synergy_diff", dist.Laplace(synergy_diff, conc_dist), obs=zeros
                )
        return mu

    def _add_observations(self, data: ScreenBase):
        if not (data.observations >= 0.0).all():
            raise ValueError(
                "Observations should be non-negative, please check input data"
            )
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = unpack_data(
            data=data, drugname2idx=self.drugname2idx, use_mask=True
        )

        self.sample_ids = np.concatenate([self.sample_ids, sample_ids])

        self.log_concs_1 = np.concatenate([self.log_concs_1, log_conc1])
        self.log_concs_2 = np.concatenate([self.log_concs_2, log_conc2])

        self.drug_ids_1 = np.concatenate([self.drug_ids_1, drug_ids_1])
        self.drug_ids_2 = np.concatenate([self.drug_ids_2, drug_ids_2])

        mask = data.observation_mask
        self.y = np.concatenate(
            [self.y, np.clip(data.observations[mask], a_min=0.0, a_max=1.0)]
        )

    def fit(self):
        ## TODO: Figure out a way to avoid this line
        pyro.clear_param_store()
        self.auto_guide = AutoLowRankMultivariateNormal(self.model)

        ## Organize data
        sample_ids = torch.from_numpy(self.sample_ids).long()
        drug_ids_1 = torch.from_numpy(self.drug_ids_1).long()
        drug_ids_2 = torch.from_numpy(self.drug_ids_2).long()

        log_conc1 = torch.from_numpy(self.log_concs_1).float()
        log_conc2 = torch.from_numpy(self.log_concs_2).float()
        log_conc2 = torch.nan_to_num(log_conc2)  ## Just 0 out the nans

        ub_idx_1, lin_p_1 = self.conc_grid.lookup_conc(
            log_concs=log_conc1, drug_ids=self.drug_ids_1
        )
        ub_idx_2, lin_p_2 = self.conc_grid.lookup_conc(
            log_concs=log_conc2, drug_ids=self.drug_ids_2
        )

        obs = torch.from_numpy(self.y).float()

        n_data = sample_ids.shape[0]

        adam = pyro.optim.Adam({"lr": self.lr})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.auto_guide, adam, elbo)

        batch_iterator = BatchIterator(
            sample_ids,
            drug_ids_1,
            drug_ids_2,
            ub_idx_1,
            ub_idx_2,
            lin_p_1,
            lin_p_2,
            obs,
            n_grid=self.n_grid,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
            shuffle=True,
            combo_smooth=True,
        )

        pbar = tqdm(batch_iterator, total=len(batch_iterator))
        losses = []
        for step, batch in enumerate(pbar):
            loss = svi.step(batch=batch, n_data=n_data)
            losses.append(loss)
            if step % 25 == 0:
                pbar.set_postfix({"Elbo loss": loss})

        return losses

    def sample(self, num_samples: int) -> list[GridComboSample]:
        losses = self.fit()

        predictive = Predictive(
            self.model,
            guide=self.auto_guide,
            num_samples=num_samples,
            return_sites=(
                "sample_embed",
                "combo_drug_embed",
                "single_drug_embed",
                "mean_obs_sigma",
                "sample_sigma_embed",
                "single_drug_sigma_embed",
                "combo_drug_sigma_embed",
                "obs_sigma",
            ),
        )

        batch = (
            torch.LongTensor([0, 0]),
            torch.LongTensor([0, 0]),
            torch.LongTensor([1, 1]),
            torch.LongTensor([[1], [1]]),
            torch.LongTensor([[1], [1]]),
            torch.FloatTensor([0.5, 0.5]),
            torch.FloatTensor([0.5, 0.5]),
        )

        res = predictive(batch=batch, n_data=2)

        obs_sigma = res["obs_sigma"].squeeze()
        mean_obs_sigma = res["mean_obs_sigma"].squeeze()

        sample_embed = res["sample_embed"].squeeze(1)
        combo_drug_embed = res["combo_drug_embed"].squeeze()
        single_drug_embed = res["single_drug_embed"].squeeze()

        sample_sigma_embed = res["sample_sigma_embed"].squeeze()
        combo_drug_sigma_embed = res["combo_drug_sigma_embed"].squeeze()
        single_drug_sigma_embed = res["single_drug_sigma_embed"].squeeze()

        samples = []
        for i in range(num_samples):
            x = GridComboSample(
                sample_embeddings=sample_embed[i].detach().clone(),
                single_drug_embeddings=single_drug_embed[i].detach().clone(),
                combo_drug_embeddings=combo_drug_embed[i].detach().clone(),
                sample_sigma_embeddings=sample_sigma_embed[i].detach().clone(),
                single_drug_sigma_embeddings=single_drug_sigma_embed[i]
                .detach()
                .clone(),
                combo_drug_sigma_embeddings=combo_drug_sigma_embed[i].detach().clone(),
                mean_obs_sigma=mean_obs_sigma[i].detach().clone(),
                obs_sigma=obs_sigma[i].detach().clone(),
                drugname2idx=self.drugname2idx,
                conc_grid=self.conc_grid,
                batch_size=self.batch_size,
            )
            samples.append(x)
        return samples
