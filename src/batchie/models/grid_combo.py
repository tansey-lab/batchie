from batchie.common import ArrayType
import numpy as np
import math
from dataclasses import dataclass
import h5py

from batchie.core import BayesianModel, Theta, ThetaHolder
import torch
from torch.nn.functional import softplus
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from batchie.data import ScreenBase
from batchie.models.grid_helper import interp_01_vals, BatchIterator, ConcentrationGrid
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


class GridComboResults(ThetaHolder):
    def __init__(
        self,
        n_unique_samples: int,
        n_unique_drugs: int,
        n_grid: int,
        n_embedding_dimensions: int,
        n_sigma_embedding_dimensions: int,
        n_thetas: int,
    ):
        super().__init__(n_thetas)
        self.n_unique_samples = n_unique_samples
        self.n_unique_drugs = n_unique_drugs
        self.n_grid = n_grid
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_sigma_embedding_dimensions = n_sigma_embedding_dimensions

        self.sample_embeddings = torch.zeros(
            n_thetas, self.n_unique_samples, 1, self.n_embedding_dimensions
        )
        self.single_drug_embeddings = torch.zeros(
            n_thetas, self.n_unique_drugs, self.n_grid - 1, self.n_embedding_dimensions
        )
        self.combo_drug_embeddings = torch.zeros(
            n_thetas, self.n_unique_drugs, self.n_grid - 1, self.n_embedding_dimensions
        )

        self.sample_sigma_embeddings = torch.zeros(
            n_thetas, self.n_unique_samples, 1, self.n_sigma_embedding_dimensions
        )
        self.single_drug_sigma_embeddings = torch.zeros(
            n_thetas,
            self.n_unique_drugs,
            self.n_grid - 1,
            self.n_sigma_embedding_dimensions,
        )
        self.combo_drug_sigma_embeddings = torch.zeros(
            n_thetas,
            self.n_unique_drugs,
            self.n_grid - 1,
            self.n_sigma_embedding_dimensions,
        )

        self.mean_obs_sigma = torch.zeros(n_thetas, self.n_grid)

    def combine(self, other):
        if type(self) != type(other):
            raise ValueError("Cannot combine with different type")

        if self.n_embedding_dimensions != other.n_embedding_dimensions:
            raise ValueError("Cannot combine with different embedding dimensions")

        if self.n_sigma_embedding_dimensions != other.n_sigma_embedding_dimensions:
            raise ValueError("Cannot combine with different sigma embedding dimensions")

        if self.n_grid != other.n_grid:
            raise ValueError("Cannot combine with different number of grid points")

        if self.n_unique_samples != other.n_unique_samples:
            raise ValueError("Cannot combine with different number of unique samples")

        if self.n_unique_drugs != other.n_unique_drugs:
            raise ValueError("Cannot combine with different number of unique drugs")

        output = GridComboResults(
            n_unique_samples=self.n_unique_samples,
            n_unique_drugs=self.n_unique_drugs,
            n_grid=self.n_grid,
            n_embedding_dimensions=self.n_embedding_dimensions,
            n_sigma_embedding_dimensions=self.n_sigma_embedding_dimensions,
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

    def get_theta(self, step_index: int) -> GridComboSample:
        # Test if this is beyond the step we are current at with the cursor
        if step_index >= self._cursor:
            raise ValueError("Cannot get a step beyond the current cursor position")

        return GridComboSample(
            sample_embeddings=self.sample_embeddings[step_index],
            single_drug_embeddings=self.single_drug_embeddings[step_index],
            combo_drug_embeddings=self.combo_drug_embeddings[step_index],
            single_drug_sigma_embeddings=self.single_drug_sigma_embeddings[step_index],
            combo_drug_sigma_embeddings=self.combo_drug_sigma_embeddings[step_index],
            sample_sigma_embeddings=self.sample_sigma_embeddings[step_index],
            mean_obs_sigma=self.mean_obs_sigma[step_index],
        )

    def get_variance(self, step_index: int) -> float:
        return 1.0

    def _save_theta(self, sample: GridComboSample, variance: float, sample_index: int):
        self.sample_embeddings[sample_index] = sample.sample_embeddings
        self.single_drug_embeddings[sample_index] = sample.single_drug_embeddings
        self.combo_drug_embeddings[sample_index] = sample.combo_drug_embeddings
        self.single_drug_sigma_embeddings[
            sample_index
        ] = sample.single_drug_sigma_embeddings
        self.combo_drug_sigma_embeddings[
            sample_index
        ] = sample.combo_drug_sigma_embeddings
        self.sample_sigma_embeddings[sample_index] = sample.sample_sigma_embeddings
        self.mean_obs_sigma[sample_index] = sample.mean_obs_sigma

    def save_h5(self, fn: str):
        with h5py.File(fn, "w") as f:
            f.create_dataset(
                "sample_embeddings", data=self.sample_embeddings, compression="gzip"
            )
            f.create_dataset(
                "single_drug_embeddings",
                data=self.single_drug_embeddings,
                compression="gzip",
            )
            f.create_dataset(
                "combo_drug_embeddings",
                data=self.combo_drug_embeddings,
                compression="gzip",
            )
            f.create_dataset(
                "single_drug_sigma_embeddings",
                data=self.single_drug_sigma_embeddings,
                compression="gzip",
            )
            f.create_dataset(
                "combo_drug_sigma_embeddings",
                data=self.combo_drug_sigma_embeddings,
                compression="gzip",
            )
            f.create_dataset(
                "sample_sigma_embeddings",
                data=self.sample_sigma_embeddings,
                compression="gzip",
            )
            f.create_dataset(
                "mean_obs_sigma", data=self.mean_obs_sigma, compression="gzip"
            )

            # Save the cursor value metadata
            f.attrs["cursor"] = self._cursor

    @staticmethod
    def load_h5(path: str):
        with h5py.File(path, "r") as f:
            n_unique_samples = f["sample_embeddings"].shape[1]
            n_unique_drugs = f["single_drug_embeddings"].shape[1]
            n_grid = f["single_drug_embeddings"].shape[2] + 1
            n_embedding_dimensions = f["sample_embeddings"].shape[3]
            n_sigma_embedding_dimensions = f["sample_sigma_embeddings"].shape[3]
            n_samples = f["sample_embeddings"].shape[0]

            results = GridComboResults(
                n_unique_samples=n_unique_samples,
                n_unique_drugs=n_unique_drugs,
                n_grid=n_grid,
                n_embedding_dimensions=n_embedding_dimensions,
                n_sigma_embedding_dimensions=n_sigma_embedding_dimensions,
                n_thetas=n_samples,
            )

            results.sample_embeddings = f["sample_embeddings"][:]
            results.single_drug_embeddings = f["single_drug_embeddings"][:]
            results.combo_drug_embeddings = f["combo_drug_embeddings"][:]
            results.sample_sigma_embeddings = f["sample_sigma_embeddings"][:]
            results.single_drug_sigma_embeddings = f["single_drug_sigma_embeddings"][:]
            results.combo_drug_sigma_embeddings = f["combo_drug_sigma_embeddings"][:]
            results.mean_obs_sigma = f["mean_obs_sigma"][:]
            results._cursor = f.attrs["cursor"]

        return results


class ComboGridFactorModel(BayesianModel):
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
        self.conc_grid = ConcentrationGrid(
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

    def predict_grid(
        self, sample_embeddings: torch.FloatTensor, drug_embeddings: torch.FloatTensor
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
        self,
        sample_embeddings: torch.FloatTensor,
        single_drug_embeddings: torch.FloatTensor,
        ub_idx: torch.LongTensor,
        lin_p: torch.FloatTensor,
    ):
        single_pred_grid = self.predict_grid(sample_embeddings, single_drug_embeddings)
        single_effect = (
            lin_p * single_pred_grid.gather(1, ub_idx).squeeze()
            + (1.0 - lin_p) * single_pred_grid.gather(1, (ub_idx - 1)).squeeze()
        )
        return single_effect

    def predict_synergy(
        self,
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
        self,
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
        single_effect_1 = self.predict_single(
            sample_embeddings=sample_embeddings,
            single_drug_embeddings=single_drug_embeddings_1,
            ub_idx=ub_idx_1,
            lin_p=lin_p_1,
        )

        single_effect_2 = self.predict_single(
            sample_embeddings=sample_embeddings,
            single_drug_embeddings=single_drug_embeddings_2,
            ub_idx=ub_idx_2,
            lin_p=lin_p_2,
        )

        synergy_effect = self.predict_synergy(
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
        self,
        sample_sigma_embeddings: torch.FloatTensor,
        single_drug_sigma_embeddings_1: torch.FloatTensor,
        single_drug_sigma_embeddings_2: torch.FloatTensor,
        combo_drug_sigma_embeddings_1: torch.FloatTensor,
        combo_drug_sigma_embeddings_2: torch.FloatTensor,
        mu: torch.FloatTensor,
        mean_obs_sigma: torch.FloatTensor,
        obs_sigma: float,
        single_mask: torch.BoolTensor,
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

        k_upper, k_lower, p = interp_01_vals(mu, self.n_grid)
        mean_level_sigma = (
            p * mean_obs_sigma[k_upper] + (1 - p) * mean_obs_sigma[k_lower]
        )
        sigma = sigma * mean_level_sigma
        return sigma

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
                torch.zeros(self.n_samples, self.var_dims), sample_sigma_embed_sigma
            ).to_event(2),
        )

        ## Scale factor for variance drug embeddings
        single_drug_sigma_embed_sigma = pyro.sample(
            "single_drug_sigma_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        single_drug_sigma_embed = pyro.sample(
            "single_drug_sigma_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.var_dims), single_drug_sigma_embed_sigma
            ).to_event(2),
        )

        combo_drug_sigma_embed_sigma = pyro.sample(
            "combo_drug_sigma_embed_sigma", dist.Gamma(1.0, 1.0)
        )
        combo_drug_sigma_embed = pyro.sample(
            "combo_drug_sigma_embed",
            dist.Normal(
                torch.zeros(self.n_drugs, self.var_dims), combo_drug_sigma_embed_sigma
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
            mu = self.mean_combo(
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

            sigma = self.scale_combo(
                sample_sigma_embeddings=sample_sigma_embed[sample_ids],
                single_drug_sigma_embeddings_1=single_drug_sigma_embed[drug_ids_1],
                single_drug_sigma_embeddings_2=single_drug_sigma_embed[drug_ids_2],
                combo_drug_sigma_embeddings_1=combo_drug_sigma_embed[drug_ids_1],
                combo_drug_sigma_embeddings_2=combo_drug_sigma_embed[drug_ids_2],
                obs_sigma=obs_sigma,
                mu=mu,
                mean_obs_sigma=mean_obs_sigma,
                single_mask=single_mask,
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

    def set_model_state(self, parameters: GridComboSample):
        self.model_state = parameters

    def unpack_data(self, data: ScreenBase, use_mask: bool = True):
        if use_mask:
            mask = data.observation_mask
        else:
            mask = np.ones_like(data.sample_ids, dtype=np.bool_)

        sample_ids = data.sample_ids[mask]

        with np.errstate(divide="ignore"):
            log_conc1 = np.log10(data.treatment_doses[mask, 0])
            log_conc2 = np.log10(data.treatment_doses[mask, 1])

        drugname2idx = self.drugname2idx.copy()
        drugname2idx[data.control_treatment_name] = -1
        drug_ids_1 = np.array(
            [drugname2idx[drugname] for drugname in data.treatment_names[mask, 0]]
        )
        drug_ids_2 = np.array(
            [drugname2idx[drugname] for drugname in data.treatment_names[mask, 1]]
        )

        ## Make sure all 0-dose drugs are correctly labeled as control
        drug_ids_1[log_conc1 == -np.inf] = -1
        drug_ids_2[log_conc2 == -np.inf] = -1

        ## Rearrange so that single drugs only occur in drug_ids_2
        mask = drug_ids_1 < 0
        drug_ids_1[mask] = drug_ids_2[mask].copy()
        drug_ids_2[mask] = -1
        log_conc1[mask] = log_conc2[mask].copy()

        return (sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2)

    def predict(self, data: ScreenBase) -> ArrayType:
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = self.unpack_data(
            data=data, use_mask=False
        )

        ## Organize data
        sample_ids = torch.from_numpy(sample_ids).long()
        drug_ids_1 = torch.from_numpy(drug_ids_1).long()
        drug_ids_2 = torch.from_numpy(drug_ids_2).long()

        log_conc1 = torch.from_numpy(log_conc1).float()
        log_conc2 = torch.from_numpy(log_conc1).float()

        ub_idx_1, lin_p_1 = self.conc_grid.lookup_conc(
            log_concs=log_conc1, drug_ids=self.drug_ids_1
        )
        ub_idx_2, lin_p_2 = self.conc_grid.lookup_conc(
            log_concs=log_conc2, drug_ids=self.drug_ids_2
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
            n_grid=self.n_grid,
            batch_size=self.batch_size,
            n_epochs=1,
            shuffle=False,
            combo_smooth=False,
        )

        mus = []
        for s_ids, d_ids_1, d_ids_2, u_idx_1, u_idx_2, l_p_1, l_p_2 in batch_iterator:
            samp = self.model_state.sample_embeddings[s_ids]
            sdrug1 = self.model_state.single_drug_embeddings[d_ids_1]
            sdrug2 = self.model_state.single_drug_embeddings[d_ids_2]

            cdrug1 = self.model_state.combo_drug_embeddings[d_ids_1]
            cdrug2 = self.model_state.combo_drug_embeddings[d_ids_2]

            single_mask = d_ids_2 < 0

            mu = self.mean_combo(
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

        return torch.concat(mus)

    def variance(self, data: ScreenBase) -> ArrayType:
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = self.unpack_data(
            data=data, use_mask=False
        )

        ## Organize data
        sample_ids = torch.from_numpy(sample_ids).long()
        drug_ids_1 = torch.from_numpy(drug_ids_1).long()
        drug_ids_2 = torch.from_numpy(drug_ids_2).long()

        log_conc1 = torch.from_numpy(log_conc1).float()
        log_conc2 = torch.from_numpy(log_conc1).float()

        ub_idx_1, lin_p_1 = self.conc_grid.lookup_conc(
            log_concs=log_conc1, drug_ids=self.drug_ids_1
        )
        ub_idx_2, lin_p_2 = self.conc_grid.lookup_conc(
            log_concs=log_conc2, drug_ids=self.drug_ids_2
        )

        mu = self.predict(data)

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
            samp = self.model_state.sample_sigma_embeddings[s_ids]
            sdrug1 = self.model_state.single_drug_sigma_embeddings[d_ids_1]
            sdrug2 = self.model_state.single_drug_sigma_embeddings[d_ids_2]

            cdrug1 = self.model_state.combo_drug_sigma_embeddings[d_ids_1]
            cdrug2 = self.model_state.combo_drug_sigma_embeddings[d_ids_2]

            single_mask = d_ids_2 < 0

            sigma = self.scale_combo(
                sample_sigma_embeddings=samp,
                single_drug_sigma_embeddings_1=sdrug1,
                single_drug_sigma_embeddings_2=sdrug2,
                combo_drug_sigma_embeddings_1=cdrug1,
                combo_drug_sigma_embeddings_2=cdrug2,
                obs_sigma=self.model_state.obs_sigma,
                mu=m,
                mean_obs_sigma=self.model_state.mean_obs_sigma,
                single_mask=single_mask,
            )

            sigmas.append(sigma)

        var = torch.square(torch.concat(var))
        return var

    def add_observations(self, data: ScreenBase):
        if not (data.observations >= 0.0).all():
            raise ValueError(
                "Observations should be non-negative, please check input data"
            )
        sample_ids, drug_ids_1, drug_ids_2, log_conc1, log_conc2 = self.unpack_data(
            data=data, use_mask=True
        )

        self.sample_ids = np.concatenate([self.sample_ids, sample_ids])

        self.log_conc1 = np.concatenate([self.log_concs_1, log_conc1])
        self.log_conc2 = np.concatenate([self.log_concs_2, log_conc2])

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

        log_conc1 = torch.from_numpy(self.log_conc1).float()
        log_conc2 = torch.from_numpy(self.log_conc1).float()

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

    def sample_parameters(self, num_samples: int) -> list[GridComboSample]:
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
                sample_embeddings=sample_embed[i].detact().clone(),
                single_drug_embeddings=single_drug_embed[i].detact().clone(),
                combo_drug_embeddings=combo_drug_embed[i].detact().clone(),
                sample_sigma_embeddings=sample_sigma_embed[i].detact().clone(),
                single_drug_sigma_embeddings=single_drug_sigma_embed[i]
                .detact()
                .clone(),
                combo_drug_sigma_embeddings=combo_drug_sigma_embed[i].detact().clone(),
                mean_obs_sigma=mean_obs_sigma[i].detact().clone(),
                obs_sigma=obs_sigma[i].detact().clone(),
            )
            samples.append(x)
        return samples
