import numpy as np
import math
import torch


def interp_01_vals(x: torch.FloatTensor, n_grid: int):
    """
    Inputs:
        - x: tensor<float> whose entries are assumed to lie in [0,1]
        - n_grid: int determining size of grid
    Outputs:
        - k_upper: tensor<long> the value of k such that (k-1)/n_grid <= x <= k/n_grid.
        - k_lower: tensor<long> equal to max(k_upper-1,0)
        - p: tensor<float> satisfies that x = p*k/n_grid + (1-p)*(k-1)/n_grid
    """
    ## Mean-level contribution
    k_upper = torch.ceil(x * (n_grid - 1))
    g_k = k_upper / (n_grid - 1)
    g_kmin1 = (k_upper - 1) / (n_grid - 1)
    p = (x - g_kmin1) / (g_k - g_kmin1)
    k_lower = torch.clip(k_upper - 1, min=0)
    return (k_upper.long(), k_lower.long(), p)


class BatchIterator:
    def __init__(
        self,
        *args,
        batch_size: int,
        n_epochs: int,
        n_grid: int,
        min_steps: int = 1,
        max_steps: int = None,
        shuffle: bool = False,
        combo_smooth: bool = False,
    ):
        self.args = args
        self.n = len(self.args[0])
        self.batch_size = batch_size

        n_steps = n_epochs * math.ceil(self.n / batch_size)
        n_steps = max(n_steps, min_steps)
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        self.total_steps = n_steps
        self.shuffle = shuffle
        self.curr_epoch = 0
        self.n_grid = n_grid
        self.combo_smooth = combo_smooth

    def __iter__(self):
        self.step = 0
        self.index = 0
        if self.shuffle:
            self.ordering = torch.randperm(self.n)
        else:
            self.ordering = torch.arange(self.n)

        conc_combo_idxs = np.random.choice(
            int((self.n_grid) ** 2), size=int(2 * self.n), replace=True
        )
        self.conc_combo_idxs = (
            torch.from_numpy(conc_combo_idxs).long().reshape(self.n, 2)
        )
        return self

    def __next__(self):
        if self.step < self.total_steps:
            if self.index >= self.n:
                self.curr_epoch += 1
                self.index = 0
                if self.shuffle:
                    self.ordering = torch.randperm(self.n)
                conc_combo_idxs = np.random.choice(
                    int((self.n_grid) ** 2), size=int(2 * self.n), replace=True
                )
                self.conc_combo_idxs = (
                    torch.from_numpy(conc_combo_idxs).long().reshape(self.n, 2)
                )
            idx = self.ordering[self.index : (self.index + self.batch_size)]
            self.index += self.batch_size
            self.step += 1
            curr = [x[idx] for x in self.args]
            if self.combo_smooth:
                curr.append(self.conc_combo_idxs[idx])
            return curr
        else:
            raise StopIteration

    def __len__(self):
        return self.total_steps

    def just_passed_epoch(self):
        return self.index >= self.n


class ConcentrationGrid:
    def __init__(
        self,
        log_conc_range: dict | tuple,
        drugname2idx: dict,
        n_grid: int,
        log_conc_padding: float = 2.0,
    ):
        n_drugs = len(log_conc_range)
        if isinstance(log_conc_range, (list, tuple)) and (len(log_conc_range) == 2):
            lower, upper = log_conc_range
            min_conc = {i: lower for i in range(n_drugs)}
            max_conc = {i: upper for i in range(n_drugs)}
        elif isinstance(log_conc_range, dict) and (len(drugname2idx) == n_drugs):
            min_conc = {
                drugname2idx[drugname]: lower
                for drugname, (lower, upper) in log_conc_range.items()
            }
            max_conc = {
                drugname2idx[drugname]: upper
                for drugname, (lower, upper) in log_conc_range.items()
            }
        else:
            raise ValueError(
                "log_conc_range must be dict or pair. drugname2idx must be dict. keys of log_conc_range/drugname2idx must match."
            )

        conc_grid = np.linspace(
            [min_conc[drug] for drug in range(n_drugs)],
            [max_conc[drug] for drug in range(n_drugs)],
            num=(n_grid - 2),
            axis=1,
        )

        ## Append padding values to conc_grid
        min_v = conc_grid[:, 0] - log_conc_padding
        max_v = conc_grid[:, -1] + log_conc_padding
        self.conc_grid = torch.from_numpy(
            np.concatenate(
                [min_v[:, np.newaxis], conc_grid, max_v[:, np.newaxis]], axis=1
            )
        ).float()

    def lookup_conc(self, log_concs: torch.FloatTensor, drug_ids: torch.LongTensor):
        ## Look up where log concentration falls within grid
        drug_grid = self.conc_grid[drug_ids]

        ## drug_grid[ub_idx[i]] >= log_concs[i] (except on upper boundary)
        ub_idx = torch.clip(
            torch.searchsorted(drug_grid, log_concs.unsqueeze(1)),
            min=0,
            max=(drug_grid.shape[1]),
        )

        ## drug_grid[lb_idx[i]] <= log_concs[i] (except on lower boundary)
        lb_idx = torch.clip(ub_idx - 1, min=0)

        ## upper-bounding value
        ub = drug_grid.gather(1, ub_idx).squeeze()

        ## lower-bounding value
        lb = drug_grid.gather(1, lb_idx).squeeze()

        ## Linear interpolation coefficient on upper bounding value
        # lin_p = (log_concs-lb)/(ub - lb)
        lin_p = torch.clip(
            torch.where(ub == lb, 1.0, (log_concs - lb) / (ub - lb)), 0.0, 1.0
        )

        return (ub_idx, lin_p)

    def grid_scale(self):
        diff = self.conc_grid[:, 1:] - self.conc_grid[:, :-1]
        return diff.mean(dim=-1)
