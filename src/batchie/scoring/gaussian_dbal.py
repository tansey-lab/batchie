import numpy as np
from scipy.special import logsumexp, comb
from batchie.common import ArrayType
from batchie.core import Scorer, BayesianModel, DistanceMatrix, SamplesHolder
from batchie.data import Data


def generate_combination_at_sorted_index(index, n, k):
    """
    Generate all range(n) choose k combinations.

    Represent each combination as a descending sorted tuple.

    Sort all the tuples is ascending order, and return the tuple that would be found at `index`.

    Do this without materializing the actual list of combinations.
    """
    n_ck = 1
    for n_minus_i, i_plus_1 in zip(range(n, n - k, -1), range(1, k + 1)):
        n_ck *= n_minus_i
        n_ck //= i_plus_1
    current_index = n_ck
    for k in range(k, 0, -1):
        n_ck *= k
        n_ck //= n
        while current_index - n_ck > index:
            current_index -= n_ck
            n_ck *= n - k
            n_ck -= n_ck % k
            n -= 1
            n_ck //= n
        n -= 1
        yield n


def get_combination_at_sorted_index(index, n, k):
    return tuple(generate_combination_at_sorted_index(index, n, k))


## mean_preds: n posterior samples x m data points x d vec/plate size
## dists: n x n
## max_triples: parameter determining how many triples we look at
def dbal_fast_gauss_scoring_vec(
    mean_preds: ArrayType,
    variances: ArrayType,
    dists: ArrayType,
    rng: np.random.Generator,
    max_combos: int = 5000,
    dfactor: float = 1.0,
):
    n, m, d = mean_preds.shape
    ncombs = comb(n, 3, exact=True)
    n_combos = min(ncombs, max_combos)
    unpacked_indices = rng.choice(ncombs, size=n_combos, replace=False)
    idx1, idx2, idx3 = zip(
        *[get_combination_at_sorted_index(ind, n, 3) for ind in unpacked_indices]
    )
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    idx3 = np.array(idx3)

    with np.errstate(divide="ignore"):
        log_triple_dists = dfactor * np.log(
            dists[idx1, idx2] + dists[idx2, idx3] + dists[idx1, idx3]
        )

    alpha = (
        variances[idx1] * variances[idx2]
        + variances[idx2] * variances[idx3]
        + variances[idx1] * variances[idx3]
    )
    exp_factor = (
        0.5 * (variances[idx1] * variances[idx2] * variances[idx3]) / np.square(alpha)
    )
    log_norm_factor = 0.5 * d * np.log(1.0 / alpha)
    d12 = np.sum(np.square(mean_preds[idx1, :, :] - mean_preds[idx2, :, :]), axis=-1)
    d13 = np.sum(np.square(mean_preds[idx1, :, :] - mean_preds[idx3, :, :]), axis=-1)
    d23 = np.sum(np.square(mean_preds[idx2, :, :] - mean_preds[idx3, :, :]), axis=-1)
    ll = -exp_factor[:, np.newaxis] * (
        variances[idx3, np.newaxis] * d12
        + variances[idx2, np.newaxis] * d13
        + variances[idx1, np.newaxis] * d23
    )  ## n_triples x m
    scores = logsumexp(
        log_norm_factor[:, np.newaxis] + ll + log_triple_dists[:, np.newaxis], axis=0
    )

    return scores


class GaussianDBALScorer(Scorer):
    def _score(
        self,
        data: Data,
        distance_matrix: DistanceMatrix,
        samples: SamplesHolder,
        rng: np.random.Generator,
    ):
        variances = np.array([m.variance() for m in samples.predictor_list()])
        combo_plates = {
            key: plate.combine(prev_plates) for key, plate in plates.items()
        }
        plate_keys = list(combo_plates.keys())
        n_plates = len(plate_keys)

        n_subs = np.ceil(n_plates / self.max_chunk)
        index_arrs = np.array_split(np.arange(n_plates), n_subs)
        scores = dict()
        for idx_arr in tqdm(index_arrs, disable=disable):
            curr_plate_keys = [plate_keys[i] for i in idx_arr]
            current_plates = {key: combo_plates[key] for key in curr_plate_keys}

            mean_preds = samples.predict_plates(current_plates)
            vals = dbal_fast_gauss_scoring_vec(
                mean_preds=mean_preds,
                variances=variances,
                dists=dists,
                max_triples=self.max_triples,
            )
            scores.update(dict(zip(current_plates.keys(), vals)))
        assert len(scores) == n_plates, "Need to assign a score to every plate!"
        return scores
