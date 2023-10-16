from batchie.interfaces import DistanceMetric
from scipy.special import expit, comb
from scipy.spatial.distance import cdist, pdist, squareform
from batchie.models.sparse_combo import SparseDrugComboMCMCSample


class MSEDistance(DistanceMetric):
    def __init__(self, sigmoid: bool = True, max_chunk: int = 15000, nproc=4, **kwargs):
        super().__init__(**kwargs)
        self.sigmoid = sigmoid
        self.max_chunk = max_chunk
        self.nproc = nproc

    def distance(self, a: SparseDrugComboMCMCSample, b: SparseDrugComboMCMCSample):
        self.model.predict()

    def square_form(self, pred_holder: ComboPredictorHolder) -> np.ndarray:
        start_time = time()
        total_size = pred_holder.total_size
        n_predictors = len(pred_holder.pred_list)

        nsubs = int(np.ceil(total_size / self.max_chunk))

        parallelpdistrun = delayed(parallel_pdist)
        jobs = (
            parallelpdistrun(
                pred_holder,
                start=(i * self.max_chunk),
                end=min((i + 1) * self.max_chunk, total_size),
                sigmoid=self.sigmoid,
            )
            for i in range(nsubs)
        )
        result = prun(jobs, self.nproc)

        p_size = int(comb(n_predictors, 2, exact=True))
        p_dists = np.zeros(p_size)
        for p_dists_i in result:
            p_dists += p_dists_i

        p_dists = squareform(p_dists / total_size)  ## Normalize so that it's MSE

        end_time = time()
        logger.info(
            f"\tParallel distance computation took {end_time - start_time} seconds"
        )
        return p_dists

    def one_v_rest(
        self, u: ComboPredictor, pred_holder: ComboPredictorHolder
    ) -> np.ndarray:
        predictions = pred_holder.all_predictions()
        u_preds = u.predict(pred_holder.total_plate)
        if self.sigmoid:
            predictions = expit(predictions)
            u_preds = expit(u_preds)
        _, total_size = predictions.shape
        dists = np.squeeze(
            cdist(predictions, u_preds[:, np.newaxis], metric="sqeuclidean")
        )
        dists = dists / total_size
        return dists
