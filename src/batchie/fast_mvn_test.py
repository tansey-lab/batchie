from batchie import fast_mvn
import numpy as np


def test_sample_mvn_from_precision():
    Q = np.array([[1, 0.4], [0.4, 1]])

    x = fast_mvn.sample_mvn_from_precision(Q, chol_factor=False)

    assert x.shape == (2,)
