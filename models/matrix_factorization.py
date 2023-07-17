import sys
sys.path.append('../ucu-recommender-system-2023/')

from surprise import SVD, NMF

from models.base import BaseClass


class MatrixFactorization(BaseClass):
    """
    Matrix Factorization with gradient descent optimization used as independent algorithms for recommendations
    There are 3 factorization approaches:
    SVD - Singular Value Decomposition
    PMF - Probabilistic Matrix Factorization
    NMF - Non-negative Matrix Factorization
    """

    def __init__(self):
        super().__init__()

    def fit(self, factorization='svd'):
        if factorization == 'svd':
            self.model = SVD(biased=True)
        elif factorization == 'pmf':
            self.model = SVD(biased=False)
        elif factorization == 'nmf':
            self.model = NMF()

        self.model.fit(self.train_set)
