import sys
sys.path.append('../ucu-recommender-system-2023/')

from surprise import SVD, NMF

from data_uploader.uploader import DataUploader


class MatrixFactorization:
    """
    Matrix Factorization with gradient descent optimization used as independent algorithms for recommendations
    There are 3 factorization approaches:
    SVD - Singular Value Decomposition
    PMF - Probabilistic Matrix Factorization
    NMF - Non-negative Matrix Factorization
    """

    def __init__(self):
        data_uploader = DataUploader()
        _, self.train_set, self.test_set = data_uploader.get_user_item_data_surprise()

    def fit(self, factorization='svd'):
        if factorization == 'svd':
            algo = SVD(biased=True)
        elif factorization == 'pmf':
            algo = SVD(biased=False)
        elif factorization == 'nmf':
            algo = NMF()

        algo.fit(self.train_set)
        return algo

    def predict_on_testset(self, model):
        return model.test(self.test_set)
