import sys
sys.path.append('../ucu-recommender-system-2023/')

from surprise.prediction_algorithms.knns import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import dump

from models.base import BaseClass


class CollaborativeFiltering(BaseClass):
    """
    A user-user & item-item Collaborative filtering algorithm

    With 3 KNN modifications:
    basic - a basic collaborative filtering algorithm;
    means - -||- taking into account the mean ratings of each user;
    z-score - taking into account the z-score normalization of each user.

    And 3 similarity measures:
    cosine, msd (Mean Squared Difference) and pearson
    """

    def __init__(self, data_uploader):
        super().__init__(data_uploader)

    def fit(self, n_neighbours: int = 5, knn_modification="basic", user_based=True,
            sim_measure='cosine', save=True, file_name=None):
        sim_options = {
            "name": sim_measure,
            "user_based": user_based,  # compute similarities between users or items
        }

        if knn_modification == 'basic':
            self.model = KNNBasic(k=n_neighbours, sim_options=sim_options)
        elif knn_modification == 'means':
            self.model = KNNWithMeans(k=n_neighbours, sim_options=sim_options)
        elif knn_modification == 'z-score':
            self.model = KNNWithZScore(k=n_neighbours, sim_options=sim_options)

        print('---Training start...---')
        self.model.fit(self.train_set)
        print('---Training finished!---')

        if save:
            self.save_model(file_name)

